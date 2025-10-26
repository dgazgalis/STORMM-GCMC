// -*-c++-*-
#include "copyright.h"
#include <algorithm>
#include <iostream>
#include <memory>
#include "Accelerator/hybrid.h"
#include "Constants/behavior.h"
#include "Math/series_ops.h"
#include "Potential/scorecard.h"
#include "Reporting/error_format.h"
#include "Synthesis/atomgraph_synthesis.h"
#include "Synthesis/phasespace_synthesis.h"
#include "Synthesis/static_mask_synthesis.h"
#include "Trajectory/phasespace.h"
#include "hpc_dynamics.h"
#include "hpc_lambda_dynamics.h"
#include "hpc_lambda_dynamics_wrapper.h"

namespace stormm {
namespace mm {

using card::Hybrid;
using card::HybridTargetLevel;
using constants::PrecisionModel;
using energy::CacheResource;
using energy::ClashResponse;
using energy::EvaluateEnergy;
using energy::EvaluateForce;
using energy::ScoreCard;
using numerics::AccumulationMethod;
using stmath::incrementingSeries;
using errors::rtErr;
using errors::rtWarn;
using synthesis::AtomGraphSynthesis;
using synthesis::ImplicitSolventWorkspace;
using synthesis::maximum_valence_work_unit_atoms;
using synthesis::PhaseSpaceSynthesis;
using synthesis::PsSynthesisWriter;
using synthesis::SeMaskSynthesisReader;
using synthesis::small_block_max_atoms;
using synthesis::StaticExclusionMaskSynthesis;
using synthesis::NbwuKind;
using synthesis::SyNonbondedKit;
using synthesis::VwuGoal;
using topology::AtomGraph;
using topology::ImplicitSolventModel;
using topology::UnitCellType;
using trajectory::PhaseSpace;
using trajectory::PhaseSpaceWriter;

//-------------------------------------------------------------------------------------------------
ScoreCard launchLambdaDynamics(const AtomGraphSynthesis &poly_ag,
                               const StaticExclusionMaskSynthesis &poly_se,
                               Thermostat *tst,
                               PhaseSpaceSynthesis *poly_ps,
                               const DynamicsControls &dyncon,
                               const SystemCache &sysc,
                               const SynthesisCacheMap &syscmap,
                               const GpuDetails &gpu,
                               PrecisionModel valence_prec,
                               PrecisionModel nonbond_prec,
                               int energy_bits,
                               StopWatch *timer,
                               const std::string &task_name) {

  // ========================================================================
  // LAMBDA DYNAMICS PARITY TESTING
  // ========================================================================
  // This function runs lambda-aware dynamics with all lambda values set to 1.0
  // to verify that launchGpuLambdaDynamicsStep produces identical results
  // to standard dynamics when all particles are fully coupled.
  //
  // KEY LIMITATION: Currently supports single-system only due to type mismatch
  // between PsSynthesisWriter (llint*) and launchGpuLambdaDynamicsStep (double*)
  // ========================================================================

  const HybridTargetLevel devc_tier = HybridTargetLevel::DEVICE;

  // Extract dynamics parameters
  const int nstep = dyncon.getStepCount();
  const int ntpr = dyncon.getDiagnosticPrintFrequency();
  const int ntwx = dyncon.getTrajectoryPrintFrequency();
  const double dt = dyncon.getTimeStep();

  // Verify single system (required for parity testing)
  const int n_systems = poly_ag.getSystemCount();
  if (n_systems != 1) {
    rtErr("Lambda dynamics parity testing currently supports single system only.\n"
          "Multi-system support requires handling PsSynthesisWriter fixed-precision (llint*).\n"
          "For multi-system lambda dynamics, use the GCMC sampler directly.",
          "launchLambdaDynamics");
  }

  // Get atom count for first (and only) system
  const int n_atoms = poly_ag.getAtomCount(0);

  rtWarn("launchLambdaDynamics: Running lambda-aware dynamics for parity testing.\n"
         "All lambda values set to 1.0 (fully coupled).\n"
         "Results should match dynamics.stormm.cuda within numerical precision.",
         task_name.c_str());

  // ========================================================================
  // SETUP PHASE: Initialize lambda arrays (all 1.0 for parity mode)
  // ========================================================================

  Hybrid<double> lambda_vdw(n_atoms, "lambda_vdw_parity");
  Hybrid<double> lambda_ele(n_atoms, "lambda_ele_parity");
  Hybrid<double> atom_sigma(n_atoms, "atom_sigma");
  Hybrid<double> atom_epsilon(n_atoms, "atom_epsilon");

  // Get nonbonded kit for charge array
  const SyNonbondedKit<double, double2> nbk =
      poly_ag.getDoublePrecisionNonbondedKit(devc_tier);

  // Initialize all lambda values to 1.0 (fully coupled)
  // Initialize sigma/epsilon to dummy values (kernel gets real values from topology)
  for (int i = 0; i < n_atoms; i++) {
    lambda_vdw.putHost(1.0, i);
    lambda_ele.putHost(1.0, i);
    atom_sigma.putHost(1.0, i);      // Dummy value
    atom_epsilon.putHost(1.0, i);    // Dummy value
  }

  // Build coupled atom indices (all atoms coupled in parity mode)
  Hybrid<int> coupled_indices(n_atoms, "coupled_indices");
  for (int i = 0; i < n_atoms; i++) {
    coupled_indices.putHost(i, i);
  }
  const int n_coupled = n_atoms;

  // ========================================================================
  // UPLOAD PHASE: Send data to GPU
  // ========================================================================

  lambda_vdw.upload();
  lambda_ele.upload();
  coupled_indices.upload();
  atom_sigma.upload();
  atom_epsilon.upload();

  // Upload coordinates and topology
  poly_ps->upload();
  // poly_ag and poly_se should already be uploaded by caller

  // ========================================================================
  // GET DEVICE POINTERS FOR LAMBDA PARAMETERS
  // ========================================================================
  // The new launchLambdaDynamicsStep interface handles all coordinate/force
  // extraction internally. We only need to pass lambda arrays.

  // Lambda arrays
  const double* d_lambda_vdw = lambda_vdw.data(devc_tier);
  const double* d_lambda_ele = lambda_ele.data(devc_tier);
  const int* d_coupled_indices = coupled_indices.data(devc_tier);

  // ========================================================================
  // SETUP GPU INFRASTRUCTURE
  // ========================================================================

  // Create kernel launcher
  const CoreKlManager launcher(gpu, poly_ag);

  // Create MM controls for integration
  MolecularMechanicsControls mmctrl(dt, nstep);
  mmctrl.primeWorkUnitCounters(
      launcher,
      EvaluateForce::YES,
      EvaluateEnergy::NO,  // Will override on energy steps
      ClashResponse::NONE,
      VwuGoal::MOVE_PARTICLES,  // CRITICAL: Full integration mode
      valence_prec,
      nonbond_prec,
      poly_ag);

  // Create cache resources for thread blocks
  const int2 vale_lp = launcher.getValenceKernelDims(
      valence_prec,
      EvaluateForce::YES,
      EvaluateEnergy::NO,
      AccumulationMethod::SPLIT,
      VwuGoal::MOVE_PARTICLES,
      ClashResponse::NONE);
  static std::unique_ptr<CacheResource> valence_cache_ptr;
  static int cached_valence_blocks = 0;
  static int cached_valence_atoms = 0;
  const int required_valence_blocks = vale_lp.x;
  const int required_valence_atoms = maximum_valence_work_unit_atoms;
  if (valence_cache_ptr == nullptr ||
      cached_valence_blocks < required_valence_blocks ||
      cached_valence_atoms < required_valence_atoms) {
    cached_valence_blocks = std::max(cached_valence_blocks, required_valence_blocks);
    cached_valence_atoms = std::max(cached_valence_atoms, required_valence_atoms);
    valence_cache_ptr = std::make_unique<CacheResource>(
        cached_valence_blocks, cached_valence_atoms);
    std::cout << "# DEBUG hpc_lambda_dynamics: allocated valence CacheResource blocks="
              << cached_valence_blocks << " atoms=" << cached_valence_atoms << std::endl;
  }
  CacheResource* valence_cache = valence_cache_ptr.get();

  const NbwuKind nb_work_type = poly_ag.getNonbondedWorkType();
  const ImplicitSolventModel gb_model = poly_ag.getImplicitSolventModel();
  const int2 nonb_lp = launcher.getNonbondedKernelDims(
      nonbond_prec,
      nb_work_type,
      EvaluateForce::YES,
      EvaluateEnergy::NO,
      AccumulationMethod::SPLIT,
      gb_model,
      ClashResponse::NONE);
  static std::unique_ptr<CacheResource> nonb_cache_ptr;
  static int cached_nonb_blocks = 0;
  static int cached_nonb_atoms = 0;
  const int required_nonb_blocks = nonb_lp.x;
  const int required_nonb_atoms = small_block_max_atoms;
  if (nonb_cache_ptr == nullptr ||
      cached_nonb_blocks < required_nonb_blocks ||
      cached_nonb_atoms < required_nonb_atoms) {
    cached_nonb_blocks = std::max(cached_nonb_blocks, required_nonb_blocks);
    cached_nonb_atoms = std::max(cached_nonb_atoms, required_nonb_atoms);
    nonb_cache_ptr = std::make_unique<CacheResource>(
        cached_nonb_blocks, cached_nonb_atoms);
    std::cout << "# DEBUG hpc_lambda_dynamics: allocated nonbonded CacheResource blocks="
              << cached_nonb_blocks << " atoms=" << cached_nonb_atoms << std::endl;
  }
  CacheResource* nonb_cache = nonb_cache_ptr.get();

  // Create ScoreCard for energy tracking
  ScoreCard sc(n_systems, ((nstep + ntpr - 1) / ntpr) + 1, energy_bits);
  // Note: upload() is unnecessary - Hybrid<> constructor already allocates GPU memory

  // ========================================================================
  // MAIN DYNAMICS LOOP
  // ========================================================================

  for (int step_idx = 0; step_idx < nstep; step_idx++) {
    // Determine if this is an energy evaluation step
    const bool on_energy_step = (step_idx % ntpr == 0) || (step_idx == nstep - 1);
    const EvaluateEnergy eval_energy = on_energy_step ?
        EvaluateEnergy::YES : EvaluateEnergy::NO;

    // Initialize energy accumulators
    if (on_energy_step) {
      sc.initialize(devc_tier, gpu);
    }

    // ====================================================================
    // CALL NEW CLEAN LAMBDA DYNAMICS INTERFACE
    // ====================================================================
    // This uses the new launchLambdaDynamicsStep which:
    // - Accepts PhaseSpaceSynthesis* directly (like standard dynamics)
    // - Extracts coordinates internally
    // - Handles all memory management consistently

    launchLambdaDynamicsStep(
        d_lambda_vdw,               // Per-atom VDW lambda (all 1.0 for parity)
        d_lambda_ele,               // Per-atom electrostatic lambda (all 1.0 for parity)
        d_coupled_indices,          // Indices of coupled atoms
        n_coupled,                  // Number of coupled atoms (all atoms in parity mode)
        poly_ag,                    // Topology synthesis
        poly_se,                    // Static exclusion mask synthesis
        tst,                        // Thermostat (nullptr for NVE)
        poly_ps,                    // Phase space synthesis (coordinates, velocities, forces)
        &mmctrl,                    // MM controls (timestep, counters, integration mode)
        &sc,                        // ScoreCard - always pass (eval_energy flag controls computation)
        launcher,                   // Kernel launch manager
        valence_cache,              // Valence cache resource
        nonb_cache,                 // Nonbonded cache resource
        eval_energy,                // Energy evaluation flag (controls whether energies computed)
        nullptr,                    // GB workspace (nullptr = GB disabled)
        ImplicitSolventModel::NONE); // GB model (NONE for now)

    // Increment step counter
    mmctrl.incrementStep();
    if (tst != nullptr) {
      tst->incrementStep();
    }

    // Finalize energy tracking
    if (on_energy_step) {
      sc.commit(devc_tier, gpu);
      sc.incrementSampleCount();

      // DEBUG: Check energies after commit
      printf("DEBUG wrapper: Step %d, after commit - checking ScoreCard\n", step_idx);
    }

    // Output trajectory (if requested)
    if (ntwx > 0 && (step_idx + 1) % ntwx == 0) {
      poly_ps->download();
      const double current_time = static_cast<double>(step_idx + 1) * dt;

      const int sysc_idx = syscmap.getSystemCacheIndex(0);
      const std::string& traj_name = sysc.getTrajectoryName(sysc_idx);

      if (traj_name.size() > 0) {
        // Export system 0 from synthesis
        PhaseSpace ps_export = poly_ps->exportSystem(0);
        ps_export.exportToFile(traj_name, current_time,
                              trajectory::TrajectoryKind::POSITIONS,
                              trajectory::CoordinateFileKind::AMBER_CRD,
                              diskutil::PrintSituation::APPEND);
      }

      poly_ps->upload();  // Re-upload for next iteration
    }

    // Progress reporting (every 100 steps)
    if (timer != nullptr && step_idx % 100 == 0 && step_idx > 0) {
      // Timer updates handled by caller
    }
  }

  // ========================================================================
  // CLEANUP: Compute final energies and download
  // ========================================================================

  sc.computePotentialEnergy(devc_tier, gpu);
  sc.computeTotalEnergy(devc_tier, gpu);
  sc.download();

  // Download final coordinates
  poly_ps->download();

  return sc;
}

} // namespace mm
} // namespace stormm
