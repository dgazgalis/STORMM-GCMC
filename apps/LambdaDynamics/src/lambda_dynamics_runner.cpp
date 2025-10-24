// Minimal lambda dynamics test runner for parity testing with standard MD
// Uses all lambda=1.0 to verify dynamics produces identical results to regular MD

#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <cstring>
#include "copyright.h"

// Core includes
#include "Accelerator/gpu_details.h"
#include "Accelerator/core_kernel_manager.h"
#include "Accelerator/hybrid.h"
#include "DataTypes/common_types.h"
#include "Reporting/error_format.h"

// Topology and trajectory
#include "Topology/atomgraph.h"
#include "Trajectory/phasespace.h"
#include "Trajectory/thermostat.h"

// Synthesis (GPU abstractions)
#include "Synthesis/atomgraph_synthesis.h"
#include "Synthesis/phasespace_synthesis.h"
#include "Synthesis/static_mask_synthesis.h"
#include "Synthesis/nonbonded_workunit.h"  // For small_block_max_atoms

// Potential
#include "Potential/static_exclusionmask.h"
#include "Potential/scorecard.h"
#include "Potential/cacheresource.h"

// Molecular mechanics
#include "MolecularMechanics/mm_controls.h"
#include "MolecularMechanics/hpc_lambda_dynamics.h"

// Numerics
#include "Numerics/numeric_enumerators.h"

#ifdef STORMM_USE_CUDA
#include <cuda_runtime.h>
#endif

using namespace stormm;
using namespace stormm::card;
using namespace stormm::data_types;
using namespace stormm::energy;
using namespace stormm::errors;
using namespace stormm::mm;
using namespace stormm::numerics;
using namespace stormm::synthesis;
using namespace stormm::topology;
using namespace stormm::trajectory;

int main(int argc, char* argv[]) {

  // Simple argument parsing
  if (argc < 3) {
    std::cout << "Usage: " << argv[0] << " <topology.prmtop> <coords.inpcrd> [n_steps] [lambda]" << std::endl;
    std::cout << "Example: " << argv[0] << " benzene.prmtop benzene.inpcrd 10000 0.5" << std::endl;
    std::cout << "         lambda=1.0 (default) for parity testing, 0.0-1.0 for lambda scaling tests" << std::endl;
    return 1;
  }

  const std::string prmtop_file = argv[1];
  const std::string inpcrd_file = argv[2];
  const int n_steps = (argc > 3) ? std::atoi(argv[3]) : 10000;
  const double lambda_value = (argc > 4) ? std::atof(argv[4]) : 1.0;
  const double timestep = 0.001;  // 1 fs in ps units

  std::cout << "Lambda Dynamics Test" << std::endl;
  std::cout << "Topology: " << prmtop_file << std::endl;
  std::cout << "Coordinates: " << inpcrd_file << std::endl;
  std::cout << "Steps: " << n_steps << std::endl;
  std::cout << "Timestep: " << timestep << " ps" << std::endl;
  std::cout << "Lambda: " << lambda_value << std::endl;
  if (lambda_value == 1.0) {
    std::cout << "Mode: Parity testing (lambda=1.0, should match standard dynamics)" << std::endl;
  } else {
    std::cout << "Mode: Lambda scaling test" << std::endl;
  }

#ifndef STORMM_USE_CUDA
  std::cout << "ERROR: This test requires CUDA support. Recompile with CUDA enabled." << std::endl;
  return 1;
#endif

#ifdef STORMM_USE_CUDA

  try {
    // Load topology
    AtomGraph topology(prmtop_file, ExceptionResponse::SILENT);
    const int n_atoms = topology.getAtomCount();
    std::cout << "Loaded " << n_atoms << " atoms" << std::endl;

    // Load coordinates
    PhaseSpace ps(inpcrd_file, topology);

    // Create static exclusion mask
    StaticExclusionMask exclusions(&topology);

    // Get nonbonded kit for setting up lambda arrays
    const NonbondedKit<double> nbk = topology.getDoublePrecisionNonbondedKit();

    // Initialize lambda arrays with specified lambda value
    Hybrid<double> lambda_vdw(n_atoms, "lambda_vdw");
    Hybrid<double> lambda_ele(n_atoms, "lambda_ele");
    Hybrid<double> atom_sigma(n_atoms, "atom_sigma");
    Hybrid<double> atom_epsilon(n_atoms, "atom_epsilon");

    // Set all lambda values to the specified value
    for (int i = 0; i < n_atoms; i++) {
      lambda_vdw.putHost(lambda_value, i);
      lambda_ele.putHost(lambda_value, i);

      // Get LJ parameters from topology
      const int lj_idx = nbk.lj_idx[i];
      atom_sigma.putHost(nbk.lj_sigma[lj_idx], i);
      atom_epsilon.putHost(nbk.lja_coeff[lj_idx], i);  // lja_coeff contains epsilon in STORMM
    }

    // All atoms are "coupled" in parity mode
    Hybrid<int> coupled_indices(n_atoms, "coupled_indices");
    for (int i = 0; i < n_atoms; i++) {
      coupled_indices.putHost(i, i);
    }

    // Upload arrays to GPU
    lambda_vdw.upload();
    lambda_ele.upload();
    atom_sigma.upload();
    atom_epsilon.upload();
    coupled_indices.upload();

    // Create synthesis objects
    // CRITICAL: Create PhaseSpaceSynthesis on heap with proper lifetime management
    // The synthesis object must outlive all GPU kernel calls
    std::vector<AtomGraph*> ag_list = {&topology};
    std::vector<PhaseSpace> ps_vec = {ps};  // PhaseSpaceSynthesis wants vector of objects, not pointers
    AtomGraphSynthesis ag_synthesis(ag_list);
    PhaseSpaceSynthesis* ps_synthesis = new PhaseSpaceSynthesis(ps_vec, ag_list);

    // Create static exclusion mask synthesis
    std::vector<StaticExclusionMask*> mask_list = {&exclusions};
    std::vector<int> topology_indices = {0};
    StaticExclusionMaskSynthesis poly_se(mask_list, topology_indices);

    // Load nonbonded work units (required before creating CoreKlManager)
    ag_synthesis.loadNonbondedWorkUnits(poly_se);

    // Create score card for energy tracking
    ScoreCard scorecard(1);  // 1 system
    scorecard.upload();  // Upload to GPU before use

    // Create a thermostat (required even for NVE)
    // Use ThermostatKind::NONE for NVE dynamics (no temperature control)
    Thermostat thermostat(topology, ThermostatKind::NONE, 300.0);
    Thermostat* thermostat_ptr = &thermostat;

    // Get GPU details and create kernel launcher
    const GpuDetails gpu;
    const CoreKlManager launcher(gpu, ag_synthesis);

    // Create MM controls
    MolecularMechanicsControls mmctrl(timestep, n_steps);
    mmctrl.primeWorkUnitCounters(
        launcher,
        EvaluateForce::YES,
        EvaluateEnergy::NO,
        ClashResponse::NONE,
        VwuGoal::MOVE_PARTICLES,  // Full integration mode
        PrecisionModel::DOUBLE,
        PrecisionModel::DOUBLE,
        ag_synthesis);

    // Create cache resources for kernel thread blocks
    const PrecisionModel valence_prec = PrecisionModel::DOUBLE;
    const int2 vale_lp = launcher.getValenceKernelDims(
        valence_prec,
        EvaluateForce::YES,
        EvaluateEnergy::NO,
        AccumulationMethod::SPLIT,
        VwuGoal::MOVE_PARTICLES,
        ClashResponse::NONE);
    CacheResource valence_cache(vale_lp.x, maximum_valence_work_unit_atoms);

    // Get nonbonded launch parameters
    const int2 nonb_lp = launcher.getNonbondedKernelDims(
        PrecisionModel::DOUBLE,
        ag_synthesis.getNonbondedWorkType(),
        EvaluateForce::YES,
        EvaluateEnergy::NO,
        AccumulationMethod::SPLIT,
        ImplicitSolventModel::NONE,
        ClashResponse::NONE);
    CacheResource nonb_cache(nonb_lp.x, small_block_max_atoms);

    // Get device pointers
    const int* d_coupled_indices = coupled_indices.data(HybridTargetLevel::DEVICE);
    const double* d_lambda_vdw = lambda_vdw.data(HybridTargetLevel::DEVICE);
    const double* d_lambda_ele = lambda_ele.data(HybridTargetLevel::DEVICE);
    const double* d_atom_sigma = atom_sigma.data(HybridTargetLevel::DEVICE);
    const double* d_atom_epsilon = atom_epsilon.data(HybridTargetLevel::DEVICE);

    // Get exclusion mask info
    const StaticExclusionMaskReader ser = exclusions.data();
    const uint* d_exclusion_mask = ser.mask_data;
    const int* d_supertile_map = ser.supertile_map_idx;
    const int* d_tile_map = ser.tile_map_idx;
    const int supertile_stride = ser.supertile_stride_count;

    // Upload topology and PhaseSpace data to GPU
    // CRITICAL: Must upload the topology before running dynamics!
    // The synthesis contains copies made during construction, but we use
    // raw pointers from the original objects for the kernel
    topology.upload();
    ps.upload();

    // Get device pointers from the original PhaseSpace
    PhaseSpaceWriter psw = ps.data();
    double* d_xcrd = psw.xcrd;
    double* d_ycrd = psw.ycrd;
    double* d_zcrd = psw.zcrd;
    double* d_xfrc = psw.xfrc;
    double* d_yfrc = psw.yfrc;
    double* d_zfrc = psw.zfrc;

    // Unit cell info
    const double* d_umat = psw.umat;
    const UnitCellType unit_cell = psw.unit_cell;

    // Energy output arrays (optional, can be nullptr)
    Hybrid<double> energy_output_elec(n_atoms);
    Hybrid<double> energy_output_vdw(n_atoms);
    double* d_per_atom_elec = energy_output_elec.data(HybridTargetLevel::DEVICE);
    double* d_per_atom_vdw = energy_output_vdw.data(HybridTargetLevel::DEVICE);

    // Ewald coefficient (0.0 for cutoff, non-zero for PME direct space)
    double ewald_coeff = 0.0;
    if (unit_cell != UnitCellType::NONE) {
      // Use standard Ewald coefficient for PME direct space
      // Typical value is ~0.35 for 8 Angstrom cutoff
      ewald_coeff = 0.35;
    }

    std::cout << "\nStarting dynamics..." << std::endl;

    // Time the dynamics loop
    auto start_time = std::chrono::high_resolution_clock::now();

    // Main dynamics loop
    for (int step = 0; step < n_steps; step++) {
      // Enable energy evaluation on first step, every 1000 steps, and last step
      const bool on_energy_step = (step == 0) || (step % 1000 == 0) || (step == n_steps - 1);
      const EvaluateEnergy eval_energy = on_energy_step ?
          EvaluateEnergy::YES : EvaluateEnergy::NO;

      // Launch lambda dynamics step using new clean API
      // NOTE: Changed from deprecated launchGpuLambdaDynamicsStep to launchLambdaDynamicsStep
      // The new API takes high-level STORMM objects and handles all GPU pointer extraction internally
      launchLambdaDynamicsStep(
          d_lambda_vdw,           // Per-atom VDW lambda (all 1.0 for parity)
          d_lambda_ele,           // Per-atom electrostatic lambda (all 1.0 for parity)
          d_coupled_indices,      // Indices of coupled atoms
          n_atoms,                // Number of coupled atoms (all atoms in parity mode)
          ag_synthesis,           // Topology synthesis (const reference)
          poly_se,                // Static exclusion mask synthesis (const reference)
          thermostat_ptr,         // Thermostat (nullptr for NVE)
          ps_synthesis,           // Phase space synthesis pointer (coordinates, velocities, forces)
          &mmctrl,                // MM controls (timestep, counters, integration mode)
          &scorecard,             // ScoreCard - always pass (eval_energy flag controls computation)
          launcher,               // Kernel launch manager (const reference)
          &valence_cache,         // Valence cache resource
          &nonb_cache,            // Nonbonded cache resource
          eval_energy,            // Energy evaluation flag (controls whether energies computed)
          nullptr,                // GB workspace (nullptr = GB disabled)
          ImplicitSolventModel::NONE); // GB model (NONE for now)

      // Increment step counter
      mmctrl.incrementStep();

      // Print progress every 1000 steps
      if (step % 1000 == 0 && step > 0) {
        std::cout << "Step " << step << "/" << n_steps << std::endl;
      }
    }

    // Synchronize GPU
    cudaDeviceSynchronize();

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    // Calculate timing
    double total_time_ms = duration_ms.count();
    double time_per_step_ms = total_time_ms / n_steps;

    // Download energies from GPU to host
    scorecard.download();

    // Get final energy if available
    double total_energy = 0.0;
    if (scorecard.getSystemCount() > 0) {
      total_energy = scorecard.reportTotalEnergy(0);
    }

    // Print results
    std::cout << "\n=== Results ===" << std::endl;
    std::cout << "Total time: " << std::fixed << std::setprecision(2)
              << total_time_ms << " ms" << std::endl;
    std::cout << "Time per step: " << std::fixed << std::setprecision(2)
              << time_per_step_ms << " ms" << std::endl;
    std::cout << "Total energy: " << std::fixed << std::setprecision(2)
              << total_energy << " kcal/mol" << std::endl;

    // Clean up heap-allocated synthesis object
    delete ps_synthesis;

  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

#endif // STORMM_USE_CUDA

  return 0;
}