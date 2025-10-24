// -*-c++-*-
#include "copyright.h"
#include <cuda_runtime.h>
#include "Accelerator/hybrid.h"
#include "Potential/hpc_lambda_nonbonded.h"
#include "Potential/hpc_nonbonded_potential.h"
#include "Potential/hpc_valence_potential.h"
#include "Potential/energy_enumerators.h"
#include "Potential/energy_abstracts.h"
#include "Potential/cacheresource.h"
#include "Potential/scorecard.h"
#include "Trajectory/thermostat.h"
#include "Trajectory/trajectory_enumerators.h"
#include "Synthesis/atomgraph_synthesis.h"
#include "Synthesis/phasespace_synthesis.h"
#include "Synthesis/implicit_solvent_workspace.h"
#include "Synthesis/static_mask_synthesis.h"
#include "MolecularMechanics/mm_controls.h"
#include "Numerics/numeric_enumerators.h"
#include "hpc_lambda_dynamics.h"

namespace stormm {
namespace mm {

using card::CoreKlManager;
using card::Hybrid;
using card::HybridTargetLevel;
using energy::EvaluateForce;
using energy::EvaluateEnergy;
using energy::launchNonbonded;
using energy::launchValence;
using energy::NbwuKind;
using energy::PrecisionModel;
using synthesis::AtomGraphSynthesis;
using synthesis::ISWorkspaceKit;
using synthesis::PhaseSpaceSynthesis;
using synthesis::PsSynthesisWriter;
using synthesis::SeMaskSynthesisReader;
using synthesis::SyNonbondedKit;
using synthesis::StaticExclusionMaskSynthesis;
using synthesis::VwuGoal;
using trajectory::CoordinateCycle;


//-------------------------------------------------------------------------------------------------
// GPU kernel: Zero force arrays (llint* fixed-precision version)
//-------------------------------------------------------------------------------------------------
__global__ void kZeroForces(
    const int n_atoms,
    llint* __restrict__ fx,
    llint* __restrict__ fy,
    llint* __restrict__ fz)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_atoms) return;

  fx[idx] = 0LL;
  fy[idx] = 0LL;
  fz[idx] = 0LL;
}

//-------------------------------------------------------------------------------------------------
// Launch wrapper for zeroing forces (llint* version)
//-------------------------------------------------------------------------------------------------
void launchZeroForces(
    int n_atoms,
    llint* fx,
    llint* fy,
    llint* fz)
{
  const int threads_per_block = 256;
  const int num_blocks = (n_atoms + threads_per_block - 1) / threads_per_block;

  kZeroForces<<<num_blocks, threads_per_block>>>(n_atoms, fx, fy, fz);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    // Error occurred - STORMM will handle it
  }
}

//-------------------------------------------------------------------------------------------------
// DEPRECATED: Complete GPU velocity Verlet MD step with lambda-scaled forces and full integration
//
// NOTE: This old function used double* arrays and is no longer used.
// Use launchLambdaDynamicsStep() instead, which works with llint* fixed-precision.
//
// This function performed a complete MD cycle using STORMM's standard MOVE_PARTICLES integration
// mode, which automatically handled:
//   - Velocity Verlet integration (both half-steps)
//   - SETTLE constraints (rigid water geometries)
//   - RATTLE constraints (general bond constraints)
//   - Virtual site position updates
//   - Virtual site force redistribution
//   - Thermostat velocity scaling (Andersen, Langevin, etc.)
//
// The integration strategy is:
//   1. Zero forces
//   2. Compute lambda-scaled nonbonded forces at current positions
//   3. Call valence kernel with MOVE_PARTICLES mode, which does:
//      a. Compute valence forces and add to nonbonded forces
//      b. First velocity half-update: v(t) -> v(t+dt/2)
//      c. Coordinate update: x(t) -> x(t+dt)
//      d. Apply SETTLE/RATTLE constraints to coordinates and velocities
//      e. Update virtual site positions
//      f. Recompute all forces at new positions x(t+dt)
//      g. Second velocity half-update: v(t+dt/2) -> v(t+dt)
//
// NOTE: This computes DIRECT SPACE electrostatics only using erfc(α·r)/r (Ewald splitting).
//       Full PME (reciprocal space via FFT) is not yet implemented in STORMM.
//
// DESIGN NOTE on Lambda Scaling:
//   - Nonbonded forces (VDW + electrostatics) ARE lambda-scaled for GCMC sampling
//   - Valence forces (bonds, angles, dihedrals) are NOT lambda-scaled
//   - This is CORRECT GCMC physics:
//       * Ghost molecules must maintain proper geometry during MD equilibration
//       * Intramolecular energies cancel in GCMC acceptance criteria
//       * Only intermolecular (nonbonded) interactions need lambda-scaling
//-------------------------------------------------------------------------------------------------
/*
void launchGpuLambdaDynamicsStep(
    int n_atoms,
    int n_coupled,
    const int* coupled_indices,
    const double* lambda_vdw,
    const double* lambda_ele,
    const double* atom_sigma,
    const double* atom_epsilon,
    const uint* exclusion_mask,
    const int* supertile_map,
    const int* tile_map,
    int supertile_stride,
    double coulomb_const,
    double ewald_coeff,
    double* per_atom_elec,
    double* per_atom_vdw,
    double* xcrd,
    double* ycrd,
    double* zcrd,
    double* xfrc,
    double* yfrc,
    double* zfrc,
    const double* umat,
    topology::UnitCellType unit_cell,
    const double* charges,
    AtomGraphSynthesis* poly_ag,
    PhaseSpaceSynthesis* poly_ps,
    const StaticExclusionMaskSynthesis* poly_se,
    MolecularMechanicsControls* mmctrl,
    trajectory::Thermostat* thermostat,
    energy::ScoreCard* sc,
    const CoreKlManager* launcher,
    energy::CacheResource* valence_cache,
    energy::CacheResource* nonb_cache,
    EvaluateEnergy eval_energy,
    synthesis::ImplicitSolventWorkspace* gb_workspace,
    topology::ImplicitSolventModel gb_model)
{
  // DEPRECATED - Function body removed
  // Use launchLambdaDynamicsStep() instead
}
*/

//-------------------------------------------------------------------------------------------------
void launchLambdaDynamicsStep(
    const double* lambda_vdw,
    const double* lambda_ele,
    const int* coupled_indices,
    int n_coupled,
    const AtomGraphSynthesis& poly_ag,
    const StaticExclusionMaskSynthesis& poly_se,
    trajectory::Thermostat* tst,
    PhaseSpaceSynthesis* poly_ps,
    MolecularMechanicsControls* mmctrl,
    energy::ScoreCard* sc,
    const CoreKlManager& launcher,
    energy::CacheResource* valence_cache,
    energy::CacheResource* nonb_cache,
    EvaluateEnergy eval_energy,
    synthesis::ImplicitSolventWorkspace* gb_workspace,
    topology::ImplicitSolventModel gb_model)
{
  using card::HybridTargetLevel;
  using synthesis::PsSynthesisWriter;

  const HybridTargetLevel devc_tier = HybridTargetLevel::DEVICE;
  const int n_atoms = poly_ag.getAtomCount();

  // Get current cycle position for coordinate/force access (needed for zeroing forces)
  const CoordinateCycle curr_cyc = poly_ps->getCyclePosition();
  PsSynthesisWriter psw = poly_ps->data(curr_cyc, devc_tier);

  // Zero forces before computing new forces
  const int threads_per_block = 256;
  const int num_blocks = (n_atoms + threads_per_block - 1) / threads_per_block;
  kZeroForces<<<num_blocks, threads_per_block>>>(n_atoms, psw.xfrc, psw.yfrc, psw.zfrc);

  // Call high-level lambda nonbonded launcher (matches standard dynamics pattern)
  // This extracts coordinates/forces internally from PhaseSpaceSynthesis
  energy::launchLambdaNonbonded(
      lambda_vdw,                       // Per-atom VDW lambda (device array)
      lambda_ele,                       // Per-atom electrostatic lambda (device array)
      coupled_indices,                  // Indices of coupled atoms (device array)
      n_coupled,                        // Number of coupled atoms
      constants::PrecisionModel::DOUBLE,  // Precision model
      poly_ag,                          // Topology synthesis
      poly_se,                          // Static exclusion mask synthesis
      mmctrl,                           // MM controls
      poly_ps,                          // Phase space synthesis (coordinates, velocities, forces)
      nullptr,                          // Thermostat (not used in lambda nonbonded)
      sc,                               // Score card for energy tracking
      nonb_cache,                       // Cache resource for nonbonded kernel
      gb_workspace,                     // GB workspace (nullptr if GB disabled)
      EvaluateForce::YES,               // We need forces computed
      eval_energy,                      // Energy evaluation (only on diagnostic steps)
      launcher);                        // Kernel launch manager

  // Call valence kernel to:
  // 1. Compute bonded forces (NOT lambda-scaled - correct for GCMC)
  // 2. Perform full Velocity Verlet integration with constraints
  // 3. Apply thermostat if present
  launchValence(
      PrecisionModel::DOUBLE,           // Match coordinate/force precision
      poly_ag,                           // Topology synthesis
      mmctrl,                            // MM controls (timestep, integration mode)
      poly_ps,                           // Phase space synthesis
      tst,                               // Thermostat (nullptr = NVE)
      sc,                                // Score card for energy tracking
      valence_cache,                     // Thread-block cache resource
      EvaluateForce::YES,                // We need forces computed
      eval_energy,                       // Energy evaluation (only on diagnostic steps)
      VwuGoal::MOVE_PARTICLES,           // CRITICAL: Full integration mode
      launcher,                          // Kernel launch parameters
      0.0,                               // clash_distance (no clash mitigation)
      0.0);                              // clash_ratio (no clash mitigation)
}

} // namespace mm
} // namespace stormm
