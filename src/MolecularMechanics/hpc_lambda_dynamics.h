// -*-c++-*-
#ifndef STORMM_HPC_LAMBDA_DYNAMICS_H
#define STORMM_HPC_LAMBDA_DYNAMICS_H

#include "copyright.h"
#include "Topology/atomgraph_enumerators.h"

// Forward declarations
namespace stormm {
namespace card {
  class CoreKlManager;
}
namespace energy {
  class PMIGrid;
  class ScoreCard;
  class CacheResource;
}
namespace mm {
  class MolecularMechanicsControls;
}
namespace trajectory {
  class Thermostat;
}
namespace synthesis {
  class AtomGraphSynthesis;
  class PhaseSpaceSynthesis;
  class ImplicitSolventWorkspace;
  class StaticExclusionMaskSynthesis;
}
}

namespace stormm {
namespace mm {

using topology::UnitCellType;

/// \brief Launch GPU kernel for velocity Verlet velocity update: v(t) -> v(t+dt/2)
///
/// Updates velocities using current forces: v += F * dt / (2m)
///
/// \param n_atoms      Number of atoms
/// \param vx           Velocity X component (device array, size n_atoms)
/// \param vy           Velocity Y component (device array, size n_atoms)
/// \param vz           Velocity Z component (device array, size n_atoms)
/// \param fx           Force X component (device array, size n_atoms)
/// \param fy           Force Y component (device array, size n_atoms)
/// \param fz           Force Z component (device array, size n_atoms)
/// \param masses       Atomic masses (device array, size n_atoms)
/// \param dt_half      Half timestep (dt/2)
void launchVelocityUpdate(
    int n_atoms,
    double* vx,
    double* vy,
    double* vz,
    const double* fx,
    const double* fy,
    const double* fz,
    const double* masses,
    double dt_half);

/// \brief Launch GPU kernel for velocity Verlet coordinate update: x(t) -> x(t+dt)
///
/// Updates coordinates using mid-step velocities: x += v * dt
///
/// \param n_atoms      Number of atoms
/// \param x            Coordinate X component (device array, size n_atoms)
/// \param y            Coordinate Y component (device array, size n_atoms)
/// \param z            Coordinate Z component (device array, size n_atoms)
/// \param vx           Velocity X component (device array, size n_atoms)
/// \param vy           Velocity Y component (device array, size n_atoms)
/// \param vz           Velocity Z component (device array, size n_atoms)
/// \param dt           Full timestep
void launchCoordinateUpdate(
    int n_atoms,
    double* x,
    double* y,
    double* z,
    const double* vx,
    const double* vy,
    const double* vz,
    double dt);

/// \brief Launch GPU kernel to zero force arrays
///
/// \param n_atoms      Number of atoms
/// \param fx           Force X component (device array, size n_atoms)
/// \param fy           Force Y component (device array, size n_atoms)
/// \param fz           Force Z component (device array, size n_atoms)
void launchZeroForces(
    int n_atoms,
    double* fx,
    double* fy,
    double* fz);

/// \brief Complete GPU velocity Verlet MD step with lambda-scaled forces and full integration
///
/// Performs a complete MD cycle using STORMM's MOVE_PARTICLES integration mode,  which
/// automatically handles:
///   - Velocity Verlet integration (both half-steps)
///   - SETTLE constraints (rigid water geometries)
///   - RATTLE constraints (general bond constraints)
///   - Virtual site position updates
///   - Virtual site force redistribution
///   - Thermostat velocity scaling (Andersen, Langevin, etc.)
///
/// Integration strategy:
/// 1. Zero force arrays
/// 2. Compute lambda-scaled nonbonded forces at current positions x(t)
/// 3. Call valence kernel with MOVE_PARTICLES mode, which performs:
///    a. Compute valence forces and add to nonbonded forces
///    b. First velocity half-update: v(t) -> v(t+dt/2) using F(t)
///    c. Coordinate update: x(t) -> x(t+dt) using v(t+dt/2)
///    d. Apply SETTLE/RATTLE constraints to coordinates and velocities
///    e. Update virtual site positions
///    f. Recompute all forces at new positions x(t+dt)
///    g. Second velocity half-update: v(t+dt/2) -> v(t+dt) using F(t+dt)
///
/// NOTE: Uses Ewald direct space only (erfc splitting). Full PME reciprocal space is not yet
///       implemented in STORMM. This provides reasonable electrostatics for most systems.
///
/// LAMBDA SCALING STRATEGY:
///   - Nonbonded forces (VDW + electrostatics) ARE lambda-scaled for GCMC sampling
///   - Valence forces (bonds, angles, dihedrals) are NOT lambda-scaled
///   - This is correct GCMC physics: ghost molecules must maintain proper geometry, and
///     intramolecular energies cancel in GCMC acceptance criteria
///
/// All operations occur on GPU with no CPU-GPU transfers.
///
/// \param n_atoms           Total number of atoms
/// \param n_coupled         Number of coupled atoms (lambda > threshold)
/// \param coupled_indices   Indices of coupled atoms (device array)
/// \param lambda_vdw        Per-atom VDW lambda (device array, size n_atoms)
/// \param lambda_ele        Per-atom electrostatic lambda (device array, size n_atoms)
/// \param atom_sigma        Per-atom LJ sigma (device array, size n_atoms)
/// \param atom_epsilon      Per-atom LJ epsilon (device array, size n_atoms)
/// \param exclusion_mask    Exclusion mask data (device array)
/// \param supertile_map     Supertile map indices (device array)
/// \param tile_map          Tile map indices (device array)
/// \param supertile_stride  Stride for supertile indexing
/// \param coulomb_const     Coulomb constant
/// \param ewald_coeff       Ewald coefficient for PME direct space (0.0 for cutoff Coulomb)
/// \param per_atom_elec     Per-atom elec energies (device array, size n_coupled, can be nullptr)
/// \param per_atom_vdw      Per-atom vdw energies (device array, size n_coupled, can be nullptr)
/// \param poly_ag           Topology synthesis (contains constraints, virtual sites, valence terms)
/// \param poly_ps           Phase space synthesis (contains coordinates, velocities, forces)
/// \param poly_se           Static exclusion mask synthesis for nonbonded interactions
/// \param mmctrl            Molecular mechanics controls (timestep, counters, integration mode)
/// \param thermostat        Thermostat for temperature control (nullptr = NVE, otherwise NVT)
/// \param sc                Score card for energy tracking (nullptr = skip energy tracking)
/// \param launcher          Kernel launch manager for GPU parameters
/// \param valence_cache     Cache resource for valence kernel thread blocks
/// \param nonb_cache        Cache resource for nonbonded kernel thread blocks
/// \param eval_energy       Whether to evaluate energies (YES on diagnostic steps, NO otherwise)
/// \param gb_workspace      Implicit solvent workspace for GB calculations (nullptr = GB disabled)
/// \param gb_model          Implicit solvent model (NONE = GB disabled)
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
    synthesis::AtomGraphSynthesis* poly_ag,
    synthesis::PhaseSpaceSynthesis* poly_ps,
    const synthesis::StaticExclusionMaskSynthesis* poly_se,
    MolecularMechanicsControls* mmctrl,
    trajectory::Thermostat* thermostat,
    energy::ScoreCard* sc,
    const card::CoreKlManager* launcher,
    energy::CacheResource* valence_cache,
    energy::CacheResource* nonb_cache,
    energy::EvaluateEnergy eval_energy,
    synthesis::ImplicitSolventWorkspace* gb_workspace = nullptr,
    topology::ImplicitSolventModel gb_model = topology::ImplicitSolventModel::NONE);


} // namespace mm
} // namespace stormm

#endif // STORMM_HPC_LAMBDA_DYNAMICS_H
