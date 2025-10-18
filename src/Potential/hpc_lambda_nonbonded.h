// -*-c++-*-
#ifndef STORMM_HPC_LAMBDA_NONBONDED_H
#define STORMM_HPC_LAMBDA_NONBONDED_H

#include "copyright.h"
#include "DataTypes/common_types.h"
#include "Topology/atomgraph_enumerators.h"

// Forward declarations
namespace stormm {
namespace synthesis {
  class ImplicitSolventWorkspace;
}
}

namespace stormm {
namespace energy {

using topology::UnitCellType;

/// \brief GPU-accelerated lambda-scaled nonbonded energy evaluation for GCMC.
///
/// This kernel evaluates nonbonded interactions (electrostatics and van der Waals)
/// with per-atom lambda coupling parameters. It uses a simplified O(N_coupled × N) approach
/// that's much faster on GPU than the CPU nested loop, while being simpler to maintain
/// than adapting the complex tile-based nonbonded kernel.
///
/// \param n_atoms           Total number of atoms in the system
/// \param n_coupled         Number of coupled atoms (lambda > threshold)
/// \param coupled_indices   Indices of coupled atoms (device array, size n_coupled)
/// \param xcrd              X coordinates (device array, size n_atoms)
/// \param ycrd              Y coordinates (device array, size n_atoms)
/// \param zcrd              Z coordinates (device array, size n_atoms)
/// \param charges           Partial charges (device array, size n_atoms)
/// \param lambda_vdw        Per-atom VDW lambda (device array, size n_atoms)
/// \param lambda_ele        Per-atom electrostatic lambda (device array, size n_atoms)
/// \param atom_sigma        Per-atom LJ sigma (device array, size n_atoms)
/// \param atom_epsilon      Per-atom LJ epsilon (device array, size n_atoms)
/// \param exclusion_mask    Exclusion mask data (device array)
/// \param supertile_map     Supertile map indices (device array)
/// \param tile_map          Tile map indices (device array)
/// \param supertile_stride  Stride for supertile indexing
/// \param umat              Unit cell transformation matrix (device array, 9 elements)
/// \param unit_cell         Unit cell type (NONE, ORTHORHOMBIC, TRICLINIC)
/// \param coulomb_const     Coulomb constant
/// \param ewald_coeff       Ewald coefficient for PME direct space (0.0 for cutoff Coulomb)
/// \param output_elec       Output array for electrostatic energy (device array, size n_coupled)
/// \param output_vdw        Output array for VDW energy (device array, size n_coupled)
/// \param xfrc              Output force array X component (device array, size n_atoms, nullptr for energy-only)
/// \param yfrc              Output force array Y component (device array, size n_atoms, nullptr for energy-only)
/// \param zfrc              Output force array Z component (device array, size n_atoms, nullptr for energy-only)
/// \param gb_workspace      Implicit solvent workspace for GB calculations (nullptr = GB disabled)
/// \param gb_model          Implicit solvent model (NONE = GB disabled)
void launchLambdaScaledNonbonded(
    int n_atoms,
    int n_coupled,
    const int* coupled_indices,
    const double* xcrd,
    const double* ycrd,
    const double* zcrd,
    const double* charges,
    const double* lambda_vdw,
    const double* lambda_ele,
    const double* atom_sigma,
    const double* atom_epsilon,
    const uint* exclusion_mask,
    const int* supertile_map,
    const int* tile_map,
    int supertile_stride,
    const double* umat,
    UnitCellType unit_cell,
    double coulomb_const,
    double ewald_coeff,
    double* output_elec,
    double* output_vdw,
    double* xfrc = nullptr,
    double* yfrc = nullptr,
    double* zfrc = nullptr,
    synthesis::ImplicitSolventWorkspace* gb_workspace = nullptr,
    topology::ImplicitSolventModel gb_model = topology::ImplicitSolventModel::NONE);

/// \brief GPU-accelerated lambda-scaled nonbonded energy evaluation with on-device reduction.
///
/// This function combines the nonbonded kernel with GPU reduction to compute scalar total energies
/// without downloading large per-atom arrays. This eliminates the download bottleneck by only
/// transferring 2 scalar values (total elec + total vdw) instead of n_coupled values.
///
/// \param n_atoms           Total number of atoms in the system
/// \param n_coupled         Number of coupled atoms (lambda > threshold)
/// \param coupled_indices   Indices of coupled atoms (device array)
/// \param xcrd              X coordinates (device array)
/// \param ycrd              Y coordinates (device array)
/// \param zcrd              Z coordinates (device array)
/// \param charges           Partial charges (device array)
/// \param lambda_vdw        Per-atom VDW lambda (device array)
/// \param lambda_ele        Per-atom electrostatic lambda (device array)
/// \param atom_sigma        Per-atom LJ sigma (device array)
/// \param atom_epsilon      Per-atom LJ epsilon (device array)
/// \param exclusion_mask    Exclusion mask data (device array)
/// \param supertile_map     Supertile map indices (device array)
/// \param tile_map          Tile map indices (device array)
/// \param supertile_stride  Stride for supertile indexing
/// \param umat              Unit cell transformation matrix (device array)
/// \param unit_cell         Unit cell type
/// \param coulomb_const     Coulomb constant
/// \param ewald_coeff       Ewald coefficient for PME direct space (0.0 for cutoff Coulomb)
/// \param per_atom_elec     Intermediate per-atom elec energies (device array, size n_coupled)
/// \param per_atom_vdw      Intermediate per-atom vdw energies (device array, size n_coupled)
/// \param total_elec_out    Output scalar total elec energy (device pointer to single double)
/// \param total_vdw_out     Output scalar total vdw energy (device pointer to single double)
/// \param xfrc              Output force array X component (device array, size n_atoms, nullptr for energy-only)
/// \param yfrc              Output force array Y component (device array, size n_atoms, nullptr for energy-only)
/// \param zfrc              Output force array Z component (device array, size n_atoms, nullptr for energy-only)
/// \param gb_workspace      Implicit solvent workspace for GB calculations (nullptr = GB disabled)
/// \param gb_model          Implicit solvent model (NONE = GB disabled)
void launchLambdaScaledNonbondedWithReduction(
    int n_atoms,
    int n_coupled,
    const int* coupled_indices,
    const double* xcrd,
    const double* ycrd,
    const double* zcrd,
    const double* charges,
    const double* lambda_vdw,
    const double* lambda_ele,
    const double* atom_sigma,
    const double* atom_epsilon,
    const uint* exclusion_mask,
    const int* supertile_map,
    const int* tile_map,
    int supertile_stride,
    const double* umat,
    UnitCellType unit_cell,
    double coulomb_const,
    double ewald_coeff,
    double* per_atom_elec,
    double* per_atom_vdw,
    double* total_elec_out,
    double* total_vdw_out,
    double* xfrc = nullptr,
    double* yfrc = nullptr,
    double* zfrc = nullptr,
    synthesis::ImplicitSolventWorkspace* gb_workspace = nullptr,
    topology::ImplicitSolventModel gb_model = topology::ImplicitSolventModel::NONE);

/// \brief GPU kernel to accumulate NCMC work delta on device.
///
/// Computes work += (E_after - E_before) entirely on GPU, eliminating download.
/// This is called once per NCMC perturbation step (50 times per insertion/deletion).
///
/// \param elec_before   Energy before lambda change - electrostatic (device scalar)
/// \param vdw_before    Energy before lambda change - VDW (device scalar)
/// \param elec_after    Energy after lambda change - electrostatic (device scalar)
/// \param vdw_after     Energy after lambda change - VDW (device scalar)
/// \param work_accumulator  Running sum of work (device scalar, accumulated via atomicAdd)
void launchAccumulateWorkDelta(
    const double* elec_before,
    const double* vdw_before,
    const double* elec_after,
    const double* vdw_after,
    double* work_accumulator);

/// \brief GPU kernel to update per-atom lambda values from NCMC schedule.
///
/// Updates lambda values for a specific molecule based on the NCMC step index.
/// This eliminates CPU-side lambda updates and uploads (100 per NCMC move).
/// Implements two-stage coupling: VDW first, then electrostatics.
///
/// \param step_index              Current NCMC step index (0 to n_steps)
/// \param lambda_schedule         Array of lambda values for each step (device array, size n_steps+1)
/// \param molecule_indices        Atom indices for the target molecule (device array)
/// \param n_molecule_atoms        Number of atoms in the molecule
/// \param vdw_coupling_threshold  Lambda value where VDW is fully coupled (typically 0.75)
/// \param lambda_vdw              Per-atom VDW lambda array to update (device array)
/// \param lambda_ele              Per-atom electrostatic lambda array to update (device array)
void launchUpdateLambdaFromSchedule(
    int step_index,
    const double* lambda_schedule,
    const int* molecule_indices,
    int n_molecule_atoms,
    double vdw_coupling_threshold,
    double* lambda_vdw,
    double* lambda_ele);

/// \brief GPU-accelerated lambda-aware Born radii computation for GB calculations.
///
/// Computes Born radii for all atoms in the system. No lambda scaling is applied
/// as Born radii are calculated for all atoms regardless of coupling state.
///
/// \param n_atoms         Total number of atoms
/// \param xcrd            X coordinates (device array)
/// \param ycrd            Y coordinates (device array)
/// \param zcrd            Z coordinates (device array)
/// \param pb_radii        Perfect Born radii from topology (device array)
/// \param gb_screen       Screening parameters (device array)
/// \param gb_offset       GB offset parameter
/// \param psi             Output psi values for Born radii calculation (device array)
/// \param born_radii      Output Born radii (device array)
/// \param gb_workspace    Implicit solvent workspace
/// \param gb_model        Implicit solvent model
void launchLambdaBornRadii(
    int n_atoms,
    const double* xcrd,
    const double* ycrd,
    const double* zcrd,
    const double* pb_radii,
    const double* gb_screen,
    double gb_offset,
    double* psi,
    double* born_radii,
    synthesis::ImplicitSolventWorkspace* gb_workspace,
    topology::ImplicitSolventModel gb_model);

/// \brief GPU-accelerated lambda-aware Born derivative force computation.
///
/// Adds GB derivative forces to the force arrays WITH lambda scaling for proper GCMC physics.
/// This kernel processes only coupled atoms to minimize computation.
///
/// \param n_atoms         Total number of atoms
/// \param n_coupled       Number of coupled atoms
/// \param coupled_indices Indices of coupled atoms (device array)
/// \param xcrd            X coordinates (device array)
/// \param ycrd            Y coordinates (device array)
/// \param zcrd            Z coordinates (device array)
/// \param charges         Partial charges (device array)
/// \param lambda_ele      Electrostatic lambda values (device array)
/// \param born_radii      Born radii (device array)
/// \param sum_deijda      Derivative of GB energy w.r.t. Born radii (device array)
/// \param gb_offset       GB offset parameter
/// \param coulomb_const   Coulomb constant
/// \param xfrc            Output force X component (device array)
/// \param yfrc            Output force Y component (device array)
/// \param zfrc            Output force Z component (device array)
/// \param gb_workspace    Implicit solvent workspace
/// \param gb_model        Implicit solvent model
void launchLambdaBornDerivatives(
    int n_atoms,
    int n_coupled,
    const int* coupled_indices,
    const double* xcrd,
    const double* ycrd,
    const double* zcrd,
    const double* charges,
    const double* lambda_ele,
    const double* born_radii,
    const double* sum_deijda,
    double gb_offset,
    double coulomb_const,
    double* xfrc,
    double* yfrc,
    double* zfrc,
    synthesis::ImplicitSolventWorkspace* gb_workspace,
    topology::ImplicitSolventModel gb_model);

} // namespace energy
} // namespace stormm

#endif // STORMM_HPC_LAMBDA_NONBONDED_H
