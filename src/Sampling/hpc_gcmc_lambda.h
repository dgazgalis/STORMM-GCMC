// -*-c++-*-
#ifndef STORMM_HPC_GCMC_LAMBDA_H
#define STORMM_HPC_GCMC_LAMBDA_H

#include "copyright.h"

namespace stormm {
namespace sampling {

#ifdef STORMM_USE_HPC

/// \brief Launch GPU kernel to update lambda values for a single molecule
///
/// Updates both lambda_vdw and lambda_ele arrays for all atoms in the specified molecule.
/// This eliminates the need to rebuild the entire lambda array on CPU and upload it.
///
/// NOTE: This function includes cudaDeviceSynchronize() to ensure the kernel
/// completes before returning. Modified lambda values are safe to use immediately
/// after this call returns.
///
/// \param n_atoms_in_molecule  Number of atoms in the molecule being modified
/// \param d_atom_indices       Device pointer to molecule's atom indices (size n_atoms_in_molecule)
/// \param new_lambda_vdw       New VDW lambda value [0, 1]
/// \param new_lambda_ele       New electrostatic lambda value [0, 1]
/// \param d_lambda_vdw         Device pointer to per-atom VDW lambda array (modified in-place)
/// \param d_lambda_ele         Device pointer to per-atom electrostatic lambda array (modified in-place)
/// \param n_atoms_total        Total number of atoms in system (for bounds checking)
void launchUpdateMoleculeLambda(
    int n_atoms_in_molecule,
    const int* d_atom_indices,
    double new_lambda_vdw,
    double new_lambda_ele,
    double* d_lambda_vdw,
    double* d_lambda_ele,
    int n_atoms_total);

/// \brief Launch GPU kernel to rebuild the coupled indices array
///
/// Scans all per-atom lambda values and builds a compact list of indices for atoms
/// with lambda > threshold. This is needed after lambda modifications to keep the
/// coupled atoms list up-to-date for energy evaluation.
///
/// NOTE: This implementation uses a simple sequential scan on GPU. For large systems,
/// a parallel scan approach would be more efficient, but for GCMC with ~1200 atoms,
/// the sequential approach is sufficient and simpler to implement.
///
/// NOTE: This function includes cudaDeviceSynchronize() to ensure the kernel
/// completes before returning. The coupled indices array and count are safe to use
/// immediately after this call returns.
///
/// \param n_atoms              Total number of atoms in the system
/// \param d_lambda_vdw         Device pointer to per-atom VDW lambda array
/// \param d_lambda_ele         Device pointer to per-atom electrostatic lambda array
/// \param lambda_threshold     Threshold for considering an atom coupled (typically 0.01)
/// \param d_coupled_indices    Device pointer to output coupled indices array (modified in-place)
/// \param h_n_coupled_out      Host pointer to receive the number of coupled atoms (output)
void launchRebuildCoupledIndices(
    int n_atoms,
    const double* d_lambda_vdw,
    const double* d_lambda_ele,
    double lambda_threshold,
    int* d_coupled_indices,
    int* h_n_coupled_out);

#endif // STORMM_USE_HPC

} // namespace sampling
} // namespace stormm

#endif // STORMM_HPC_GCMC_LAMBDA_H
