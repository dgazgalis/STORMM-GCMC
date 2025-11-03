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
/// OPTIMIZATION: This function no longer performs cudaDeviceSynchronize() or downloads
/// the exact count. Instead, it returns a conservative upper bound based on the
/// previous count plus a buffer. This eliminates ~0.5-1ms overhead per energy evaluation.
///
/// \param n_atoms              Total number of atoms in the system
/// \param d_lambda_vdw         Device pointer to per-atom VDW lambda array
/// \param d_lambda_ele         Device pointer to per-atom electrostatic lambda array
/// \param lambda_threshold     Threshold for considering an atom coupled (typically 0.01)
/// \param d_coupled_indices    Device pointer to output coupled indices array (modified in-place)
/// \param h_n_coupled_out      Host pointer to receive the number of coupled atoms (output, approximate)
/// \param previous_count       Previous coupled atom count (used as hint for conservative bound)
void launchRebuildCoupledIndices(
    int n_atoms,
    const double* d_lambda_vdw,
    const double* d_lambda_ele,
    double lambda_threshold,
    int* d_coupled_indices,
    int* h_n_coupled_out,
    int previous_count = 0);

/// \brief Extract total energy from ScoreCard on GPU
///
/// This kernel reads all energy components from the ScoreCard and sums them to produce
/// a single total energy value. This is used as part of the GPU-resident Metropolis
/// acceptance workflow to avoid downloading the full ScoreCard.
///
/// \param sc_writer        ScoreCard writer abstract (device pointer)
/// \param d_total_energy   Device pointer to store total energy (size=1)
void launchExtractTotalEnergy(
    const energy::ScoreCardWriter& sc_writer,
    double* d_total_energy);

/// \brief GPU-resident Metropolis acceptance for GCMC insertion/deletion
///
/// This kernel performs the Metropolis acceptance decision entirely on GPU:
/// 1. Reads E_initial and E_final from device memory
/// 2. Computes delta_E
/// 3. Calculates acceptance probability using GCMC formula
/// 4. Generates random number and makes accept/reject decision
/// 5. Writes result (0=reject, 1=accept) to device memory
///
/// This eliminates 2× full ScoreCard downloads (~2-4ms) and replaces with
/// 1× integer download (~0.01ms), providing ~2-4ms speedup per GCMC cycle.
///
/// \param d_E_initial          Device pointer to initial energy
/// \param d_E_final            Device pointer to final energy
/// \param B                    Adams B parameter (chemical potential related)
/// \param beta                 1 / (k_B * T)
/// \param N_active             Current number of active molecules
/// \param rng_states           Device pointer to cuRAND states (void* to avoid CUDA dependency in C++)
/// \param d_acceptance_result  Device pointer to result (0=reject, 1=accept)
void launchMetropolisAcceptance(
    const double* d_E_initial,
    const double* d_E_final,
    double B,
    double beta,
    int N_active,
    void* rng_states,
    int* d_acceptance_result);

/// \brief GPU-resident Metropolis acceptance decision for GCMC DELETION
///
/// Computes acceptance probability for GCMC deletion:
/// P_acc = min(1, N * exp(-B - beta*delta_E))
/// Eliminates ~16KB ScoreCard downloads by evaluating directly on GPU, downloading only
/// 1× integer download (~0.01ms), providing ~2-4ms speedup per GCMC cycle.
///
/// \param d_E_initial          Device pointer to initial energy (molecule active)
/// \param d_E_final            Device pointer to final energy (molecule as ghost)
/// \param B                    Adams B parameter (chemical potential related)
/// \param beta                 1 / (k_B * T)
/// \param N_active             Current number of active molecules
/// \param rng_states           Device pointer to cuRAND states (void* to avoid CUDA dependency in C++)
/// \param d_acceptance_result  Device pointer to result (0=reject, 1=accept)
void launchMetropolisAcceptanceDeletion(
    const double* d_E_initial,
    const double* d_E_final,
    double B,
    double beta,
    int N_active,
    void* rng_states,
    int* d_acceptance_result);

#endif // STORMM_USE_HPC

} // namespace sampling
} // namespace stormm

#endif // STORMM_HPC_GCMC_LAMBDA_H
