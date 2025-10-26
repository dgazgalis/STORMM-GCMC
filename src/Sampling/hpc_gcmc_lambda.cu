// -*-c++-*-
#include "copyright.h"
#include <cuda_runtime.h>
#include "Accelerator/hybrid.h"
#include "Reporting/error_format.h"
#include "hpc_gcmc_lambda.h"

namespace stormm {
namespace sampling {

using card::Hybrid;
using card::HybridTargetLevel;

//-------------------------------------------------------------------------------------------------
// GPU kernel: Update lambda values for a single molecule
//
// Directly modifies the per-atom lambda arrays on GPU for all atoms in the specified molecule.
// This eliminates the need to rebuild entire lambda arrays on CPU and upload them.
//
// Thread-parallel over atoms in molecule: each thread handles one atom.
//-------------------------------------------------------------------------------------------------
__global__ void kUpdateMoleculeLambda(
    const int n_atoms_in_molecule,
    const int* __restrict__ atom_indices,
    const double new_lambda_vdw,
    const double new_lambda_ele,
    double* __restrict__ lambda_vdw,
    double* __restrict__ lambda_ele,
    const int n_atoms_total)  // FIX: Add total atom count for bounds checking
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_atoms_in_molecule) return;

  const int atom = atom_indices[idx];

  // FIX: Validate atom index is within bounds (safety check)
  if (atom < 0 || atom >= n_atoms_total) return;

  lambda_vdw[atom] = new_lambda_vdw;
  lambda_ele[atom] = new_lambda_ele;
}

//-------------------------------------------------------------------------------------------------
// GPU kernel: Rebuild coupled indices array
//
// Scans all atoms and builds a list of indices for atoms with lambda > threshold.
// Uses atomicAdd to append indices sequentially to the output array.
//
// PERFORMANCE NOTE: For GCMC systems with ~1000-2000 atoms, this simple approach is efficient.
// For much larger systems (>10000 atoms), a two-pass parallel scan or stream compaction
// approach would be more efficient.
//
// Thread-parallel over all atoms: each thread checks one atom and conditionally appends.
//-------------------------------------------------------------------------------------------------
__global__ void kRebuildCoupledIndices(
    const int n_atoms,
    const double* __restrict__ lambda_vdw,
    const double* __restrict__ lambda_ele,
    const double lambda_threshold,
    int* __restrict__ coupled_indices,
    int* __restrict__ n_coupled)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_atoms) return;

  // Check if this atom is coupled (either VDW or electrostatic lambda > threshold)
  if (lambda_vdw[idx] > lambda_threshold || lambda_ele[idx] > lambda_threshold) {
    // Atomically increment the counter and get the position for this atom
    const int pos = atomicAdd(n_coupled, 1);
    coupled_indices[pos] = idx;
  }
}

//-------------------------------------------------------------------------------------------------
// Launch wrapper: Update molecule lambda values
//-------------------------------------------------------------------------------------------------
void launchUpdateMoleculeLambda(
    int n_atoms_in_molecule,
    const int* d_atom_indices,
    double new_lambda_vdw,
    double new_lambda_ele,
    double* d_lambda_vdw,
    double* d_lambda_ele,
    int n_atoms_total)  // FIX: Add total atom count for bounds checking
{
  if (n_atoms_in_molecule == 0) {
    return;  // Nothing to update
  }

  const int threads_per_block = 256;
  const int num_blocks = (n_atoms_in_molecule + threads_per_block - 1) / threads_per_block;

  kUpdateMoleculeLambda<<<num_blocks, threads_per_block>>>(
      n_atoms_in_molecule, d_atom_indices, new_lambda_vdw, new_lambda_ele,
      d_lambda_vdw, d_lambda_ele, n_atoms_total);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    rtErr("CUDA kernel launch failed in launchUpdateMoleculeLambda: " +
          std::string(cudaGetErrorString(err)), "launchUpdateMoleculeLambda");
  }

  // Synchronize to ensure completion before subsequent operations
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    rtErr("CUDA kernel execution failed in launchUpdateMoleculeLambda: " +
          std::string(cudaGetErrorString(err)), "launchUpdateMoleculeLambda");
  }
}

//-------------------------------------------------------------------------------------------------
// Launch wrapper: Rebuild coupled indices array
//-------------------------------------------------------------------------------------------------
void launchRebuildCoupledIndices(
    int n_atoms,
    const double* d_lambda_vdw,
    const double* d_lambda_ele,
    double lambda_threshold,
    int* d_coupled_indices,
    int* h_n_coupled_out)
{
  if (n_atoms == 0) {
    *h_n_coupled_out = 0;
    return;
  }

  // FIX: Use static Hybrid for counter to avoid repeated cudaMalloc/cudaFree (fragmentation)
  // This counter is allocated once per program run instead of on every energy evaluation
  static Hybrid<int> coupled_counter(1, "coupled_counter");
  static bool counter_initialized = false;

  if (!counter_initialized) {
    // First call: allocate GPU memory
    coupled_counter.upload();
    counter_initialized = true;
  }

  // Reset counter to zero on device
  cudaError_t err = cudaMemset(coupled_counter.data(HybridTargetLevel::DEVICE), 0, sizeof(int));
  if (err != cudaSuccess) {
    rtErr("CUDA memset failed in launchRebuildCoupledIndices: " +
          std::string(cudaGetErrorString(err)), "launchRebuildCoupledIndices");
  }

  // Launch kernel
  const int threads_per_block = 256;
  const int num_blocks = (n_atoms + threads_per_block - 1) / threads_per_block;

  kRebuildCoupledIndices<<<num_blocks, threads_per_block>>>(
      n_atoms, d_lambda_vdw, d_lambda_ele, lambda_threshold,
      d_coupled_indices, coupled_counter.data(HybridTargetLevel::DEVICE));

  err = cudaGetLastError();
  if (err != cudaSuccess) {
    rtErr("CUDA kernel launch failed in launchRebuildCoupledIndices: " +
          std::string(cudaGetErrorString(err)), "launchRebuildCoupledIndices");
  }

  // Synchronize to ensure completion
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    rtErr("CUDA kernel execution failed in launchRebuildCoupledIndices: " +
          std::string(cudaGetErrorString(err)), "launchRebuildCoupledIndices");
  }

  // Download result to host
  coupled_counter.download();
  *h_n_coupled_out = coupled_counter.data()[0];
}

} // namespace sampling
} // namespace stormm
