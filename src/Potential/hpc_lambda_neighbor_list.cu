// -*-c++-*-
#include "copyright.h"
#include <cuda_runtime.h>
#include "lambda_neighbor_list.h"

namespace stormm {
namespace energy {

//-------------------------------------------------------------------------------------------------
// GPU kernel: Build lambda-aware neighbor list
//
// Each thread handles one coupled atom and finds its neighbors
//-------------------------------------------------------------------------------------------------
__global__ void kBuildLambdaNeighborList(
    const int n_coupled,
    const int* __restrict__ coupled_indices,
    const double* __restrict__ xcrd,
    const double* __restrict__ ycrd,
    const double* __restrict__ zcrd,
    const double cutoff_sq,
    int* __restrict__ neighbor_counts,
    int* __restrict__ neighbor_list,
    const int max_neighbors_capacity)
{
  const int i_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (i_idx >= n_coupled) return;

  const int i = coupled_indices[i_idx];
  const double xi = xcrd[i];
  const double yi = ycrd[i];
  const double zi = zcrd[i];

  int neighbor_count = 0;
  const int base_offset = i_idx * max_neighbors_capacity;

  // Find neighbors among all other coupled atoms
  for (int j_idx = 0; j_idx < n_coupled; j_idx++) {
    if (i_idx == j_idx) continue;  // Skip self

    const int j = coupled_indices[j_idx];
    const double dx = xcrd[j] - xi;
    const double dy = ycrd[j] - yi;
    const double dz = zcrd[j] - zi;

    const double r2 = dx*dx + dy*dy + dz*dz;

    if (r2 < cutoff_sq && neighbor_count < max_neighbors_capacity) {
      neighbor_list[base_offset + neighbor_count] = j_idx;
      neighbor_count++;
    }
  }

  neighbor_counts[i_idx] = neighbor_count;
}

//-------------------------------------------------------------------------------------------------
// GPU kernel: Count maximum neighbors (reduction)
//-------------------------------------------------------------------------------------------------
__global__ void kFindMaxNeighbors(
    const int n_coupled,
    const int* __restrict__ neighbor_counts,
    int* __restrict__ max_neighbors_out)
{
  __shared__ int shared_max[256];

  const int tid = threadIdx.x;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Load data
  int local_max = 0;
  if (idx < n_coupled) {
    local_max = neighbor_counts[idx];
  }

  shared_max[tid] = local_max;
  __syncthreads();

  // Reduction in shared memory
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared_max[tid] = max(shared_max[tid], shared_max[tid + stride]);
    }
    __syncthreads();
  }

  // Write result
  if (tid == 0) {
    atomicMax(max_neighbors_out, shared_max[0]);
  }
}

//-------------------------------------------------------------------------------------------------
// Launch GPU neighbor list construction
//-------------------------------------------------------------------------------------------------
void launchLambdaNeighborListBuild(
    int n_coupled,
    const int* coupled_indices,
    const double* xcrd,
    const double* ycrd,
    const double* zcrd,
    double cutoff_with_skin_sq,
    int* neighbor_counts,
    int* neighbor_list,
    int max_neighbors_capacity,
    int* max_neighbors_out)
{
  if (n_coupled == 0) return;

  // Zero the max neighbors output
  cudaMemset(max_neighbors_out, 0, sizeof(int));

  const int threads_per_block = 256;
  const int num_blocks = (n_coupled + threads_per_block - 1) / threads_per_block;

  // Build neighbor lists
  kBuildLambdaNeighborList<<<num_blocks, threads_per_block>>>(
      n_coupled, coupled_indices,
      xcrd, ycrd, zcrd,
      cutoff_with_skin_sq,
      neighbor_counts, neighbor_list,
      max_neighbors_capacity);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    // Error occurred
  }

  // Find maximum neighbors
  kFindMaxNeighbors<<<num_blocks, threads_per_block>>>(
      n_coupled, neighbor_counts, max_neighbors_out);

  err = cudaGetLastError();
  if (err != cudaSuccess) {
    // Error occurred
  }

  cudaDeviceSynchronize();
}

} // namespace energy
} // namespace stormm
