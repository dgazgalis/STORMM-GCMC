// -*-c++-*-
#include "copyright.h"
#include <cmath>
#include <algorithm>
#ifdef STORMM_USE_CUDA
#include <cuda_runtime.h>
#include "hpc_lambda_neighbor_list.h"
#endif
#include "lambda_neighbor_list.h"

namespace stormm {
namespace energy {

//-------------------------------------------------------------------------------------------------
LambdaNeighborListReader::LambdaNeighborListReader(int n_coupled_in,
                                                    const int* coupled_indices_in,
                                                    int max_neighbors_in,
                                                    const int* neighbor_counts_in,
                                                    const int* neighbor_list_in,
                                                    double cutoff_in) :
  n_coupled{n_coupled_in},
  coupled_indices{coupled_indices_in},
  max_neighbors{max_neighbors_in},
  neighbor_counts{neighbor_counts_in},
  neighbor_list{neighbor_list_in},
  cutoff{cutoff_in},
  cutoff_squared{cutoff_in * cutoff_in}
{}

//-------------------------------------------------------------------------------------------------
LambdaNeighborList::LambdaNeighborList(int n_atoms_in, const double* lambda_vdw_in,
                                       const double* lambda_ele_in,
                                       double cutoff_in, double skin_in) :
  n_atoms_{n_atoms_in},
  n_coupled_{0},
  cutoff_{cutoff_in},
  skin_{skin_in},
  cutoff_with_skin_{cutoff_in + skin_in},
  cutoff_with_skin_sq_{(cutoff_in + skin_in) * (cutoff_in + skin_in)},
  lambda_vdw_{n_atoms_in, "lambda_vdw"},
  lambda_ele_{n_atoms_in, "lambda_ele"},
  coupled_indices_{n_atoms_in, "coupled_indices"},
  max_neighbors_{0},
  neighbor_counts_{n_atoms_in, "neighbor_counts"},
  neighbor_list_{n_atoms_in * 512, "neighbor_list"},  // Initial allocation
  last_build_xcrd_{n_atoms_in, "last_build_xcrd"},
  last_build_ycrd_{n_atoms_in, "last_build_ycrd"},
  last_build_zcrd_{n_atoms_in, "last_build_zcrd"}
{
  // Copy lambda values to Hybrid arrays
  double* lambda_vdw_ptr = lambda_vdw_.data();
  double* lambda_ele_ptr = lambda_ele_.data();
  for (int i = 0; i < n_atoms_; i++) {
    lambda_vdw_ptr[i] = lambda_vdw_in[i];
    lambda_ele_ptr[i] = lambda_ele_in[i];
  }

  // Identify coupled atoms
  identifyCoupledAtoms();
}

//-------------------------------------------------------------------------------------------------
void LambdaNeighborList::identifyCoupledAtoms() {
  const double* lambda_vdw_ptr = lambda_vdw_.data();
  const double* lambda_ele_ptr = lambda_ele_.data();
  int* coupled_ptr = coupled_indices_.data();

  n_coupled_ = 0;
  for (int i = 0; i < n_atoms_; i++) {
    if (lambda_vdw_ptr[i] >= LAMBDA_NEIGHBOR_THRESHOLD ||
        lambda_ele_ptr[i] >= LAMBDA_NEIGHBOR_THRESHOLD) {
      coupled_ptr[n_coupled_] = i;
      n_coupled_++;
    }
  }

  // Resize arrays based on actual coupled count
  if (n_coupled_ > 0) {
    coupled_indices_.resize(n_coupled_);
    neighbor_counts_.resize(n_coupled_);
  }
}

//-------------------------------------------------------------------------------------------------
void LambdaNeighborList::updateLambdas(const double* lambda_vdw_in,
                                       const double* lambda_ele_in) {
  // Update lambda values
  double* lambda_vdw_ptr = lambda_vdw_.data();
  double* lambda_ele_ptr = lambda_ele_.data();
  for (int i = 0; i < n_atoms_; i++) {
    lambda_vdw_ptr[i] = lambda_vdw_in[i];
    lambda_ele_ptr[i] = lambda_ele_in[i];
  }

  // Reidentify coupled atoms (list may have changed)
  identifyCoupledAtoms();
}

//-------------------------------------------------------------------------------------------------
bool LambdaNeighborList::needsRebuild(const double* xcrd, const double* ycrd,
                                      const double* zcrd) const {
  const double* last_xcrd = last_build_xcrd_.data();
  const double* last_ycrd = last_build_ycrd_.data();
  const double* last_zcrd = last_build_zcrd_.data();
  const int* coupled_ptr = coupled_indices_.data();

  const double half_skin_sq = (skin_ * 0.5) * (skin_ * 0.5);

  // Check if any coupled atom has moved more than skin/2
  for (int i_idx = 0; i_idx < n_coupled_; i_idx++) {
    const int i = coupled_ptr[i_idx];
    const double dx = xcrd[i] - last_xcrd[i];
    const double dy = ycrd[i] - last_ycrd[i];
    const double dz = zcrd[i] - last_zcrd[i];
    const double r2 = dx*dx + dy*dy + dz*dz;

    if (r2 > half_skin_sq) {
      return true;
    }
  }

  return false;
}

//-------------------------------------------------------------------------------------------------
void LambdaNeighborList::build(const double* xcrd, const double* ycrd, const double* zcrd,
                                const double* umat, UnitCellType unit_cell, bool use_gpu) {
#ifdef STORMM_USE_CUDA
  if (use_gpu) {
    buildGpu(xcrd, ycrd, zcrd, umat, unit_cell);
    return;
  }
#endif

  buildCpu(xcrd, ycrd, zcrd, umat, unit_cell);
}

//-------------------------------------------------------------------------------------------------
void LambdaNeighborList::buildCpu(const double* xcrd, const double* ycrd, const double* zcrd,
                                  const double* umat, UnitCellType unit_cell) {
  if (n_coupled_ == 0) {
    return;  // No coupled atoms, nothing to do
  }

  const int* coupled_ptr = coupled_indices_.data();
  int* neighbor_counts_ptr = neighbor_counts_.data();

  // Store coordinates for rebuild detection
  double* last_xcrd = last_build_xcrd_.data();
  double* last_ycrd = last_build_ycrd_.data();
  double* last_zcrd = last_build_zcrd_.data();
  for (int i = 0; i < n_atoms_; i++) {
    last_xcrd[i] = xcrd[i];
    last_ycrd[i] = ycrd[i];
    last_zcrd[i] = zcrd[i];
  }

  const bool is_periodic = (unit_cell != UnitCellType::NONE);
  const double cutoff_sq = cutoff_with_skin_sq_;

  // First pass: count neighbors for each coupled atom
  max_neighbors_ = 0;
  for (int i_idx = 0; i_idx < n_coupled_; i_idx++) {
    const int i = coupled_ptr[i_idx];
    const double xi = xcrd[i];
    const double yi = ycrd[i];
    const double zi = zcrd[i];

    int neighbor_count = 0;

    // Loop over all OTHER coupled atoms
    for (int j_idx = 0; j_idx < n_coupled_; j_idx++) {
      if (i_idx == j_idx) continue;  // Skip self

      const int j = coupled_ptr[j_idx];
      double dx = xcrd[j] - xi;
      double dy = ycrd[j] - yi;
      double dz = zcrd[j] - zi;

      // Apply minimum image convention if periodic
      if (is_periodic) {
        // Simplified for orthorhombic/cubic (would need full PBC for triclinic)
        // For now, assume non-periodic (typical for GCMC in vacuum)
      }

      const double r2 = dx*dx + dy*dy + dz*dz;
      if (r2 < cutoff_sq) {
        neighbor_count++;
      }
    }

    neighbor_counts_ptr[i_idx] = neighbor_count;
    max_neighbors_ = std::max(max_neighbors_, neighbor_count);
  }

  // Allocate neighbor list
  const size_t neighbor_list_size = static_cast<size_t>(n_coupled_) * max_neighbors_;
  if (neighbor_list_.size() < neighbor_list_size) {
    neighbor_list_.resize(neighbor_list_size);
  }

  int* neighbor_list_ptr = neighbor_list_.data();

  // Initialize to -1 (invalid)
  for (size_t k = 0; k < neighbor_list_size; k++) {
    neighbor_list_ptr[k] = -1;
  }

  // Second pass: fill neighbor list
  for (int i_idx = 0; i_idx < n_coupled_; i_idx++) {
    const int i = coupled_ptr[i_idx];
    const double xi = xcrd[i];
    const double yi = ycrd[i];
    const double zi = zcrd[i];

    int neighbor_idx = 0;
    const int base_offset = i_idx * max_neighbors_;

    for (int j_idx = 0; j_idx < n_coupled_; j_idx++) {
      if (i_idx == j_idx) continue;

      const int j = coupled_ptr[j_idx];
      double dx = xcrd[j] - xi;
      double dy = ycrd[j] - yi;
      double dz = zcrd[j] - zi;

      // Apply minimum image if periodic
      if (is_periodic) {
        // Simplified PBC
      }

      const double r2 = dx*dx + dy*dy + dz*dz;
      if (r2 < cutoff_sq) {
        neighbor_list_ptr[base_offset + neighbor_idx] = j_idx;  // Store coupled index
        neighbor_idx++;
      }
    }
  }
}

//-------------------------------------------------------------------------------------------------
void LambdaNeighborList::buildGpu(const double* xcrd, const double* ycrd, const double* zcrd,
                                  const double* umat, UnitCellType unit_cell) {
#ifdef STORMM_USE_CUDA
  if (n_coupled_ == 0) {
    return;
  }

  // Upload data to GPU
  coupled_indices_.upload();
  neighbor_counts_.upload();

  // Allocate temporary max_neighbors on GPU
  int* d_max_neighbors;
  cudaMalloc(&d_max_neighbors, sizeof(int));

  // Estimate initial capacity (start with generous estimate)
  const int initial_capacity = std::min(n_coupled_, 512);

  // Ensure neighbor list is large enough
  const size_t neighbor_list_size = static_cast<size_t>(n_coupled_) * initial_capacity;
  if (neighbor_list_.size() < neighbor_list_size) {
    neighbor_list_.resize(neighbor_list_size);
  }
  neighbor_list_.upload();

  // Get device pointers
  const int* d_coupled_indices = coupled_indices_.data(HybridTargetLevel::DEVICE);
  int* d_neighbor_counts = neighbor_counts_.data(HybridTargetLevel::DEVICE);
  int* d_neighbor_list = neighbor_list_.data(HybridTargetLevel::DEVICE);

  // Launch neighbor list construction
  launchLambdaNeighborListBuild(
      n_coupled_,
      d_coupled_indices,
      xcrd, ycrd, zcrd,  // Assumed to be device pointers
      cutoff_with_skin_sq_,
      d_neighbor_counts,
      d_neighbor_list,
      initial_capacity,
      d_max_neighbors);

  // Download results
  neighbor_counts_.download();

  // Get actual max neighbors
  int h_max_neighbors;
  cudaMemcpy(&h_max_neighbors, d_max_neighbors, sizeof(int), cudaMemcpyDeviceToHost);
  max_neighbors_ = h_max_neighbors;

  // If we exceeded capacity, rebuild with correct size
  if (max_neighbors_ > initial_capacity) {
    const size_t new_size = static_cast<size_t>(n_coupled_) * max_neighbors_;
    neighbor_list_.resize(new_size);
    neighbor_list_.upload();
    d_neighbor_list = neighbor_list_.data(HybridTargetLevel::DEVICE);

    // Rebuild with correct capacity
    launchLambdaNeighborListBuild(
        n_coupled_,
        d_coupled_indices,
        xcrd, ycrd, zcrd,
        cutoff_with_skin_sq_,
        d_neighbor_counts,
        d_neighbor_list,
        max_neighbors_,
        d_max_neighbors);
  }

  // Download final neighbor list
  neighbor_list_.download();

  // Clean up
  cudaFree(d_max_neighbors);

  // Store coordinates for rebuild detection
  double* last_xcrd = last_build_xcrd_.data();
  double* last_ycrd = last_build_ycrd_.data();
  double* last_zcrd = last_build_zcrd_.data();
  const int* coupled_ptr = coupled_indices_.data();

  // Copy coupled atom coordinates (from device pointers to host storage)
  for (int i_idx = 0; i_idx < n_coupled_; i_idx++) {
    const int i = coupled_ptr[i_idx];
    // Note: This assumes xcrd, ycrd, zcrd are device pointers
    // In practice, we'd need to copy from device to host first
    // For now, mark as TODO
    last_xcrd[i] = 0.0;  // TODO: Proper coordinate storage
    last_ycrd[i] = 0.0;
    last_zcrd[i] = 0.0;
  }
#else
  // Fall back to CPU if CUDA not available
  buildCpu(xcrd, ycrd, zcrd, umat, unit_cell);
#endif
}

//-------------------------------------------------------------------------------------------------
LambdaNeighborListReader LambdaNeighborList::data() const {
  return LambdaNeighborListReader(n_coupled_,
                                   coupled_indices_.data(),
                                   max_neighbors_,
                                   neighbor_counts_.data(),
                                   neighbor_list_.data(),
                                   cutoff_);
}

} // namespace energy
} // namespace stormm
