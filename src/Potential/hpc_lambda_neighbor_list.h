// -*-c++-*-
#ifndef STORMM_HPC_LAMBDA_NEIGHBOR_LIST_H
#define STORMM_HPC_LAMBDA_NEIGHBOR_LIST_H

#include "copyright.h"

namespace stormm {
namespace energy {

/// \brief Launch GPU kernel to build lambda-aware neighbor list
///
/// \param n_coupled                 Number of coupled atoms
/// \param coupled_indices           Indices of coupled atoms (device array)
/// \param xcrd                      X coordinates (device array, all atoms)
/// \param ycrd                      Y coordinates (device array, all atoms)
/// \param zcrd                      Z coordinates (device array, all atoms)
/// \param cutoff_with_skin_sq       Squared cutoff distance (with skin)
/// \param neighbor_counts           Output: number of neighbors per atom (device array, size n_coupled)
/// \param neighbor_list             Output: flattened neighbor list (device array)
/// \param max_neighbors_capacity    Capacity of neighbor list per atom
/// \param max_neighbors_out         Output: maximum neighbors found (device pointer, 1 int)
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
    int* max_neighbors_out);

} // namespace energy
} // namespace stormm

#endif // STORMM_HPC_LAMBDA_NEIGHBOR_LIST_H
