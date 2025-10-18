// -*-c++-*-
#ifndef STORMM_LAMBDA_NEIGHBOR_LIST_H
#define STORMM_LAMBDA_NEIGHBOR_LIST_H

#include "copyright.h"
#include "Accelerator/hybrid.h"
#include "DataTypes/common_types.h"
#include "Topology/atomgraph_enumerators.h"

namespace stormm {
namespace energy {

using card::Hybrid;
using card::HybridTargetLevel;
using topology::UnitCellType;

/// \brief Lambda threshold for determining coupled vs ghost atoms
constexpr double LAMBDA_NEIGHBOR_THRESHOLD = 0.01;

/// \brief Default cutoff distance for neighbor list construction (Angstroms)
constexpr double DEFAULT_NEIGHBOR_CUTOFF = 12.0;

/// \brief Neighbor list skin distance to reduce rebuild frequency (Angstroms)
constexpr double DEFAULT_NEIGHBOR_SKIN = 2.0;

/// \brief Abstract for passing lambda neighbor list data to kernels
struct LambdaNeighborListReader {
  int n_coupled;                    ///< Number of coupled atoms (lambda > threshold)
  const int* coupled_indices;       ///< Global indices of coupled atoms
  int max_neighbors;                ///< Maximum neighbors per atom
  const int* neighbor_counts;       ///< Number of neighbors for each coupled atom
  const int* neighbor_list;         ///< Flattened neighbor list [n_coupled * max_neighbors]
                                    ///< neighbor_list[i * max_neighbors + j] = neighbor index
  double cutoff;                    ///< Cutoff distance for interactions (Angstroms)
  double cutoff_squared;            ///< Squared cutoff for fast comparison

  /// \brief Constructor
  LambdaNeighborListReader(int n_coupled_in, const int* coupled_indices_in,
                           int max_neighbors_in, const int* neighbor_counts_in,
                           const int* neighbor_list_in, double cutoff_in);
};

/// \brief Lambda-aware neighbor list for GCMC/NCMC simulations
///
/// This lightweight neighbor list structure filters out ghost atoms and
/// maintains neighbor lists ONLY for coupled atoms (lambda > threshold).
/// Provides O(N_coupled) scaling with cutoffs instead of O(N_coupled²).
///
/// Key features:
/// - Filters ghosts automatically based on lambda values
/// - Simple cutoff-based neighbor list (no complex spatial decomposition)
/// - Verlet skin for reduced rebuild frequency
/// - GPU-accelerated construction
/// - Integrates seamlessly with existing lambda nonbonded code
class LambdaNeighborList {
public:

  /// \brief Constructor
  ///
  /// \param n_atoms_in       Total number of atoms (coupled + ghosts)
  /// \param lambda_vdw_in    Per-atom VDW lambda values
  /// \param lambda_ele_in    Per-atom electrostatic lambda values
  /// \param cutoff_in        Cutoff distance for neighbor list (Angstroms)
  /// \param skin_in          Skin distance to reduce rebuild frequency (Angstroms)
  /// \param gpu_config       GPU configuration (HybridKind::HOST_ONLY or ACCELERATED)
  LambdaNeighborList(int n_atoms_in, const double* lambda_vdw_in,
                     const double* lambda_ele_in,
                     double cutoff_in = DEFAULT_NEIGHBOR_CUTOFF,
                     double skin_in = DEFAULT_NEIGHBOR_SKIN);

  /// \brief Build/rebuild the neighbor list based on current coordinates
  ///
  /// \param xcrd            X coordinates (device or host array, size n_atoms)
  /// \param ycrd            Y coordinates (device or host array, size n_atoms)
  /// \param zcrd            Z coordinates (device or host array, size n_atoms)
  /// \param umat            Unit cell transformation matrix (9 elements, can be nullptr)
  /// \param unit_cell       Unit cell type (NONE for non-periodic)
  /// \param use_gpu         Whether to use GPU acceleration
  void build(const double* xcrd, const double* ycrd, const double* zcrd,
             const double* umat, UnitCellType unit_cell, bool use_gpu = true);

  /// \brief Update lambda values and rebuild coupled atom list if needed
  ///
  /// Call this when lambda values change (e.g., after NCMC perturbation)
  ///
  /// \param lambda_vdw_in   Updated VDW lambda values
  /// \param lambda_ele_in   Updated electrostatic lambda values
  void updateLambdas(const double* lambda_vdw_in, const double* lambda_ele_in);

  /// \brief Check if neighbor list needs rebuilding
  ///
  /// Compares current coordinates with those from last build.
  /// Returns true if any coupled atom has moved more than skin/2.
  ///
  /// \param xcrd            Current X coordinates
  /// \param ycrd            Current Y coordinates
  /// \param zcrd            Current Z coordinates
  /// \return                True if rebuild needed
  bool needsRebuild(const double* xcrd, const double* ycrd, const double* zcrd) const;

  /// \brief Get read-only abstract for kernel access
  LambdaNeighborListReader data() const;

  /// \brief Get number of coupled atoms
  int getCoupledCount() const { return n_coupled_; }

  /// \brief Get maximum neighbors per atom
  int getMaxNeighbors() const { return max_neighbors_; }

  /// \brief Get cutoff distance
  double getCutoff() const { return cutoff_; }

  /// \brief Get access to coupled indices array
  const Hybrid<int>& getCoupledIndices() const { return coupled_indices_; }

private:

  /// \brief Identify coupled atoms (lambda > threshold) and build coupled_indices
  void identifyCoupledAtoms();

  /// \brief Build neighbor list on GPU
  void buildGpu(const double* xcrd, const double* ycrd, const double* zcrd,
                const double* umat, UnitCellType unit_cell);

  /// \brief Build neighbor list on CPU (fallback)
  void buildCpu(const double* xcrd, const double* ycrd, const double* zcrd,
                const double* umat, UnitCellType unit_cell);

  int n_atoms_;                           ///< Total number of atoms
  int n_coupled_;                         ///< Number of coupled atoms
  double cutoff_;                         ///< Cutoff distance (Angstroms)
  double skin_;                           ///< Skin distance (Angstroms)
  double cutoff_with_skin_;               ///< cutoff + skin
  double cutoff_with_skin_sq_;            ///< (cutoff + skin)²

  Hybrid<double> lambda_vdw_;             ///< VDW lambda values [n_atoms]
  Hybrid<double> lambda_ele_;             ///< Electrostatic lambda values [n_atoms]
  Hybrid<int> coupled_indices_;           ///< Indices of coupled atoms [n_coupled]

  int max_neighbors_;                     ///< Maximum neighbors per coupled atom
  Hybrid<int> neighbor_counts_;           ///< Number of neighbors [n_coupled]
  Hybrid<int> neighbor_list_;             ///< Flattened neighbor list [n_coupled * max_neighbors]

  // Rebuild detection
  Hybrid<double> last_build_xcrd_;        ///< Coordinates at last build
  Hybrid<double> last_build_ycrd_;
  Hybrid<double> last_build_zcrd_;
};

} // namespace energy
} // namespace stormm

#endif // STORMM_LAMBDA_NEIGHBOR_LIST_H
