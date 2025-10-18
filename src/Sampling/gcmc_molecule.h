// -*-c++-*-
#ifndef STORMM_GCMC_MOLECULE_H
#define STORMM_GCMC_MOLECULE_H

#include <vector>
#include "copyright.h"

namespace stormm {
namespace sampling {

/// \brief Two-stage coupling threshold: VDW couples over [0, 0.75], electrostatics over [0.75, 1.0]
///
/// This constant defines the lambda value at which VDW interactions are fully coupled and
/// electrostatic interactions begin to be coupled. The two-stage approach improves acceptance
/// rates by first creating space with VDW interactions before adding electrostatics.
///
/// The choice of 0.75 is based on best practices in alchemical free energy calculations:
/// - VDW coupling over 75% of the switching protocol creates sufficient space
/// - Remaining 25% for electrostatics is enough for polar molecule insertion
/// - Values in range [0.6, 0.8] have been shown to work well in literature
constexpr double VDW_COUPLING_THRESHOLD = 0.75;

/// \brief Enumeration for tracking the state of GCMC molecules
enum class GCMCMoleculeStatus {
  GHOST = 0,      ///< Lambda = 0, non-interacting, available for insertion
  ACTIVE = 1,     ///< Lambda = 1 or being perturbed, tracked by GCMC
  UNTRACKED = 2   ///< Outside sampling region, lambda = 1 but not under GCMC control
};

/// \brief Struct to hold information about a single GCMC-controlled molecule
///
/// This struct tracks the state of a molecule that can transition between ghost (non-interacting)
/// and active states during Grand Canonical Monte Carlo sampling. It stores both the current
/// coupling parameters (lambda values) and the original force field parameters needed for
/// scaling operations.
struct GCMCMolecule {

  /// \brief Default constructor
  GCMCMolecule();

  /// \brief Constructor with residue ID initialization
  ///
  /// \param resid_in  Residue ID in the topology
  GCMCMolecule(int resid_in);

  int resid;                               ///< Residue ID in topology
  GCMCMoleculeStatus status;               ///< Current status of the molecule
  std::vector<int> atom_indices;           ///< Atom indices in topology
  std::vector<int> heavy_atom_indices;     ///< Heavy atoms (for COG calculation)
  double lambda_vdw;                       ///< Current VDW lambda [0, 1]
  double lambda_ele;                       ///< Current electrostatic lambda [0, 1]

  /// Original parameters (stored for scaling operations)
  std::vector<double> original_charges;    ///< Original partial charges (e)
  std::vector<double> original_sigma;      ///< Original LJ sigma parameters (Angstroms)
  std::vector<double> original_epsilon;    ///< Original LJ epsilon parameters (kcal/mol)

  /// \brief Check if the molecule is currently active (lambda = 1.0)
  ///
  /// \return True if the molecule is fully coupled (both lambda values = 1.0)
  bool isActive() const;

  /// \brief Check if the molecule is currently a ghost (lambda = 0.0)
  ///
  /// \return True if the molecule is fully decoupled (both lambda values = 0.0)
  bool isGhost() const;

  /// \brief Get the combined lambda value for this molecule
  ///
  /// For two-stage coupling, this returns the overall progress through the protocol.
  ///
  /// \return Combined lambda value considering both VDW and electrostatic contributions
  double getCombinedLambda() const;
};

/// \brief Statistics tracking for GCMC sampling
struct GCMCStatistics {

  /// \brief Default constructor initializes all counters to zero
  GCMCStatistics();

  // Move counters
  int n_moves;                        ///< Total move attempts
  int n_accepted;                     ///< Total accepted moves
  int n_inserts;                      ///< Insertion attempts
  int n_deletes;                      ///< Deletion attempts
  int n_accepted_inserts;             ///< Accepted insertions
  int n_accepted_deletes;             ///< Accepted deletions
  int n_explosions;                   ///< Integration failures
  int n_left_sphere;                  ///< Molecules left sphere during NCMC

  // History tracking
  std::vector<int> N_history;         ///< N after each move
  std::vector<double> acc_rate_history;        ///< Running acceptance rate
  std::vector<double> acceptance_probs;        ///< All acceptance probabilities
  std::vector<double> insert_acceptance_probs; ///< Insertion acceptance probabilities
  std::vector<double> delete_acceptance_probs; ///< Deletion acceptance probabilities
  std::vector<int> move_resids;               ///< Residue IDs for each move
  std::vector<std::string> outcomes;          ///< Move outcomes ("accepted insert", etc.)

  // NCMC-specific tracking
  std::vector<double> insert_works;           ///< Protocol work for insertions (kcal/mol)
  std::vector<double> delete_works;           ///< Protocol work for deletions (kcal/mol)
  std::vector<double> accepted_insert_works;  ///< Work for accepted insertions
  std::vector<double> accepted_delete_works;  ///< Work for accepted deletions

  /// \brief Get the current acceptance rate
  ///
  /// \return Acceptance rate as a percentage (0-100)
  double getAcceptanceRate() const;

  /// \brief Get the insertion acceptance rate
  ///
  /// \return Insertion acceptance rate as a percentage (0-100)
  double getInsertionAcceptanceRate() const;

  /// \brief Get the deletion acceptance rate
  ///
  /// \return Deletion acceptance rate as a percentage (0-100)
  double getDeletionAcceptanceRate() const;

  /// \brief Reset all statistics to initial values
  void reset();
};

} // namespace sampling
} // namespace stormm

#endif