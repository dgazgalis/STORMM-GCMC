// -*-c++-*-
#ifndef STORMM_MC_MOVER_H
#define STORMM_MC_MOVER_H

#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "copyright.h"
#include "Constants/symbol_values.h"
#include "Math/vector_ops.h"
#include "Random/random.h"
#include "Topology/atomgraph.h"
#include "Trajectory/phasespace.h"
#include "gcmc_molecule.h"

namespace stormm {
namespace sampling {

// Forward declaration
class GCMCSampler;

using random::Xoshiro256ppGenerator;
using topology::AtomGraph;
using trajectory::PhaseSpace;

/// \brief Statistics tracking for Monte Carlo moves
struct MCMoveStatistics {
  int n_attempted;  ///< Total number of attempts
  int n_accepted;   ///< Number of accepted moves
  int n_rejected;   ///< Number of rejected moves
  double total_energy_change;  ///< Total energy change from accepted moves

  /// \brief Constructor
  MCMoveStatistics();

  /// \brief Get acceptance rate
  double getAcceptanceRate() const;

  /// \brief Reset statistics
  void reset();
};

/// \brief Rotatable bond information for torsion moves
struct RotatableBond {
  int atom1;  ///< First atom of the bond
  int atom2;  ///< Second atom of the bond
  std::vector<int> rotating_atoms;  ///< Atoms that rotate with this bond
  bool is_terminal;  ///< True if this is a terminal bond (methyl, etc.)
};

/// \brief Base class for Monte Carlo moves on molecules
///
/// This abstract class provides the interface for different Monte Carlo move types
/// that can be applied to molecules in GCMC simulations. Derived classes implement
/// specific move types like translation, rotation, and torsion angle changes.
class MCMover {
public:

  /// \brief Constructor
  ///
  /// \param sampler      Pointer to the parent GCMC sampler
  /// \param beta         Inverse temperature (1/kT in mol/kcal)
  /// \param rng          Random number generator
  MCMover(GCMCSampler* sampler, double beta, Xoshiro256ppGenerator* rng);

  /// \brief Virtual destructor
  virtual ~MCMover() = default;

  /// \brief Attempt a Monte Carlo move on the specified molecule
  ///
  /// This is the main interface for performing MC moves. Each derived class
  /// implements its specific move type.
  ///
  /// \param mol  Molecule to apply the move to
  /// \return True if the move was accepted, false if rejected
  virtual bool attemptMove(GCMCMolecule& mol) = 0;

  /// \brief Get the name of this move type
  ///
  /// \return String describing the move type (e.g., "Translation", "Rotation")
  virtual std::string getMoveType() const = 0;

  /// \brief Get move statistics
  ///
  /// \return Reference to the statistics object
  const MCMoveStatistics& getStatistics() const { return stats_; }

  /// \brief Reset statistics
  void resetStatistics() { stats_.reset(); }

protected:
  GCMCSampler* sampler_;           ///< Parent sampler (for energy evaluation)
  double beta_;                    ///< Inverse temperature (mol/kcal)
  Xoshiro256ppGenerator* rng_;     ///< Random number generator
  MCMoveStatistics stats_;         ///< Move statistics

  /// \brief Apply Metropolis acceptance criterion
  ///
  /// \param delta_E  Energy change from the move (kcal/mol)
  /// \return True if the move should be accepted
  bool acceptMove(double delta_E);
};

/// \brief Translation move for molecules
///
/// Randomly displaces the center of mass of a molecule by a random vector
/// with components in the range [-max_displacement, +max_displacement].
class TranslationMover : public MCMover {
public:

  /// \brief Constructor
  ///
  /// \param sampler          Parent GCMC sampler
  /// \param beta             Inverse temperature
  /// \param rng              Random number generator
  /// \param max_displacement Maximum displacement in Angstroms
  TranslationMover(GCMCSampler* sampler, double beta, Xoshiro256ppGenerator* rng,
                   double max_displacement);

  /// \brief Attempt a translation move
  ///
  /// \param mol  Molecule to translate
  /// \return True if accepted
  bool attemptMove(GCMCMolecule& mol) override;

  /// \brief Get move type name
  std::string getMoveType() const override { return "Translation"; }

  /// \brief Get maximum displacement
  double getMaxDisplacement() const { return max_displacement_; }

  /// \brief Set maximum displacement
  void setMaxDisplacement(double disp) { max_displacement_ = disp; }

private:
  double max_displacement_;  ///< Maximum displacement per axis (Angstroms)
};

/// \brief Rotation move for molecules
///
/// Randomly rotates a molecule around its center of geometry. Can use either
/// uniform random rotations (quaternions) or limited rotations with a maximum angle.
class RotationMover : public MCMover {
public:

  /// \brief Constructor
  ///
  /// \param sampler    Parent GCMC sampler
  /// \param beta       Inverse temperature
  /// \param rng        Random number generator
  /// \param max_angle  Maximum rotation angle in radians (0 = uniform)
  RotationMover(GCMCSampler* sampler, double beta, Xoshiro256ppGenerator* rng,
                double max_angle = 0.0);

  /// \brief Attempt a rotation move
  ///
  /// \param mol  Molecule to rotate
  /// \return True if accepted
  bool attemptMove(GCMCMolecule& mol) override;

  /// \brief Get move type name
  std::string getMoveType() const override { return "Rotation"; }

  /// \brief Get maximum rotation angle
  double getMaxAngle() const { return max_angle_; }

  /// \brief Set maximum rotation angle
  void setMaxAngle(double angle) { max_angle_ = angle; }

private:
  double max_angle_;  ///< Maximum rotation angle (radians, 0 = uniform)

  /// \brief Generate a random rotation quaternion
  ///
  /// \param use_limited  If true, limit rotation to max_angle
  /// \return Quaternion (w, x, y, z)
  std::vector<double> generateRandomQuaternion(bool use_limited);

  /// \brief Apply quaternion rotation to a point
  ///
  /// \param point  3D point to rotate
  /// \param quat   Quaternion (w, x, y, z)
  /// \param center Center of rotation
  /// \return Rotated point
  double3 rotatePoint(const double3& point, const std::vector<double>& quat,
                      const double3& center);
};

/// \brief Torsion angle move for molecules
///
/// Randomly changes dihedral angles around rotatable bonds. This is particularly
/// useful for flexible molecules and ligands.
class TorsionMover : public MCMover {
public:

  /// \brief Constructor
  ///
  /// \param sampler    Parent GCMC sampler
  /// \param beta       Inverse temperature
  /// \param rng        Random number generator
  /// \param topology   Topology for bond information
  /// \param max_angle  Maximum torsion change in radians
  TorsionMover(GCMCSampler* sampler, double beta, Xoshiro256ppGenerator* rng,
               const AtomGraph* topology, double max_angle);

  /// \brief Attempt a torsion move
  ///
  /// \param mol  Molecule to modify
  /// \return True if accepted
  bool attemptMove(GCMCMolecule& mol) override;

  /// \brief Get move type name
  std::string getMoveType() const override { return "Torsion"; }

  /// \brief Get maximum torsion angle change
  double getMaxAngle() const { return max_angle_; }

  /// \brief Set maximum torsion angle change
  void setMaxAngle(double angle) { max_angle_ = angle; }

private:
  const AtomGraph* topology_;  ///< Topology for bond connectivity
  double max_angle_;           ///< Maximum torsion change (radians)

  // Cache of rotatable bonds per molecule (indexed by first atom index)
  std::unordered_map<int, std::vector<RotatableBond>> rotatable_bonds_cache_;

  /// \brief Identify rotatable bonds in a molecule
  ///
  /// \param mol  Molecule to analyze
  /// \return Vector of rotatable bonds
  std::vector<RotatableBond> identifyRotatableBonds(const GCMCMolecule& mol);

  /// \brief Check if a bond is rotatable
  ///
  /// \param atom1  First atom index
  /// \param atom2  Second atom index
  /// \return True if the bond is rotatable
  bool isBondRotatable(int atom1, int atom2) const;

  /// \brief Apply torsion rotation to atoms
  ///
  /// \param atom_indices  Atoms to rotate
  /// \param axis_start    Start of rotation axis
  /// \param axis_end      End of rotation axis
  /// \param angle         Rotation angle (radians)
  /// \param ps            Phase space with coordinates
  void rotateBondedAtoms(const std::vector<int>& atom_indices,
                         const double3& axis_start,
                         const double3& axis_end,
                         double angle,
                         PhaseSpace* ps);
};

/// \brief Factory function to create MC movers
///
/// \param move_type   Type of move ("translation", "rotation", "torsion")
/// \param sampler     Parent GCMC sampler
/// \param beta        Inverse temperature
/// \param rng         Random number generator
/// \param topology    Topology (required for torsion moves)
/// \param parameter   Move-specific parameter (max displacement/angle)
/// \return Unique pointer to the created mover
std::unique_ptr<MCMover> createMCMover(const std::string& move_type,
                                       GCMCSampler* sampler,
                                       double beta,
                                       Xoshiro256ppGenerator* rng,
                                       const AtomGraph* topology,
                                       double parameter);

} // namespace sampling
} // namespace stormm

#endif