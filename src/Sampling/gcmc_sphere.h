// -*-c++-*-
#ifndef STORMM_GCMC_SPHERE_H
#define STORMM_GCMC_SPHERE_H

#include <vector>
#include "copyright.h"
#include "DataTypes/stormm_vector_types.h"

namespace stormm {
namespace sampling {

/// \brief Defines a spherical sampling region for GCMC moves
///
/// This class manages a spherical region within which GCMC insertion and deletion moves
/// are performed. The sphere can have a fixed center or a dynamic center that tracks
/// a set of reference atoms.
class GCMCSphere {
public:

  /// \brief Default constructor creates an empty sphere
  GCMCSphere();

  /// \brief Constructor for a sphere with fixed center
  ///
  /// \param center_in  Center position in Cartesian coordinates (Angstroms)
  /// \param radius_in  Radius of the sphere (Angstroms)
  GCMCSphere(const double3 &center_in, double radius_in);

  /// \brief Constructor for a sphere with dynamic center based on reference atoms
  ///
  /// \param ref_atom_indices_in  Indices of atoms that define the sphere center
  /// \param radius_in            Radius of the sphere (Angstroms)
  GCMCSphere(const std::vector<int> &ref_atom_indices_in, double radius_in);

  /// \brief Get the current center of the sphere
  ///
  /// \return Center position in Cartesian coordinates
  double3 getCenter() const;

  /// \brief Set a new fixed center for the sphere
  ///
  /// \param new_center  New center position
  void setCenter(const double3 &new_center);

  /// \brief Update the sphere center based on reference atom positions
  ///
  /// This should be called when the reference atoms have moved and the sphere
  /// center needs to be recalculated.
  ///
  /// \param xcrd  X coordinates of all atoms
  /// \param ycrd  Y coordinates of all atoms
  /// \param zcrd  Z coordinates of all atoms
  void updateCenter(const double* xcrd, const double* ycrd, const double* zcrd);

  /// \brief Get the radius of the sphere
  ///
  /// \return Sphere radius in Angstroms
  double getRadius() const;

  /// \brief Set a new radius for the sphere
  ///
  /// \param new_radius  New radius in Angstroms
  void setRadius(double new_radius);

  /// \brief Get the volume of the sphere
  ///
  /// \return Sphere volume in cubic Angstroms
  double getVolume() const;

  /// \brief Check if a point is inside the sphere
  ///
  /// \param point  Point to test in Cartesian coordinates
  /// \return True if the point is within the sphere radius
  bool contains(const double3 &point) const;

  /// \brief Check if a point is inside the sphere
  ///
  /// \param x  X coordinate of the point
  /// \param y  Y coordinate of the point
  /// \param z  Z coordinate of the point
  /// \return True if the point is within the sphere radius
  bool contains(double x, double y, double z) const;

  /// \brief Calculate the distance from a point to the sphere center
  ///
  /// \param point  Point in Cartesian coordinates
  /// \return Distance to sphere center in Angstroms
  double distanceToCenter(const double3 &point) const;

  /// \brief Calculate the distance from a point to the sphere boundary
  ///
  /// \param point  Point in Cartesian coordinates
  /// \return Distance to sphere boundary (negative if inside)
  double distanceToBoundary(const double3 &point) const;

  /// \brief Check if the sphere has a dynamic center
  ///
  /// \return True if the center is defined by reference atoms
  bool hasDynamicCenter() const;

  /// \brief Get the reference atom indices
  ///
  /// \return Vector of atom indices that define the dynamic center
  const std::vector<int>& getReferenceAtoms() const;

  /// \brief Generate a random point within the sphere
  ///
  /// Uses a uniform distribution within the spherical volume.
  ///
  /// \param rng_state  Random number generator state
  /// \return Random point within the sphere
  double3 generateRandomPoint(ullint2* rng_state) const;

  /// \brief Generate a random point on the sphere surface
  ///
  /// \param rng_state  Random number generator state
  /// \return Random point on the sphere surface
  double3 generateRandomSurfacePoint(ullint2* rng_state) const;

private:
  double3 center_;                     ///< Current center of the sphere
  double radius_;                      ///< Radius of the sphere (Angstroms)
  double volume_;                      ///< Cached volume (4/3 * pi * r^3)
  std::vector<int> ref_atom_indices_;  ///< Atoms defining the dynamic center
  bool dynamic_center_;                ///< True if center follows reference atoms

  /// \brief Update the cached volume after radius change
  void updateVolume();
};

} // namespace sampling
} // namespace stormm

#endif