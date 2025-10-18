#include <cmath>
#include "copyright.h"
#include "Constants/symbol_values.h"
#include "Math/vector_ops.h"
#include "Random/random.h"
#include "gcmc_sphere.h"

namespace stormm {
namespace sampling {

using random::Xoroshiro128pGenerator;

//-------------------------------------------------------------------------------------------------
GCMCSphere::GCMCSphere() :
  center_{0.0, 0.0, 0.0},
  radius_{5.0},
  volume_{0.0},
  dynamic_center_{false}
{
  updateVolume();
}

//-------------------------------------------------------------------------------------------------
GCMCSphere::GCMCSphere(const double3 &center_in, double radius_in) :
  center_{center_in},
  radius_{radius_in},
  volume_{0.0},
  dynamic_center_{false}
{
  updateVolume();
}

//-------------------------------------------------------------------------------------------------
GCMCSphere::GCMCSphere(const std::vector<int> &ref_atom_indices_in, double radius_in) :
  center_{0.0, 0.0, 0.0},
  radius_{radius_in},
  volume_{0.0},
  ref_atom_indices_{ref_atom_indices_in},
  dynamic_center_{true}
{
  updateVolume();
}

//-------------------------------------------------------------------------------------------------
double3 GCMCSphere::getCenter() const {
  return center_;
}

//-------------------------------------------------------------------------------------------------
void GCMCSphere::setCenter(const double3 &new_center) {
  center_ = new_center;
  dynamic_center_ = false;
  ref_atom_indices_.clear();
}

//-------------------------------------------------------------------------------------------------
void GCMCSphere::updateCenter(const double* xcrd, const double* ycrd, const double* zcrd) {
  if (!dynamic_center_ || ref_atom_indices_.empty()) {
    return;
  }

  // Calculate center of geometry of reference atoms
  double sum_x = 0.0;
  double sum_y = 0.0;
  double sum_z = 0.0;

  for (int atom_idx : ref_atom_indices_) {
    sum_x += xcrd[atom_idx];
    sum_y += ycrd[atom_idx];
    sum_z += zcrd[atom_idx];
  }

  const double n_atoms = static_cast<double>(ref_atom_indices_.size());
  center_.x = sum_x / n_atoms;
  center_.y = sum_y / n_atoms;
  center_.z = sum_z / n_atoms;
}

//-------------------------------------------------------------------------------------------------
double GCMCSphere::getRadius() const {
  return radius_;
}

//-------------------------------------------------------------------------------------------------
void GCMCSphere::setRadius(double new_radius) {
  if (new_radius <= 0.0) {
    rtErr("Sphere radius must be positive", "GCMCSphere::setRadius");
  }
  radius_ = new_radius;
  updateVolume();
}

//-------------------------------------------------------------------------------------------------
double GCMCSphere::getVolume() const {
  return volume_;
}

//-------------------------------------------------------------------------------------------------
bool GCMCSphere::contains(const double3 &point) const {
  const double dx = point.x - center_.x;
  const double dy = point.y - center_.y;
  const double dz = point.z - center_.z;
  const double r2 = dx*dx + dy*dy + dz*dz;
  return r2 <= radius_ * radius_;
}

//-------------------------------------------------------------------------------------------------
bool GCMCSphere::contains(double x, double y, double z) const {
  const double dx = x - center_.x;
  const double dy = y - center_.y;
  const double dz = z - center_.z;
  const double r2 = dx*dx + dy*dy + dz*dz;
  return r2 <= radius_ * radius_;
}

//-------------------------------------------------------------------------------------------------
double GCMCSphere::distanceToCenter(const double3 &point) const {
  const double dx = point.x - center_.x;
  const double dy = point.y - center_.y;
  const double dz = point.z - center_.z;
  return std::sqrt(dx*dx + dy*dy + dz*dz);
}

//-------------------------------------------------------------------------------------------------
double GCMCSphere::distanceToBoundary(const double3 &point) const {
  return distanceToCenter(point) - radius_;
}

//-------------------------------------------------------------------------------------------------
bool GCMCSphere::hasDynamicCenter() const {
  return dynamic_center_;
}

//-------------------------------------------------------------------------------------------------
const std::vector<int>& GCMCSphere::getReferenceAtoms() const {
  return ref_atom_indices_;
}

//-------------------------------------------------------------------------------------------------
double3 GCMCSphere::generateRandomPoint(ullint2* rng_state) const {
  // Use rejection sampling to generate uniform point within sphere
  // For now, use a simple approach with Xoroshiro128pGenerator
  // TODO: Implement proper device-compatible random generation
  Xoroshiro128pGenerator rng(*rng_state);

  double3 point;
  bool found = false;

  while (!found) {
    // Generate random point in cube [-1, 1]^3
    const double u1 = 2.0 * rng.uniformRandomNumber() - 1.0;
    const double u2 = 2.0 * rng.uniformRandomNumber() - 1.0;
    const double u3 = 2.0 * rng.uniformRandomNumber() - 1.0;

    // Check if point is within unit sphere
    const double r2 = u1*u1 + u2*u2 + u3*u3;
    if (r2 <= 1.0) {
      // Scale to actual sphere
      point.x = center_.x + radius_ * u1;
      point.y = center_.y + radius_ * u2;
      point.z = center_.z + radius_ * u3;
      found = true;
    }
  }

  // Update the state
  *rng_state = rng.revealState();

  return point;
}

//-------------------------------------------------------------------------------------------------
double3 GCMCSphere::generateRandomSurfacePoint(ullint2* rng_state) const {
  // Generate uniform point on sphere surface using Marsaglia's method
  // For now, use a simple approach with Xoroshiro128pGenerator
  // TODO: Implement proper device-compatible random generation
  Xoroshiro128pGenerator rng(*rng_state);

  double u1, u2, s;

  // Generate point on unit circle
  do {
    u1 = 2.0 * rng.uniformRandomNumber() - 1.0;
    u2 = 2.0 * rng.uniformRandomNumber() - 1.0;
    s = u1*u1 + u2*u2;
  } while (s >= 1.0);

  // Convert to point on unit sphere
  const double factor = 2.0 * std::sqrt(1.0 - s);
  const double x = factor * u1;
  const double y = factor * u2;
  const double z = 1.0 - 2.0 * s;

  // Scale to actual sphere
  double3 point;
  point.x = center_.x + radius_ * x;
  point.y = center_.y + radius_ * y;
  point.z = center_.z + radius_ * z;

  // Update the state
  *rng_state = rng.revealState();

  return point;
}

//-------------------------------------------------------------------------------------------------
void GCMCSphere::updateVolume() {
  constexpr double pi = 3.141592653589793;
  volume_ = (4.0 / 3.0) * pi * radius_ * radius_ * radius_;
}

} // namespace sampling
} // namespace stormm