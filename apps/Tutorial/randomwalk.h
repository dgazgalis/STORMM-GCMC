// -*-c++-*-
#ifndef TUTORIAL_II_RANDOMWALK_H
#define TUTORIAL_II_RANDOMWALK_H

#include "../../src/copyright.h"
#include "../../src/Accelerator/gpu_enumerators.h"
#include "../../src/Accelerator/gpu_details.h"
#include "../../src/Accelerator/hybrid.h"
#include "../../src/DataTypes/common_types.h"
#include "../../src/DataTypes/stormm_vector_types.h"
#include "../../src/Random/random_enumerators.h"

namespace tutorial {

using stormm::card::GpuDetails;
using stormm::card::Hybrid;
using stormm::card::HybridTargetLevel;
using stormm::data_types::llint;
using stormm::data_types::ullint2;
#ifndef STORMM_USE_HPC
using stormm::data_types::double2;
#endif
using stormm::random::RandomNumberKind;

//-------------------------------------------------------------------------------------------------
// Mutable abstract for the prototypical STORMM class below, containing pointers to the data,
// critical constants for its operation, and length constants for arrays
//-------------------------------------------------------------------------------------------------
struct RandomWalkWriter {

  // The constructor will take input arguments for all member variables.
  RandomWalkWriter(int ncoord_in, int bits_in, double fluctuation_in, RandomNumberKind style_in,
                   llint* xcrd_in, llint* ycrd_in, ullint2* rng_stt_in);

  // The presence of const members prevents the default copy and move assignment operators from
  // working.  However, the default copy and move constructors are still valid.
  //
  // Arguments:
  //   original:  The original object to copy or move
  RandomWalkWriter(const RandomWalkWriter &original) = default;
  RandomWalkWriter(RandomWalkWriter &&original) = default;

  const int ncoord;              // The number of (particle) coordinates
  const int bits;                // Number of bits after the point in the internal, fixed-precision
                                 //   representation of coordinates
  const double fluctuation;      // Size of fluctuations to add with each random walk step.  This
                                 //   is the width of the range (the range is centered on zero) in
                                 //   a uniform random walk distribution or the sigma width of the
                                 //   Gaussian (also centered on zero) in a normal random walk.
  const RandomNumberKind style;  // The type of random walk, whether based on a uniform or a normal
                                 //   (Gaussian) distribution
  llint* xcrd;                   // Cartesian X coordinates of all particles
  llint* ycrd;                   // Cartesian Y coordinates of all particles
  ullint2* rng_stt;              // Random number state vectors driving each particle
};

//-------------------------------------------------------------------------------------------------
// Read-only abstract for the prototypical STORMM class below, containing pointers to the data,
// critical constants for its operation, and length constants for arrays
//-------------------------------------------------------------------------------------------------
struct RandomWalkReader {

  // As with the writer, the constructor will take input arguments for all member variables.
  RandomWalkReader(int ncoord_in, int bits_in, double fluctuation_in, RandomNumberKind style_in,
                   const llint* xcrd_in, const llint* ycrd_in, const ullint2* rng_stt_in);

  // The presence of const members prevents the default copy and move assignment operators from
  // working.  However, the default copy and move constructors are still valid.
  //
  // Arguments:
  //   original:  The original object to copy or move
  RandomWalkReader(const RandomWalkReader &original) = default;
  RandomWalkReader(RandomWalkReader &&original) = default;

  const int ncoord;              // The number of (particle) coordinates
  const int bits;                // Number of bits after the point in the internal, fixed-precision
                                 //   representation of coordinates
  const double fluctuation;      // Size of fluctuations to add with each random walk step.  This
                                 //   is the width of the range (the range is centered on zero) in
                                 //   a uniform random walk distribution or the sigma width of the
                                 //   Gaussian (also centered on zero) in a normal random walk.
  const RandomNumberKind style;  // The type of random walk, whether based on a uniform or a normal
                                 //   (Gaussian) distribution
  const llint* xcrd;             // Cartesian X coordinates of all particles
  const llint* ycrd;             // Cartesian Y coordinates of all particles
  const ullint2* rng_stt;        // Random number state vectors driving each particle.  While the
                                 //   particles cannot be moved with the read-only abstract, this
                                 //   pointer is included for reference if the developer needed to
                                 //   inspect the random number states.
};

//-------------------------------------------------------------------------------------------------
// A prototypical STORMM class, containing some data and a few accessors to perform a random walk
// simulation in two dimensions
//-------------------------------------------------------------------------------------------------
class RandomWalk {
public:

  // The constructor will let us size various arrays in the data, set the random seed, and
  // control the magnitude of random fluctuations.
  RandomWalk(int coordinate_count_in, int bits_in = 24, int prng_seed_in = 1083674,
             double fluctation_in = 1.0,
             RandomNumberKind fluctuation_style_in = RandomNumberKind::GAUSSIAN,
             const GpuDetails &gpu = null_gpu);

  // Accessor for the number of coordinates held in the object
  int getCoordinateCount() const;

  // Accessor for the random number seed used to create the coordinate set
  int getRandomSeed() const;

  // Accessor for the magnitude of fluctuations used in placing coordinates
  double getFluctuation() const;

  // Accessor for the type of random fluctuations in coordinates
  RandomNumberKind getFluctuationStyle() const;

  // Accessor for one of the coordinates
  //
  // Arguments:
  //   index:    The index of the coordinate of interest.  This will be checked for validity.
  double2 getCoordinate(int index, HybridTargetLevel tier = HybridTargetLevel::HOST) const;

  // Methods to obtain the abstract.  The read-only abstract is produced in the case of a
  // const-qualified object, the mutable abstract otherwise.
  //
  // Arguments:
  //
  RandomWalkWriter data(HybridTargetLevel tier = HybridTargetLevel::HOST);
  const RandomWalkReader data(HybridTargetLevel tier = HybridTargetLevel::HOST) const;

  // Advance the coordinates by a given number of steps.
  //
  // Arguments:
  //   step_count:  The number of steps to advance
  //   tier:        Indicate whether to run the random walk on the CPU or on the GPU
  void advance(int step_count = 1, HybridTargetLevel tier = HybridTargetLevel::HOST,
               const GpuDetails &gpu = null_gpu);

private:

  // The number of coordinates held by the class is defined when constructing the object.
  int coordinate_count;

  // The precision with which coordinates will be stored.  Convert numbers to real and divide by
  // 2 ^ (bits) to get the actual value of any coordinate.
  int bits;

  // The power and inverse power of two by which to multiply coordinates in order to convert
  // between their real-valued and fixed-precision representations
  double fp_scale;
  double fp_inv_scale;

  // The random number seed used to create the coordinate set.  A Xoroshiro128+ generator will be
  // used, although there are other choices.  (The Xoroshiro128+ generator does not pass Big Crush,
  // but it comes close, and for most purposes in computational chemistry it would be adequate
  // anyway.)
  int prng_seed;

  // The magnitude of random fluctuations, set during construction, will be remembered so that
  // it can be accessed later.
  double fluctuation;

  // Coordinates fluctuate according to some random scheme: a Gaussian or uniform distribution
  RandomNumberKind fluctuation_style;

  // Coordinates held by this class lie in the (x, y) plane.
  Hybrid<llint> x_coordinates;
  Hybrid<llint> y_coordinates;
  Hybrid<llint> storage;

  // Random number state vectors for each particle
  Hybrid<ullint2> rng_states;
  
  // A private member function will handle the allocation and assignment of the coordinates.
  void allocate(const GpuDetails &gpu);
};

} // namespace tutorial

#endif
