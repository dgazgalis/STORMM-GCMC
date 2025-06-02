#include "../../src/copyright.h"
#include "../../src/Accelerator/gpu_enumerators.h"
#include "../../src/Constants/hpc_bounds.h"
#include "../../src/Math/rounding.h"
#include "../../src/Random/random.h"
#include "../../src/Random/hpc_random.h"
#include "../../src/Reporting/error_format.h"
#include "randomwalk.h"
#ifdef STORMM_USE_HPC
#include "hpc_randomwalk.h"
#endif

namespace tutorial {

using stormm::constants::warp_size_int;
using stormm::card::HybridKind;
using stormm::data_types::ullint;
using stormm::errors::rtErr;
using stormm::stmath::roundUp;
using stormm::random::Xoroshiro128pGenerator;
using stormm::random::initXoroshiro128pArray;

//-------------------------------------------------------------------------------------------------
RandomWalkWriter::RandomWalkWriter(const int ncoord_in, const int bits_in,
                                   const double fluctuation_in, const RandomNumberKind style_in,
                                   llint* xcrd_in, llint* ycrd_in, ullint2* rng_stt_in) :
    ncoord{ncoord_in}, bits{bits_in}, fluctuation{fluctuation_in}, style{style_in},
    xcrd{xcrd_in}, ycrd{ycrd_in}, rng_stt{rng_stt_in}
{}

//-------------------------------------------------------------------------------------------------
RandomWalkReader::RandomWalkReader(const int ncoord_in, const int bits_in,
                                   const double fluctuation_in, const RandomNumberKind style_in,
                                   const llint* xcrd_in, const llint* ycrd_in,
                                   const ullint2* rng_stt_in) :
    ncoord{ncoord_in}, bits{bits_in}, fluctuation{fluctuation_in}, style{style_in},
    xcrd{xcrd_in}, ycrd{ycrd_in}, rng_stt{rng_stt_in}
{}

//-------------------------------------------------------------------------------------------------
RandomWalk::RandomWalk(const int coordinate_count_in, const int bits_in, const int prng_seed_in,
                       const double fluctuation_in, const RandomNumberKind fluctuation_style_in,
                       const GpuDetails &gpu) :
    coordinate_count{coordinate_count_in}, bits{bits_in},
    fp_scale{pow(2.0, bits_in)},
    fp_inv_scale{pow(2.0, -bits_in)},
    prng_seed{prng_seed_in}, fluctuation{fluctuation_in},
    fluctuation_style{fluctuation_style_in},
    x_coordinates{HybridKind::POINTER, "x_coord"},
    y_coordinates{HybridKind::POINTER, "y_coord"},
    storage{HybridKind::ARRAY, "coord_storage"},
    rng_states{HybridKind::ARRAY, "rng_state_vectors"}
{
  // Lay out the memory needed to hold the coordinate contents.
  allocate(gpu);

#ifdef STORMM_USE_HPC
  // Initialize random number states using the GPU.  The process involves the CPU coining a
  // temporary Xoroshiro128+ generator, then use the CPU to evaluate its long jump and seed a
  // handful of generator states on the host side of the random number states array (rng_states).
  // These seeds will be uploaded to the GPU device so that up to 1024 threads can continue to fill
  // out rng_states using the Xoroshiro128+ generator's short jump.  Once the seeding is complete,
  // each random number sequence created based on one of the state vectors will produce at least
  // 2^64 unique random numbers before beginning to repeat of the other sequences (probably much,
  // much more).  The results stored on the GPU are back-ported to the host so that both arrays of
  // random number generator states are ready to drive identical random sequences.
  //
  // All of the inputs to this function should be familiar, but for one number, 10, appearing
  // before the GPU specifications.  That's the number of "scrub cycles," iterations into the
  // random sequence taken before the state is stored.  Having each sequence walk forward a bit is
  // a good practice to ensure the quality of random numbers when starting from a seed with little
  // detail, i.e. small integers that are mostly zero in the high bits.
  //
  // In reality, a miniscule correlation between these random sequences from a Xoroshiro128+
  // generator can be detected, which is why it's not the best choice for seeding tens of thousands
  // of generators to drive each atom of a simulation (or each atom out of a collection of
  // simulations).  The Xoshiro256++ generator is preferred, but it takes longer to evaluate
  // (perhaps twice the memory, memory bandwidth, and calculations, but this is trivial as either
  // generator takes very little time).
  initXoroshiro128pArray(&rng_states, prng_seed, 10, gpu);
#else
  // Initialize random number states using only the CPU.  The process involves using the temporary
  // random number generator which is already advanced by 10 scrub cycles (see above), then
  // executing its short jump function until the array is filled out.  This is the same idea, but
  // a slightly different algorithm than what the GPU does--the results will not be the same if the
  // program is run in CPU-only or GPU-enabled modes.
  Xoroshiro128pGenerator xrs(prng_seed, 10);
  for (int i = 0; i < coordinate_count; i++) {
    rng_states.putHost(xrs.revealState(), i);
    xrs.jump();
  }
#endif
  // Initialize the coordinates of particles on the CPU and on the GPU by executing the advance()
  // member function once, starting from numbers which are initialized to zero in the Hybrid array.
  advance();
#ifdef STORMM_USE_HPC
  advance(1, HybridTargetLevel::DEVICE, gpu);
#endif
}

//-------------------------------------------------------------------------------------------------
int RandomWalk::getCoordinateCount() const {
  return coordinate_count;
}

//-------------------------------------------------------------------------------------------------
double RandomWalk::getFluctuation() const {
  return fluctuation;
}

//-------------------------------------------------------------------------------------------------
RandomNumberKind RandomWalk::getFluctuationStyle() const {
  return fluctuation_style;
}

//-------------------------------------------------------------------------------------------------
double2 RandomWalk::getCoordinate(const int index, const HybridTargetLevel tier) const {
  if (index < 0 || index >= coordinate_count) {
    rtErr("Index " + std::to_string(index) + " is invalid for a series of " +
          std::to_string(coordinate_count) + " points.", "RandomWalk", "getCoordinate");
  }
  const size_t index_zu = index;
  switch (tier) {
  case HybridTargetLevel::HOST:
    return { static_cast<double>(x_coordinates.readHost(index_zu)) * fp_inv_scale,
             static_cast<double>(y_coordinates.readHost(index_zu)) * fp_inv_scale };
#ifdef STORMM_USE_HPC
  case HybridTargetLevel::DEVICE:
    return { static_cast<double>(x_coordinates.readDevice(index_zu)) * fp_inv_scale,
             static_cast<double>(y_coordinates.readDevice(index_zu)) * fp_inv_scale };
#endif
  }
  __builtin_unreachable();
}

//-------------------------------------------------------------------------------------------------
RandomWalkWriter RandomWalk::data(const HybridTargetLevel tier) {
  return RandomWalkWriter(coordinate_count, bits, fluctuation, fluctuation_style,
                          x_coordinates.data(tier), y_coordinates.data(tier),
                          rng_states.data(tier));
}

//-------------------------------------------------------------------------------------------------
const RandomWalkReader RandomWalk::data(const HybridTargetLevel tier) const {
  return RandomWalkReader(coordinate_count, bits, fluctuation, fluctuation_style,
                          x_coordinates.data(tier), y_coordinates.data(tier),
                          rng_states.data(tier));
}

//-------------------------------------------------------------------------------------------------
void RandomWalk::advance(const int step_count, const HybridTargetLevel tier,
                         const GpuDetails &gpu) {
  switch (tier) {
  case HybridTargetLevel::HOST:
    {
      // Create a temporary random number generator to process CPU calculations.  The seed is
      // irrelevant, as its state will be replaced with each coordinate set.
      Xoroshiro128pGenerator xrs(10229384, 0);
      for (int i = 0; i < coordinate_count; i++) {
        xrs.setState(rng_states.readHost(i));
        for (int j = 0; j < step_count; j++) {
          const llint ix_crd = x_coordinates.readHost(i);
          const llint iy_crd = y_coordinates.readHost(i);
          switch (fluctuation_style) {
          case RandomNumberKind::GAUSSIAN:
            {
              const llint ix_bump = llround(xrs.gaussianRandomNumber() * fluctuation * fp_scale);
              const llint iy_bump = llround(xrs.gaussianRandomNumber() * fluctuation * fp_scale);
              x_coordinates.putHost(ix_crd + ix_bump, i);
              y_coordinates.putHost(iy_crd + iy_bump, i);
            }
            break;
          case RandomNumberKind::UNIFORM:
            {
              const llint ix_bump = llround((0.5 - xrs.uniformRandomNumber()) * fluctuation *
                                            fp_scale);
              const llint iy_bump = llround((0.5 - xrs.uniformRandomNumber()) * fluctuation *
                                            fp_scale);
              x_coordinates.putHost(ix_crd + ix_bump, i);
              y_coordinates.putHost(iy_crd + iy_bump, i);
            }
            break;
          }
        }
        rng_states.putHost(xrs.revealState(), i);
      }
    }
    break;
#ifdef STORMM_USE_HPC
  case HybridTargetLevel::DEVICE:
    {
      RandomWalkWriter rww = this->data(HybridTargetLevel::DEVICE);
      launchRandomWalkAdvance(step_count, &rww, gpu);
    }
    break;
#endif
  }
}

//-------------------------------------------------------------------------------------------------
void RandomWalk::allocate(const GpuDetails &gpu) {

  // Compute the padded length of each coordinate array
  const int padded_count = roundUp(coordinate_count, warp_size_int);
  storage.resize(2 * padded_count);
  x_coordinates.setPointer(&storage, 0, coordinate_count);
  y_coordinates.setPointer(&storage, padded_count, coordinate_count);
  rng_states.resize(coordinate_count);
}

} // namespace tutorial
