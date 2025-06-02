// -*-c++-*-
#include "../../src/copyright.h"
#include "../../src/Accelerator/gpu_enumerators.h"
#include "../../src/Constants/hpc_bounds.h"
#include "../../src/DataTypes/common_types.h"
#include "../../src/DataTypes/stormm_vector_types.h"
#include "../../src/Random/random.h"
#include "hpc_randomwalk.h"

namespace tutorial {

using stormm::constants::large_block_size;
using stormm::data_types::llint;
using stormm::data_types::ullint;
using stormm::data_types::ullint2;
using stormm::data_types::ullint4;

// These imported constant expressions are needed to let code in the .cui file below compile.  It
// would suffice to say "using namespace stormm::random" but all tutorials try to avoid that
// general import, with the exception of the tutorial namespace itself, to make it clear where each
// element originates.
using stormm::random::xrs128p_jump_i;
using stormm::random::xrs128p_jump_ii;
using stormm::random::xrs256pp_jump_i;
using stormm::random::xrs256pp_jump_ii;
using stormm::random::xrs256pp_jump_iii;
using stormm::random::xrs256pp_jump_iv;
using stormm::random::rng_unit_bin_offset;
using stormm::random::rng_unit_bin_offset_f;

#include "../../src/Random/xor_shift_rng.cui"

//-------------------------------------------------------------------------------------------------
__global__ void __launch_bounds__(large_block_size, 1)
kRandomWalkAdvance(const int step_count, RandomWalkWriter rww) {
  const int thread_stride = blockDim.x * gridDim.x;
  const double fp_scale = pow(2.0, rww.bits);
  for (int i = threadIdx.x + (blockIdx.x * blockDim.x); i < rww.ncoord; i += thread_stride) {
    ullint2 stti = rww.rng_stt[i];
    switch (rww.style) {
    case RandomNumberKind::GAUSSIAN:
      for (int j = 0; j < step_count; j++) {
        const llint ix_bump = __double2ll_rn(xoroshiro128p_normal(&stti) * rww.fluctuation *
                                             fp_scale);
        const llint iy_bump = __double2ll_rn(xoroshiro128p_normal(&stti) * rww.fluctuation *
                                             fp_scale);
        rww.xcrd[i] += ix_bump;
        rww.ycrd[i] += iy_bump;
      }
      break;
    case RandomNumberKind::UNIFORM:
      for (int j = 0; j < step_count; j++) {
        const llint ix_bump = __double2ll_rn((0.5 - xoroshiro128p_uniform(&stti)) *
                                             rww.fluctuation * fp_scale);
        const llint iy_bump = __double2ll_rn((0.5 - xoroshiro128p_uniform(&stti)) *
                                             rww.fluctuation * fp_scale);
        rww.xcrd[i] += ix_bump;
        rww.ycrd[i] += iy_bump;
      }
      break;
    }
    rww.rng_stt[i] = stti;
  }
}
                
//-------------------------------------------------------------------------------------------------
extern void launchRandomWalkAdvance(const int step_count, RandomWalkWriter *rww,
                                    const GpuDetails &gpu) {
  kRandomWalkAdvance<<<gpu.getSMPCount(), large_block_size>>>(step_count, *rww);
}

} // namespace tutorial
