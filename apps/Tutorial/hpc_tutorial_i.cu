// -*-c++-*-
#include "../../src/copyright.h"
#include "../../src/Constants/hpc_bounds.h"
#include "../../src/DataTypes/common_types.h"
#include "../../src/Math/hpc_summation.cuh"
#include "hpc_tutorial_i.h"

using stormm::constants::large_block_size;
using stormm::data_types::llint;
using stormm::data_types::int_type_index;
using stormm::data_types::llint_type_index;
using stormm::data_types::double_type_index;
using stormm::data_types::float_type_index;
using stormm::hpc_math::kSumVector;

//-------------------------------------------------------------------------------------------------
extern void wrapTheSummationLaunch(const void* vdata, const size_t n, void* vresult,
                                   const size_t ct_data, const GpuDetails &gpu) {
  if (ct_data == int_type_index) {
    const int* data = reinterpret_cast<const int*>(vdata);
    int* result = reinterpret_cast<int*>(vresult);
    kSumVector<int, int><<<gpu.getSMPCount(), large_block_size>>>(data, n, result);
  }
  else if (ct_data == llint_type_index) {
    const llint* data = reinterpret_cast<const llint*>(vdata);
    llint* result = reinterpret_cast<llint*>(vresult);
    kSumVector<llint, llint><<<gpu.getSMPCount(), large_block_size>>>(data, n, result);
  }
  else if (ct_data == double_type_index) {
    const double* data = reinterpret_cast<const double*>(vdata);
    double* result = reinterpret_cast<double*>(vresult);
    kSumVector<double, double><<<gpu.getSMPCount(), large_block_size>>>(data, n, result);
  }
  else if (ct_data == float_type_index) {
    const float* data = reinterpret_cast<const float*>(vdata);
    float* result = reinterpret_cast<float*>(vresult);
    kSumVector<float, float><<<gpu.getSMPCount(), large_block_size>>>(data, n, result);
  }
}
