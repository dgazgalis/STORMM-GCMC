#include <string>
#include <vector>
#include "../../src/copyright.h"
#include "../../src/Constants/behavior.h"
#include "../../src/Accelerator/gpu_details.h"
#include "../../src/Accelerator/gpu_enumerators.h"
#include "../../src/Accelerator/hpc_config.h"
#include "../../src/Accelerator/hybrid.h"
#include "../../src/DataTypes/common_types.h"
#include "../../src/DataTypes/stormm_vector_types.h"
#include "../../src/Math/summation.h"
#include "../../src/Math/vector_ops.h"
#include "../../src/UnitTesting/stopwatch.h"
#include "../../src/UnitTesting/approx.h"
#include "../../src/UnitTesting/unit_test.h"
#include "../../src/UnitTesting/test_environment.h"
#include "hpc_tutorial_i.h"

using stormm::card::GpuDetails;
using stormm::card::HpcConfig;
using stormm::card::Hybrid;
using stormm::card::HybridTargetLevel;
using stormm::constants::ExceptionResponse;
using stormm::data_types::int_type_index;
using stormm::stmath::sum;
using stormm::testing::StopWatch;

//-------------------------------------------------------------------------------------------------
// main
//-------------------------------------------------------------------------------------------------
int main(int argc, const char* argv[]) {

  // Initialize a StopWatch to perform time tracking
  StopWatch the_clock("STORMM Tutorial I");
  const int gpu_asgn_tm = the_clock.addCategory("Assign a GPU");
  const int gpu_prep_tm = the_clock.addCategory("Prep the GPU");
  the_clock.assignTime();
#ifdef STORMM_USE_HPC
  const HpcConfig gpu_config(ExceptionResponse::WARN);
  const std::vector<int> my_gpus = gpu_config.getGpuDevice(1);
  const GpuDetails gpu = gpu_config.getGpuInfo(my_gpus[0]);
  the_clock.assignTime("Assign a GPU");
  Hybrid<int> force_gpu_to_engage(1);
#  ifdef STORMM_USE_CUDA
  cudaDeviceSynchronize();
#  endif
  the_clock.assignTime("Prep the GPU");
  printf("This program is running on a %s card:\n", gpu.getCardName().c_str());
  printf("  Major.minor architecture version: %d.%d\n", gpu.getArchMajor(), gpu.getArchMinor());
  printf("  Streaming multiprocessors: %d\n", gpu.getSMPCount());
  printf("  Card RAM: %d megabtyes\n", gpu.getCardRam());
  printf("  Global cache size: %d bytes\n\n", gpu.getGlobalCacheSize());
#else
  GpuDetails gpu = null_gpu;
  printf("This program is running in CPU-only mode.  No GPU is available.\n\n");
#endif
  
  // Create an array of integers on the CPU host and (if available) on the GPU device
  const int int_experiment_tm = the_clock.addCategory("Experiment with integers");
  Hybrid<int> xferable_integers(128, "Test_Int_Array");
  int ri = -5;
  for (size_t i = 0; i < xferable_integers.size(); i++) {
    xferable_integers.putHost(ri, i);
    if (i == 0) {
      ri++;
    }
    else if (ri == 16) {
      ri -= 2;
    }
    else if (ri == -8) {
      ri++;
    }
    else {
      if (xferable_integers.readHost(i - 1) < ri) {
        ri++;
      }
      else {
        ri -= 2;
      }
    }
  }
  the_clock.assignTime(int_experiment_tm);

  // Grab a pointer to the array's host-side data and print the contents
  const int* host_xi_ptr = xferable_integers.data(HybridTargetLevel::HOST);
  const int nxi = xferable_integers.size();
  printf("Contents of xferable_integers:\n  ");
  for (int i = 0; i < nxi; i++) {
    printf(" %3d", host_xi_ptr[i]);
    if ((i + 1) % 16 == 0 || i == nxi - 1) {
      printf("   (%3d)\n  ", i + 1);
    }
  }
  printf("\n");

  // Allocate a buffer for the answer
  Hybrid<int> sum_of_xi(gpu.getSMPCount(), "Our_Answer");

#ifdef STORMM_USE_HPC
  // Upload the data to the GPU, then grab pointers to the data and allocated answer on the GPU
  xferable_integers.upload();
  const int* devc_xi_ptr = xferable_integers.data(HybridTargetLevel::DEVICE);
  const void* vdevc_xi_ptr = reinterpret_cast<const void*>(devc_xi_ptr);
  int* sum_ptr = sum_of_xi.data(HybridTargetLevel::DEVICE);
  void* vsum_ptr = reinterpret_cast<void*>(sum_ptr);

  // Launch a kernel via a call to a function compiled in the associated CUDA unit
  wrapTheSummationLaunch(vdevc_xi_ptr, nxi, vsum_ptr, int_type_index, gpu);

  // Download the result
  printf("The sum of the set of integers on the host is:          %d\n",
         sum<int>(host_xi_ptr, xferable_integers.size()));
  printf("Before downloading, the sum of the answer buffer reads: %d\n",
         sum<int>(sum_of_xi.data(HybridTargetLevel::HOST), gpu.getSMPCount()));
  sum_of_xi.download();
  printf("After downloading, the sum of the answer buffer reads:  %d\n\n",
         sum<int>(sum_of_xi.data(HybridTargetLevel::HOST), gpu.getSMPCount()));
#endif
  
  // Print the timings
  the_clock.assignTime();
  the_clock.printResults();
  
  // Return success
  return 0;
}
