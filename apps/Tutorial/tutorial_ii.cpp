#include <string>
#include <vector>
#include "../../src/copyright.h"
#include "../../src/Constants/behavior.h"
#include "../../src/Constants/hpc_bounds.h"
#include "../../src/Accelerator/gpu_details.h"
#include "../../src/Accelerator/gpu_enumerators.h"
#include "../../src/Accelerator/hpc_config.h"
#include "../../src/Accelerator/hybrid.h"
#include "../../src/DataTypes/common_types.h"
#include "../../src/DataTypes/stormm_vector_types.h"
#include "../../src/Math/vector_ops.h"
#include "../../src/Parsing/parse.h"
#include "../../src/Parsing/parsing_enumerators.h"
#include "../../src/Random/random.h"
#include "../../src/Reporting/error_format.h"
#include "../../src/UnitTesting/stopwatch.h"
#include "randomwalk.h"

using stormm::card::GpuDetails;
using stormm::card::HpcConfig;
using stormm::card::Hybrid;
using stormm::card::HybridTargetLevel;
using stormm::constants::CaseSensitivity;
using stormm::constants::ExceptionResponse;
using stormm::constants::warp_size_int;
using stormm::constants::large_block_size;
using stormm::data_types::llint;
using stormm::errors::rtErr;
using stormm::parse::NumberFormat;
using stormm::parse::strcmpCased;
using stormm::parse::verifyContents;
using stormm::testing::StopWatch;
using namespace tutorial;

//-------------------------------------------------------------------------------------------------
// main
//-------------------------------------------------------------------------------------------------
int main(int argc, const char* argv[]) {

  // Boot up the GPU
#ifdef STORMM_USE_HPC
  const HpcConfig gpu_config(ExceptionResponse::WARN);
  const std::vector<int> my_gpus = gpu_config.getGpuDevice(1);
  const GpuDetails gpu = gpu_config.getGpuInfo(my_gpus[0]);
  Hybrid<int> force_gpu_to_engage(1);
#else
  const GpuDetails gpu = null_gpu;
#endif

  // Read in some command-line variables.  This is a simple C-based CLI.
  if (argc < 3) {
    printf("Usage: %s\n"
           "       [ -n number of particles ] [ -f fluctuation ] [ -s step kind ]\n",
           argv[0]);
    printf("  -n : Select the number of particles to simulate (default 50)\n");
    printf("  -f : Indicate the step size (unitless, default 1.0)\n");
    printf("  -s : Indicate whether steps should involve UNIFORM or GAUSSIAN random numbers\n");
    printf("       (default GAUSSIAN)\n");
    printf("  -x : The number of steps to simulate (default 100)\n");
    printf("  -r : The number of particles to report (default 4, indices will be distributed\n");
    printf("       throughout the collection)\n");
    exit(0);
  }
  int particle_count = 50;
  double step_size = 1.0;
  RandomNumberKind step_style = RandomNumberKind::GAUSSIAN;
  int step_count = 100;
  int report_count = 4;
  for (int i = 1; i < argc - 1; i++) {
    if (strcmpCased(argv[i], "-n", CaseSensitivity::NO)) {
      if (verifyContents(argv[i + 1], NumberFormat::INTEGER)) {
        particle_count = atoi(argv[i + 1]);
      }
      else {
        rtErr("Unrecognized integer " + std::string(argv[i + 1]) + " for the particle count");
      }
      i++;
    }
    else if (strcmpCased(argv[i], "-f", CaseSensitivity::NO)) {
      if (verifyContents(argv[i + 1], NumberFormat::STANDARD_REAL) ||
          verifyContents(argv[i + 1], NumberFormat::SCIENTIFIC)) {
        step_size = atof(argv[i + 1]);
      }
      else {
        rtErr("Unrecognized real value " + std::string(argv[i + 1]) + " for the particle count");
      }
      i++;
    }
    else if (strcmpCased(argv[i], "-x", CaseSensitivity::NO)) {
      if (verifyContents(argv[i + 1], NumberFormat::INTEGER)) {
        step_count = atoi(argv[i + 1]);
      }
      else {
        rtErr("Unrecognized integer " + std::string(argv[i + 1]) + " for the step count");
      }
      i++;
    }
    else if (strcmpCased(argv[i], "-s", CaseSensitivity::NO)) {
      if (strcmpCased(argv[i + 1], "gaussian")) {
        step_style = RandomNumberKind::GAUSSIAN;
      }
      else if (strcmpCased(argv[i + 1], "uniform")) {
        step_style = RandomNumberKind::UNIFORM;
      }
      else {
        rtErr("Unrecognized particle move type '" + std::string(argv[i + 1]) + "'");
      }
    }
    else if (strcmpCased(argv[i], "-r", CaseSensitivity::NO)) {
      if (verifyContents(argv[i + 1], NumberFormat::INTEGER)) {
        report_count = atoi(argv[i + 1]);
      }
      else {
        rtErr("Unrecognized integer " + std::string(argv[i + 1]) + " for the particle count");
      }
      i++;
    }
    else {
      rtErr("Unrecognized command line argument '" + std::string(argv[i]) + "'");
    }
  }

  // Create the simulation class object
  RandomWalk rw(particle_count, 24, 1083674, step_size, step_style, gpu);

  // Select a subset of the particles to report, and print their initial positions
  std::vector<int> report_indices(report_count);
  int cidx = 0;
  for (int i = 0; i < report_count; i++) {
    report_indices[i] = cidx;
    cidx += particle_count / report_count;
  }
  printf("\nInitial coordinates for selected particles:\n");
  printf("                      X (host)    Y (host)");
#ifdef STORMM_USE_HPC
  printf("    X (device)  Y (device)\n");
#else
  printf("\n");
#endif
  for (int i = 0; i < report_count; i++) {
    const int pidx = report_indices[i];
    const double2 pcrd_host = rw.getCoordinate(pidx, HybridTargetLevel::HOST);
    printf("  Particle %6d :  %10.4lf  %10.4lf", pidx, pcrd_host.x, pcrd_host.y);
#ifdef STORMM_USE_HPC
    const double2 pcrd_devc = rw.getCoordinate(pidx, HybridTargetLevel::DEVICE);
    printf("   %10.4lf  %10.4lf\n", pcrd_devc.x, pcrd_devc.y);
#else
    printf("\n");
#endif
  }

  // Advance the simulation on the CPU and, if applicable, on the GPU as well
  rw.advance(step_count, HybridTargetLevel::HOST);
#ifdef STORMM_USE_HPC
  rw.advance(step_count, HybridTargetLevel::DEVICE, gpu);
#endif
  printf("\nFinal coordinates for selected particles:\n");
  printf("                      X (host)    Y (host)");
#ifdef STORMM_USE_HPC
  printf("    X (device)  Y (device)\n");
#else
  printf("\n");
#endif
  for (int i = 0; i < report_count; i++) {
    const int pidx = report_indices[i];
    const double2 pcrd_host = rw.getCoordinate(pidx, HybridTargetLevel::HOST);
    printf("  Particle %6d :  %10.4lf  %10.4lf", pidx, pcrd_host.x, pcrd_host.y);
#ifdef STORMM_USE_HPC
    const double2 pcrd_devc = rw.getCoordinate(pidx, HybridTargetLevel::DEVICE);
    printf("   %10.4lf  %10.4lf\n", pcrd_devc.x, pcrd_devc.y);
#else
    printf("\n");
#endif
  }

  // Return success
  return 0;
}
