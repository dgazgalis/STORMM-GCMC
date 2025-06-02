// -*-c++-*-
#include "../../src/copyright.h"
#include "../../src/Accelerator/gpu_details.h"
#include "randomwalk.h"

namespace tutorial {
  
using stormm::card::GpuDetails;

//-------------------------------------------------------------------------------------------------
// Launch a kernel to carry out particle movement (coordinate advancement) in the prototypical
// STORMM class.
//
// Arguments:
//   step_count:  The number of steps to advance
//   rww:         A writeable abstract for the main simulation class object
//   gpu:         Details of the GPU that will carry out the simulation
//-------------------------------------------------------------------------------------------------
void launchRandomWalkAdvance(int step_count, RandomWalkWriter *rww, const GpuDetails &gpu);

} // namespace tutorial
