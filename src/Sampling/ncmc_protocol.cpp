#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string>
#include "copyright.h"
#include "Reporting/error_format.h"
#include "ncmc_protocol.h"

namespace stormm {
namespace sampling {

using errors::rtErr;

//-------------------------------------------------------------------------------------------------
NCMCProtocol::NCMCProtocol() :
  n_pert_steps_{0},
  n_prop_steps_per_pert_{0},
  timestep_{2.0},
  vdw_coupling_end_{VDW_COUPLING_THRESHOLD},
  switching_time_{0.0}
{
  // Create instantaneous switching (lambda jumps from 0 to 1)
  lambdas_ = {0.0, 1.0};
}

//-------------------------------------------------------------------------------------------------
NCMCProtocol::NCMCProtocol(int n_pert_steps, int n_prop_steps_per_pert, double timestep,
                           double vdw_coupling_end) :
  n_pert_steps_{n_pert_steps},
  n_prop_steps_per_pert_{n_prop_steps_per_pert},
  timestep_{timestep},
  vdw_coupling_end_{vdw_coupling_end}
{
  generateLinearSchedule(n_pert_steps);
  updateSwitchingTime();
}

//-------------------------------------------------------------------------------------------------
NCMCProtocol::NCMCProtocol(const std::vector<double> &lambdas, int n_prop_steps_per_pert,
                           double timestep, double vdw_coupling_end) :
  n_pert_steps_{static_cast<int>(lambdas.size()) - 1},
  n_prop_steps_per_pert_{n_prop_steps_per_pert},
  timestep_{timestep},
  vdw_coupling_end_{vdw_coupling_end},
  lambdas_{lambdas}
{
  validateSchedule();
  updateSwitchingTime();
}

//-------------------------------------------------------------------------------------------------
int NCMCProtocol::getPerturbationSteps() const {
  return n_pert_steps_;
}

//-------------------------------------------------------------------------------------------------
int NCMCProtocol::getPropagationStepsPerPerturbation() const {
  return n_prop_steps_per_pert_;
}

//-------------------------------------------------------------------------------------------------
void NCMCProtocol::setPropagationStepsPerPerturbation(int n_steps) {
  n_prop_steps_per_pert_ = n_steps;
  updateSwitchingTime();
}

//-------------------------------------------------------------------------------------------------
double NCMCProtocol::getTimestep() const {
  return timestep_;
}

//-------------------------------------------------------------------------------------------------
void NCMCProtocol::setTimestep(double dt) {
  timestep_ = dt;
  updateSwitchingTime();
}

//-------------------------------------------------------------------------------------------------
double NCMCProtocol::getSwitchingTime() const {
  return switching_time_;
}

//-------------------------------------------------------------------------------------------------
const std::vector<double>& NCMCProtocol::getLambdaSchedule() const {
  return lambdas_;
}

//-------------------------------------------------------------------------------------------------
void NCMCProtocol::setLambdaSchedule(const std::vector<double> &lambdas) {
  lambdas_ = lambdas;
  n_pert_steps_ = static_cast<int>(lambdas.size()) - 1;
  validateSchedule();
  updateSwitchingTime();
}

//-------------------------------------------------------------------------------------------------
void NCMCProtocol::generateLinearSchedule(int n_steps) {
  if (n_steps <= 0) {
    rtErr("Number of perturbation steps must be positive", "NCMCProtocol::generateLinearSchedule");
  }

  n_pert_steps_ = n_steps;
  lambdas_.clear();
  lambdas_.reserve(n_steps + 1);

  // Start from 0.05 instead of 0.0 to ensure molecule is included in energy calculations
  // with LAMBDA_GHOST_THRESHOLD = 0.01 optimization
  constexpr double MIN_LAMBDA = 0.05;
  for (int i = 0; i <= n_steps; i++) {
    const double frac = static_cast<double>(i) / static_cast<double>(n_steps);
    lambdas_.push_back(MIN_LAMBDA + frac * (1.0 - MIN_LAMBDA));
  }

  updateSwitchingTime();
}

//-------------------------------------------------------------------------------------------------
void NCMCProtocol::generateSigmoidalSchedule(int n_steps, double steepness) {
  if (n_steps <= 0) {
    rtErr("Number of perturbation steps must be positive",
          "NCMCProtocol::generateSigmoidalSchedule");
  }
  if (steepness <= 0.0) {
    rtErr("Steepness must be positive", "NCMCProtocol::generateSigmoidalSchedule");
  }

  n_pert_steps_ = n_steps;
  lambdas_.clear();
  lambdas_.reserve(n_steps + 1);

  for (int i = 0; i <= n_steps; i++) {
    // Map to [-1, 1]
    double x = 2.0 * static_cast<double>(i) / static_cast<double>(n_steps) - 1.0;

    // Apply sigmoid function: tanh(steepness * x)
    double sigmoid = std::tanh(steepness * x);

    // Map back to [0, 1]
    double lambda = 0.5 * (sigmoid + 1.0);

    lambdas_.push_back(lambda);
  }

  // Ensure endpoints are exactly 0.05 and 1 (start from 0.05 for LAMBDA_GHOST_THRESHOLD optimization)
  constexpr double MIN_LAMBDA = 0.05;
  lambdas_[0] = MIN_LAMBDA;
  lambdas_[n_steps] = 1.0;

  updateSwitchingTime();
}

//-------------------------------------------------------------------------------------------------
double NCMCProtocol::getLambda(int step) const {
  if (step < 0 || step >= static_cast<int>(lambdas_.size())) {
    rtErr("Step index " + std::to_string(step) + " out of range [0, " +
          std::to_string(lambdas_.size() - 1) + "]", "NCMCProtocol::getLambda");
  }
  return lambdas_[step];
}

//-------------------------------------------------------------------------------------------------
double NCMCProtocol::getVdwCouplingEnd() const {
  return vdw_coupling_end_;
}

//-------------------------------------------------------------------------------------------------
void NCMCProtocol::setVdwCouplingEnd(double lambda_vdw_end) {
  if (lambda_vdw_end <= 0.0 || lambda_vdw_end >= 1.0) {
    rtErr("VDW coupling endpoint must be in (0, 1)", "NCMCProtocol::setVdwCouplingEnd");
  }
  vdw_coupling_end_ = lambda_vdw_end;
}

//-------------------------------------------------------------------------------------------------
void NCMCProtocol::splitLambda(double lambda_combined, double &lambda_vdw,
                                double &lambda_ele) const {
  // Two-stage coupling: VDW first, then electrostatics
  if (lambda_combined <= vdw_coupling_end_) {
    lambda_vdw = lambda_combined / vdw_coupling_end_;
    lambda_ele = 0.0;
  } else {
    lambda_vdw = 1.0;
    lambda_ele = (lambda_combined - vdw_coupling_end_) / (1.0 - vdw_coupling_end_);
  }
}

//-------------------------------------------------------------------------------------------------
int NCMCProtocol::getTotalMDSteps() const {
  // Total steps = initial equilibration + (n_pert_steps * n_prop_steps_per_pert)
  return (n_pert_steps_ + 1) * n_prop_steps_per_pert_;
}

//-------------------------------------------------------------------------------------------------
bool NCMCProtocol::isValid() const {
  if (lambdas_.empty()) {
    return false;
  }

  // Check that schedule starts at 0.05 (for LAMBDA_GHOST_THRESHOLD optimization)
  constexpr double MIN_LAMBDA = 0.05;
  if (std::abs(lambdas_.front() - MIN_LAMBDA) > 1.0e-10) {
    return false;
  }

  // Check that schedule ends at 1
  if (std::abs(lambdas_.back() - 1.0) > 1.0e-10) {
    return false;
  }

  // Check monotonicity
  for (size_t i = 1; i < lambdas_.size(); i++) {
    if (lambdas_[i] < lambdas_[i - 1]) {
      return false;
    }
  }

  // Check all values are in [0, 1]
  for (double lambda : lambdas_) {
    if (lambda < -1.0e-10 || lambda > 1.0 + 1.0e-10) {
      return false;
    }
  }

  return true;
}

//-------------------------------------------------------------------------------------------------
std::string NCMCProtocol::getDescription() const {
  std::ostringstream oss;

  oss << "NCMC Protocol:\n";
  oss << "  Perturbation steps: " << n_pert_steps_ << "\n";
  oss << "  Propagation steps per perturbation: " << n_prop_steps_per_pert_ << "\n";
  oss << "  Timestep: " << timestep_ << " fs\n";
  oss << "  Total switching time: " << switching_time_ << " ps\n";
  oss << "  VDW coupling endpoint: " << vdw_coupling_end_ << "\n";
  oss << "  Lambda schedule: ";

  if (lambdas_.size() <= 10) {
    // Print all values if not too many
    for (size_t i = 0; i < lambdas_.size(); i++) {
      if (i > 0) oss << ", ";
      oss << lambdas_[i];
    }
  } else {
    // Print first and last few values
    for (int i = 0; i < 3; i++) {
      if (i > 0) oss << ", ";
      oss << lambdas_[i];
    }
    oss << ", ..., ";
    for (size_t i = lambdas_.size() - 3; i < lambdas_.size(); i++) {
      if (i > lambdas_.size() - 3) oss << ", ";
      oss << lambdas_[i];
    }
  }
  oss << "\n";

  return oss.str();
}

//-------------------------------------------------------------------------------------------------
void NCMCProtocol::updateSwitchingTime() {
  // Total time = (n_pert_steps + 1) * n_prop_steps_per_pert * timestep
  // The +1 accounts for initial equilibration
  const double total_steps = static_cast<double>((n_pert_steps_ + 1) * n_prop_steps_per_pert_);
  switching_time_ = total_steps * timestep_ / 1000.0;  // Convert fs to ps
}

//-------------------------------------------------------------------------------------------------
void NCMCProtocol::validateSchedule() {
  if (!isValid()) {
    rtErr("Invalid lambda schedule: must start at 0.05, end at 1, and be monotonically increasing",
          "NCMCProtocol::validateSchedule");
  }
}

} // namespace sampling
} // namespace stormm