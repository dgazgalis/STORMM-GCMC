// -*-c++-*-
#ifndef STORMM_NCMC_PROTOCOL_H
#define STORMM_NCMC_PROTOCOL_H

#include <vector>
#include "copyright.h"
#include "gcmc_molecule.h"  // For VDW_COUPLING_THRESHOLD

namespace stormm {
namespace sampling {

/// \brief Defines the protocol for Nonequilibrium Candidate Monte Carlo (NCMC) switching
///
/// This class manages the lambda schedule and propagation settings for NCMC moves,
/// which reduce the computational cost of alchemical transformations by using
/// nonequilibrium switching protocols.
class NCMCProtocol {
public:

  /// \brief Default constructor creates a simple instantaneous switching protocol
  NCMCProtocol();

  /// \brief Constructor with linear lambda schedule
  ///
  /// Creates a protocol with linearly spaced lambda values from 0 to 1.
  ///
  /// \param n_pert_steps           Number of lambda perturbation steps
  /// \param n_prop_steps_per_pert  MD steps between lambda changes
  /// \param timestep               Integration timestep (femtoseconds)
  /// \param vdw_coupling_end       Lambda value where VDW is fully coupled (default VDW_COUPLING_THRESHOLD)
  NCMCProtocol(int n_pert_steps, int n_prop_steps_per_pert, double timestep,
               double vdw_coupling_end = VDW_COUPLING_THRESHOLD);

  /// \brief Constructor with custom lambda schedule
  ///
  /// \param lambdas                Custom lambda schedule (must start at 0, end at 1)
  /// \param n_prop_steps_per_pert  MD steps between lambda changes
  /// \param timestep               Integration timestep (femtoseconds)
  /// \param vdw_coupling_end       Lambda value where VDW is fully coupled (default VDW_COUPLING_THRESHOLD)
  NCMCProtocol(const std::vector<double> &lambdas, int n_prop_steps_per_pert,
               double timestep, double vdw_coupling_end = VDW_COUPLING_THRESHOLD);

  /// \brief Get the number of perturbation steps
  ///
  /// \return Number of lambda changes in the protocol
  int getPerturbationSteps() const;

  /// \brief Get the number of propagation steps per perturbation
  ///
  /// \return Number of MD steps between lambda changes
  int getPropagationStepsPerPerturbation() const;

  /// \brief Set the number of propagation steps per perturbation
  ///
  /// \param n_steps  New number of MD steps between lambda changes
  void setPropagationStepsPerPerturbation(int n_steps);

  /// \brief Get the integration timestep
  ///
  /// \return Timestep in femtoseconds
  double getTimestep() const;

  /// \brief Set the integration timestep
  ///
  /// \param dt  New timestep in femtoseconds
  void setTimestep(double dt);

  /// \brief Get the total switching time
  ///
  /// \return Total protocol time in picoseconds
  double getSwitchingTime() const;

  /// \brief Get the lambda schedule
  ///
  /// \return Vector of lambda values for the protocol
  const std::vector<double>& getLambdaSchedule() const;

  /// \brief Set a custom lambda schedule
  ///
  /// The schedule must start at 0 and end at 1. The number of values
  /// determines the number of perturbation steps.
  ///
  /// \param lambdas  New lambda schedule
  void setLambdaSchedule(const std::vector<double> &lambdas);

  /// \brief Generate a linear lambda schedule
  ///
  /// \param n_steps  Number of perturbation steps
  void generateLinearSchedule(int n_steps);

  /// \brief Generate a sigmoidal lambda schedule
  ///
  /// Creates a smooth S-curve that spends more time at the endpoints.
  ///
  /// \param n_steps    Number of perturbation steps
  /// \param steepness  Steepness of the sigmoid (higher = sharper transition)
  void generateSigmoidalSchedule(int n_steps, double steepness = 1.0);

  /// \brief Get lambda value at a specific step
  ///
  /// \param step  Step index (0 to n_pert_steps)
  /// \return Lambda value at the specified step
  double getLambda(int step) const;

  /// \brief Get the VDW coupling endpoint
  ///
  /// In two-stage coupling, VDW is coupled from lambda 0 to this value,
  /// then electrostatics from this value to 1.
  ///
  /// \return Lambda value where VDW coupling is complete
  double getVdwCouplingEnd() const;

  /// \brief Set the VDW coupling endpoint
  ///
  /// \param lambda_vdw_end  New VDW endpoint (typically VDW_COUPLING_THRESHOLD)
  void setVdwCouplingEnd(double lambda_vdw_end);

  /// \brief Convert combined lambda to VDW and electrostatic components
  ///
  /// For two-stage coupling: VDW first (0 to vdw_coupling_end),
  /// then electrostatics (vdw_coupling_end to 1.0).
  ///
  /// \param lambda_combined  Combined lambda value [0, 1]
  /// \param lambda_vdw       Output VDW lambda [0, 1]
  /// \param lambda_ele       Output electrostatic lambda [0, 1]
  void splitLambda(double lambda_combined, double &lambda_vdw, double &lambda_ele) const;

  /// \brief Get the total number of MD steps in the protocol
  ///
  /// \return Total MD steps including initial equilibration
  int getTotalMDSteps() const;

  /// \brief Check if the protocol is valid
  ///
  /// Validates that the lambda schedule starts at 0, ends at 1,
  /// and is monotonically increasing.
  ///
  /// \return True if the protocol is valid
  bool isValid() const;

  /// \brief Get a string description of the protocol
  ///
  /// \return Human-readable description of the protocol settings
  std::string getDescription() const;

private:
  int n_pert_steps_;                 ///< Number of lambda perturbation steps
  int n_prop_steps_per_pert_;        ///< MD steps between lambda changes
  double timestep_;                   ///< Integration timestep (fs)
  double vdw_coupling_end_;          ///< Lambda where VDW fully coupled (default VDW_COUPLING_THRESHOLD)
  double switching_time_;             ///< Total protocol time (ps)
  std::vector<double> lambdas_;      ///< Lambda schedule [0, ..., 1]

  /// \brief Update the cached switching time
  void updateSwitchingTime();

  /// \brief Validate the lambda schedule
  void validateSchedule();
};

} // namespace sampling
} // namespace stormm

#endif