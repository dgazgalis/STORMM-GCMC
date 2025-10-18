#include "copyright.h"
#include "Constants/symbol_values.h"
#include "pme_util_lambda.h"

namespace stormm {
namespace energy {

using symbols::pi;

//-------------------------------------------------------------------------------------------------
double pmeSelfEnergyLambda(const double* charges, const double* lambda_ele,
                           const double ewald_coeff, const int n_atoms,
                           const double lambda_threshold) {
  return pmeSelfEnergyLambda(charges, lambda_ele, ewald_coeff, 0, n_atoms, lambda_threshold);
}

//-------------------------------------------------------------------------------------------------
double pmeSelfEnergyLambda(const double* charges, const double* lambda_ele,
                           const double ewald_coeff, const int atom_start, const int atom_end,
                           const double lambda_threshold) {

  // E_self = -(α/√π) Σᵢ (qᵢ·λᵢ)²
  // Coefficient: -(α/√π)
  const double coeff = -ewald_coeff / sqrt(pi);

  double energy = 0.0;
  for (int i = atom_start; i < atom_end; i++) {
    const double lambda = lambda_ele[i];

    // Skip atoms below lambda threshold (ghost atoms)
    if (lambda < lambda_threshold) {
      continue;
    }

    // Compute lambda-scaled charge
    const double scaled_charge = charges[i] * lambda;

    // Add (q·λ)² contribution to self-energy
    energy += scaled_charge * scaled_charge;
  }

  return coeff * energy;
}

} // namespace energy
} // namespace stormm
