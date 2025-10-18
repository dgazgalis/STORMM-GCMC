// -*-c++-*-
#ifndef STORMM_PME_UTIL_LAMBDA_H
#define STORMM_PME_UTIL_LAMBDA_H

#include "copyright.h"
#include <cmath>

namespace stormm {
namespace energy {

/// \brief Compute lambda-aware PME self-energy correction.
///
/// Standard PME self-energy:
///   E_self = -(α/√π) Σᵢ qᵢ²
///
/// Lambda-aware version for GCMC:
///   E_self = -(α/√π) Σᵢ (qᵢ·λᵢ)²
///
/// The self-energy correction removes the spurious interaction of each atom with its own smeared
/// charge distribution in reciprocal space. With lambda scaling:
/// - Only coupled atoms (λ > threshold) contribute
/// - Ghost atoms (λ ≈ 0) have no self-interaction to correct
/// - Scaling goes as λ² because energy depends on (q·λ)²
///
/// \param charges           Atomic charges [e]
/// \param lambda_ele        Electrostatic lambda values [0.0, 1.0]
/// \param ewald_coeff       Ewald coefficient α [Å⁻¹]
/// \param n_atoms           Number of atoms
/// \param lambda_threshold  Atoms with λ < threshold are skipped (default 0.01)
/// \return                  Self-energy correction [kcal/mol]
double pmeSelfEnergyLambda(const double* charges, const double* lambda_ele,
                           double ewald_coeff, int n_atoms,
                           double lambda_threshold = 0.01);

/// \brief Compute lambda-aware PME self-energy for a subset of atoms.
///
/// \param charges           Atomic charges for all atoms
/// \param lambda_ele        Lambda values for all atoms
/// \param ewald_coeff       Ewald coefficient
/// \param atom_start        Starting atom index (inclusive)
/// \param atom_end          Ending atom index (exclusive)
/// \param lambda_threshold  Threshold for including atoms
/// \return                  Self-energy for atoms in range [atom_start, atom_end)
double pmeSelfEnergyLambda(const double* charges, const double* lambda_ele,
                           double ewald_coeff, int atom_start, int atom_end,
                           double lambda_threshold = 0.01);

} // namespace energy
} // namespace stormm

#endif
