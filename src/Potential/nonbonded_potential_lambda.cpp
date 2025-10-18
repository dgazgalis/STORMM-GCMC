#include "copyright.h"
#include <cmath>
#include <algorithm>
#include "Constants/symbol_values.h"
#include "DataTypes/common_types.h"
#include "Math/vector_ops.h"
#include "Reporting/error_format.h"
#include "nonbonded_potential.h"
#include "soft_core_potentials.h"

namespace stormm {
namespace energy {

using errors::rtErr;

/// \brief Threshold for determining if an atom is fully active (lambda ≈ 1.0)
/// Atoms with lambda < this threshold are considered ghosts and require special treatment.
/// Using 0.999 rather than strict 1.0 to handle floating-point imprecision.
constexpr double LAMBDA_ACTIVE_THRESHOLD = 0.999;

/// \brief Minimum lambda threshold for ghost atoms
/// Atoms with lambda below this threshold are completely decoupled (no interactions).
/// Only atoms with LAMBDA_GHOST_THRESHOLD < lambda < LAMBDA_ACTIVE_THRESHOLD need evaluation.
/// This excludes lambda=0 ghost molecules in GCMC, which contribute zero energy.
constexpr double LAMBDA_GHOST_THRESHOLD = 0.01;

//-------------------------------------------------------------------------------------------------
double2 evaluateLambdaScaledNonbonded(
    const NonbondedKit<double>& nbk,
    const StaticExclusionMaskReader& ser,
    PhaseSpaceWriter& psw,
    const std::vector<double>& lambda_vdw,
    const std::vector<double>& lambda_ele,
    const std::vector<double>& atom_sigma,
    const std::vector<double>& atom_epsilon,
    ScoreCard* ecard,
    EvaluateForce eval_force,
    int system_index)
{
  const int n_atoms = nbk.natom;
  double elec_energy = 0.0;
  double vdw_energy = 0.0;

  // Validate lambda and LJ parameter array sizes
  if (static_cast<int>(lambda_vdw.size()) != n_atoms ||
      static_cast<int>(lambda_ele.size()) != n_atoms ||
      static_cast<int>(atom_sigma.size()) != n_atoms ||
      static_cast<int>(atom_epsilon.size()) != n_atoms) {
    rtErr("Lambda and LJ parameter arrays must have size equal to atom count",
          "evaluateLambdaScaledNonbonded");
  }

  // Get coordinates
  const double* xcrd = psw.xcrd;
  const double* ycrd = psw.ycrd;
  const double* zcrd = psw.zcrd;

  // Get forces (if evaluating)
  double* xfrc = (eval_force == EvaluateForce::YES) ? psw.xfrc : nullptr;
  double* yfrc = (eval_force == EvaluateForce::YES) ? psw.yfrc : nullptr;
  double* zfrc = (eval_force == EvaluateForce::YES) ? psw.zfrc : nullptr;

  // Get box dimensions and transformation matrices
  const double* umat = psw.umat;
  const double* invu = psw.invu;
  const UnitCellType unit_cell = psw.unit_cell;

  // Coulomb constant (convert from internal units if needed)
  const double coulomb_const = nbk.coulomb_constant;

  // Define softcore parameters
  const double softcore_alpha = 0.5;  // Standard softcore alpha

  // Build list of coupled atom indices (atoms with lambda > LAMBDA_GHOST_THRESHOLD)
  // This excludes fully decoupled atoms (lambda=0) which contribute zero energy.
  // For GCMC: includes both partially coupled ghosts AND fully active molecules.
  // This is critical for GCMC insertions to see repulsion from existing active molecules.
  std::vector<int> ghost_atom_indices;
  for (int i = 0; i < n_atoms; i++) {
    const double lambda_vdw_i = lambda_vdw[i];
    const double lambda_ele_i = lambda_ele[i];

    // Include atom if lambda > LAMBDA_GHOST_THRESHOLD (not fully decoupled)
    // CRITICAL FIX: Remove upper bound check to include active molecules at lambda=1.0
    const bool is_coupled =
        (lambda_vdw_i > LAMBDA_GHOST_THRESHOLD) ||
        (lambda_ele_i > LAMBDA_GHOST_THRESHOLD);

    if (is_coupled) {
      ghost_atom_indices.push_back(i);
    }
  }

  // DISABLED OPTIMIZATION: Previously returned 0.0 if no ghosts, but this breaks GCMC
  // where we need to calculate interaction energy even when the molecule is fully coupled.
  // If there are no ghost atoms, we should still calculate the full nonbonded energy.
  // However, for performance, if there are no ghosts we can return 0.0 ONLY if we know
  // we're in a mode where all atoms are meant to be fully interacting.
  // For now, comment out the early return to fix GCMC:
  // if (ghost_atom_indices.empty()) {
  //   return double2{0.0, 0.0};
  // }

  // DEBUG: Print ghost atom count for first few evaluations
  static int eval_count = 0;
  if (false && eval_count < 5) {  // Disabled DEBUG output
    std::cout << "DEBUG Energy eval " << eval_count << ": ghost_atom_indices.size() = "
              << ghost_atom_indices.size() << "\n";
    eval_count++;
  }

  // Optimized loop: only evaluate pairs involving ghost atoms
  // This reduces complexity from O(N²) to O(N_ghost * N) where N_ghost << N
  for (int ghost_idx : ghost_atom_indices) {
    for (int j = 0; j < n_atoms; j++) {
      if (j == ghost_idx) continue;

      // For coupled-coupled pairs, avoid double counting
      // Check if j is also coupled (not fully decoupled)
      const bool j_is_coupled =
          (lambda_vdw[j] > LAMBDA_GHOST_THRESHOLD) ||
          (lambda_ele[j] > LAMBDA_GHOST_THRESHOLD);

      if (j_is_coupled) {
        // Only process if ghost_idx < j to avoid double counting
        if (ghost_idx > j) continue;
      }

      // Determine i and j indices for this pair
      int i = ghost_idx;
      // Note: j is already set

      // Check exclusion mask using the tile-based approach
      // Compute which supertile and tile this pair falls into
      const int supertile_length = 256;
      const int tile_length = 16;

      const int sti = i / supertile_length;
      const int stj = j / supertile_length;
      const int ti = (i % supertile_length) / tile_length;
      const int tj = (j % supertile_length) / tile_length;
      const int local_i = i % tile_length;
      const int local_j = j % tile_length;

      // Get the supertile map index
      const int stij_map_index = ser.supertile_map_idx[(stj * ser.supertile_stride_count) + sti];

      // Get the tile map index within the supertile
      const int tij_map_index = ser.tile_map_idx[stij_map_index + (tj * 16) + ti];

      // Get the exclusion mask for atom i
      const uint mask_i = ser.mask_data[tij_map_index + local_i];

      // Test if atom j is excluded from atom i
      if ((mask_i >> local_j) & 0x1) {
        continue;  // This pair is excluded
      }

      // Calculate distance with PBC
      double dx = xcrd[j] - xcrd[i];
      double dy = ycrd[j] - ycrd[i];
      double dz = zcrd[j] - zcrd[i];

      // Apply minimum image convention based on unit cell type
      if (unit_cell != UnitCellType::NONE) {
        // For orthorhombic box (simplified)
        if (unit_cell == UnitCellType::ORTHORHOMBIC) {
          const double box_x = umat[0];
          const double box_y = umat[4];
          const double box_z = umat[8];

          dx -= std::round(dx / box_x) * box_x;
          dy -= std::round(dy / box_y) * box_y;
          dz -= std::round(dz / box_z) * box_z;
        }
        // For triclinic, would need full transformation using umat/invu
      }

      const double r2 = dx*dx + dy*dy + dz*dz;
      const double r = std::sqrt(r2);

      // Get scaled charges
      const double qi = nbk.charge[i] * lambda_ele[i];
      const double qj = nbk.charge[j] * lambda_ele[j];
      const double qiqj = qi * qj;

      // Get LJ parameters from pre-computed arrays
      const double sigma_i = atom_sigma[i];
      const double sigma_j = atom_sigma[j];
      const double epsilon_i = atom_epsilon[i];
      const double epsilon_j = atom_epsilon[j];

      // Combine LJ parameters using Lorentz-Berthelot rules
      // Arithmetic mean for sigma, geometric mean for epsilon
      const double sigma = 0.5 * (sigma_i + sigma_j);
      const double epsilon = std::sqrt(epsilon_i * epsilon_j);

      // Combined VDW lambda
      const double lambda_ij_vdw = lambda_vdw[i] * lambda_vdw[j];

      // Electrostatic energy (simple Coulomb)
      double elec_contrib = 0.0;
      double elec_force_mag = 0.0;

      if (std::abs(qiqj) > 1.0e-10) {
        elec_contrib = coulomb_const * qiqj / r;
        elec_energy += elec_contrib;

        if (eval_force == EvaluateForce::YES) {
          elec_force_mag = -coulomb_const * qiqj / (r * r2);
        }
      }

      // VDW energy with softcore
      double vdw_contrib = 0.0;
      double vdw_force_mag = 0.0;

      // DEBUG: Print first few VDW calculations
      static int vdw_calc_count = 0;
      if (false && vdw_calc_count < 3 && lambda_ij_vdw > 1.0e-10 && r < 10.0) {  // Disabled DEBUG output
        std::cout << "DEBUG VDW calc " << vdw_calc_count << ": i=" << i << " j=" << j
                  << " r=" << r << " eps_i=" << epsilon_i << " eps_j=" << epsilon_j
                  << " eps_combined=" << epsilon << " lambda_ij=" << lambda_ij_vdw << "\n";
        vdw_calc_count++;
      }

      if (epsilon > 1.0e-10 && lambda_ij_vdw > 1.0e-10) {
        // Softcore potential to avoid singularities during insertion/deletion
        // Using Beutler et al. 1994 softcore form: U_sc = lambda * U(r_eff)
        // where r_eff^6 = r^6 + alpha * sigma^6 * (1-lambda)

        // Compute softcore effective distance
        const double alpha = 0.5;  // Standard softcore alpha parameter
        const double one_minus_lambda = 1.0 - lambda_ij_vdw;

        // r^6 + alpha * sigma^6 * (1 - lambda)
        const double r6 = r2 * r2 * r2;
        const double sigma2 = sigma * sigma;
        const double sigma6 = sigma2 * sigma2 * sigma2;
        const double r_eff6 = r6 + alpha * sigma6 * one_minus_lambda;

        // Compute LJ energy with softcore modification
        const double inv_r_eff6 = 1.0 / r_eff6;
        const double inv_r_eff12 = inv_r_eff6 * inv_r_eff6;

        // U_LJ = 4ε[(σ/r_eff)^12 - (σ/r_eff)^6]
        // With r_eff^6 pre-computed: U_LJ = 4ε[σ^12/r_eff^12 - σ^6/r_eff^6]
        const double sigma12 = sigma6 * sigma6;
        vdw_contrib = lambda_ij_vdw * 4.0 * epsilon * (sigma12 * inv_r_eff12 - sigma6 * inv_r_eff6);
        vdw_energy += vdw_contrib;

        if (eval_force == EvaluateForce::YES && r > 1.0e-10) {
          // Force from softcore LJ (derivative with respect to r)
          // Softcore LJ potential: U = 4ε[(σ^6/r_eff^6)^2 - (σ^6/r_eff^6)]
          // where r_eff^6 = r^6 + α(1-λ)σ^6
          //
          // Apply chain rule: dU/dr = dU/dr_eff^6 * dr_eff^6/dr
          // dU/dr_eff^6 = 4ε[-12σ^12/r_eff^18 + 6σ^6/r_eff^12]
          // dr_eff^6/dr = d(r^6)/dr = 6r^5

          const double inv_r_eff12 = inv_r_eff6 * inv_r_eff6;
          const double inv_r_eff18 = inv_r_eff12 * inv_r_eff6;

          // dU/dr_eff^6
          const double dU_dr_eff6 = 4.0 * epsilon *
              (-12.0 * sigma12 * inv_r_eff18 + 6.0 * sigma6 * inv_r_eff12);

          // dr_eff^6/dr = 6r^5
          const double r4 = r2 * r2;
          const double dr_eff6_dr = 6.0 * r4 * r;

          // Force magnitude: -dU/dr = -dU/dr_eff^6 * dr_eff^6/dr
          // Divide by r to get force per unit vector
          vdw_force_mag = -lambda_ij_vdw * dU_dr_eff6 * dr_eff6_dr / r;
        }
      }

      // Accumulate forces
      if (eval_force == EvaluateForce::YES && r > 1.0e-10) {
        const double total_force_mag = elec_force_mag + vdw_force_mag;

        // Force components
        const double fx = total_force_mag * dx;
        const double fy = total_force_mag * dy;
        const double fz = total_force_mag * dz;

        // Newton's third law
        xfrc[i] -= fx;
        yfrc[i] -= fy;
        zfrc[i] -= fz;

        xfrc[j] += fx;
        yfrc[j] += fy;
        zfrc[j] += fz;
      }
    }
  }

  // Accumulate in scorecard
  if (ecard != nullptr) {
    ecard->contribute(StateVariable::ELECTROSTATIC, elec_energy, system_index);
    ecard->contribute(StateVariable::VDW, vdw_energy, system_index);
  }

  return {elec_energy, vdw_energy};
}

} // namespace energy
} // namespace stormm