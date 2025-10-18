// -*-c++-*-
#include "copyright.h"
#include <cmath>

namespace stormm {
namespace energy {

//-------------------------------------------------------------------------------------------------
template <typename Tcoord, typename Tforce, typename Tcalc>
void evaluateLambdaNonbondedEnergy(const LambdaNonbondedKit<Tcalc> &lambda_nbk,
                                   const StaticExclusionMaskReader &ser,
                                   const Tcoord* xcrd, const Tcoord* ycrd, const Tcoord* zcrd,
                                   const double* umat, const double* invu,
                                   const UnitCellType unit_cell, Tforce* xfrc, Tforce* yfrc,
                                   Tforce* zfrc, ScoreCard *sc, const EvaluateForce eval_force,
                                   const EvaluateEnergy eval_energy, const int system_index,
                                   const Tcalc inv_gpos_factor, const Tcalc force_factor,
                                   const Tcalc clash_distance, const Tcalc clash_ratio,
                                   const Tcalc vdw_coupling_threshold, const Tcalc softcore_alpha) {

  // Energy scale for fixed-precision accumulation (from STORMM conventions)
  constexpr Tcalc energy_scale = 1.0e6;  // Energy in units of 1/millionth kcal/mol

  // Lambda threshold for considering an atom as a ghost
  constexpr Tcalc LAMBDA_THRESHOLD = 0.01;

  // Tile constants for exclusion checking
  constexpr int supertile_length = 256;
  constexpr int tile_length = 16;
  constexpr int tiles_per_supertile = supertile_length / tile_length;  // 16

  // Energy accumulators (fixed precision)
  llint elec_acc = 0LL;
  llint vdw_acc = 0LL;
  int clash_count = 0;

  // For vacuum simulations (typical for GCMC), no periodic boundaries
  const bool is_vacuum = (unit_cell == UnitCellType::NONE);

  // OPTIMIZATION: Build list of coupled atoms (lambda > threshold)
  // This allows us to skip ghost-ghost AND coupled-ghost interactions
  // Changes from O(N²) with early exits to O(N_coupled²)
  std::vector<int> coupled_indices;
  for (int atom = 0; atom < lambda_nbk.natom; atom++) {
    const Tcalc lambda_vdw_atom = lambda_nbk.lambda_vdw[atom];
    const Tcalc lambda_ele_atom = lambda_nbk.lambda_ele[atom];
    if (lambda_vdw_atom >= LAMBDA_THRESHOLD || lambda_ele_atom >= LAMBDA_THRESHOLD) {
      coupled_indices.push_back(atom);
    }
  }

  const int n_coupled = coupled_indices.size();

  // Loop over coupled atom pairs (i, j) with i < j
  for (int i_idx = 0; i_idx < n_coupled - 1; i_idx++) {
    const int i = coupled_indices[i_idx];

    // Get lambda values for atom i (known to be coupled)
    const Tcalc lambda_vdw_i = lambda_nbk.lambda_vdw[i];
    const Tcalc lambda_ele_i = lambda_nbk.lambda_ele[i];

    // Get coordinates for atom i
    const Tcalc xi = static_cast<Tcalc>(xcrd[i]) * inv_gpos_factor;
    const Tcalc yi = static_cast<Tcalc>(ycrd[i]) * inv_gpos_factor;
    const Tcalc zi = static_cast<Tcalc>(zcrd[i]) * inv_gpos_factor;

    // Get parameters for atom i
    const Tcalc qi = lambda_nbk.charge[i];
    const int lj_type_i = lambda_nbk.lj_idx[i];

    // Loop over coupled atoms j > i to avoid double counting
    for (int j_idx = i_idx + 1; j_idx < n_coupled; j_idx++) {
      const int j = coupled_indices[j_idx];

      // Get lambda values for atom j (known to be coupled)
      const Tcalc lambda_vdw_j = lambda_nbk.lambda_vdw[j];
      const Tcalc lambda_ele_j = lambda_nbk.lambda_ele[j];

      // Check exclusions using STORMM's tiled mask system
      // Compute supertile and tile indices for atoms i and j
      const int sti = i / supertile_length;
      const int stj = j / supertile_length;
      const int ti = (i % supertile_length) / tile_length;
      const int tj = (j % supertile_length) / tile_length;
      const int local_i = i % tile_length;
      const int local_j = j % tile_length;

      // Look up the supertile map index
      const int stij_map_index = ser.supertile_map_idx[(stj * ser.supertile_stride_count) + sti];
      if (stij_map_index >= 0) {
        // Look up the tile map index within this supertile pair
        const int tij_map_index = ser.tile_map_idx[stij_map_index + (tj * tiles_per_supertile) + ti];
        if (tij_map_index >= 0) {
          // Test the exclusion bit mask
          const uint mask_i = ser.mask_data[tij_map_index + local_i];
          if ((mask_i >> local_j) & 0x1) {
            continue;  // This pair is excluded (1-2, 1-3, or 1-4 interaction)
          }
        }
      }

      // Compute distance vector
      const Tcalc xj = static_cast<Tcalc>(xcrd[j]) * inv_gpos_factor;
      const Tcalc yj = static_cast<Tcalc>(ycrd[j]) * inv_gpos_factor;
      const Tcalc zj = static_cast<Tcalc>(zcrd[j]) * inv_gpos_factor;

      Tcalc dx = xj - xi;
      Tcalc dy = yj - yi;
      Tcalc dz = zj - zi;

      // Apply minimum image convention if periodic (not typical for GCMC)
      if (!is_vacuum) {
        // Simplified PBC for cubic box (full PBC would use umat/invu)
        // This is placeholder - GCMC typically uses vacuum
      }

      const Tcalc r2 = dx*dx + dy*dy + dz*dz;
      const Tcalc r = sqrt(r2);

      // Check for clashes if requested
      if (clash_distance > 0.0 && r < clash_distance) {
        clash_count++;
      }

      const Tcalc invr = 1.0 / r;

      // =================================================================
      // ELECTROSTATIC CONTRIBUTION (lambda-scaled)
      // =================================================================
      const Tcalc lambda_ele_ij = lambda_ele_i * lambda_ele_j;

      if (fabs(qi * lambda_nbk.charge[j] * lambda_ele_ij) > 1.0e-10) {
        // Scale charges by lambda
        const Tcalc qi_scaled = qi * lambda_ele_i;
        const Tcalc qj_scaled = lambda_nbk.charge[j] * lambda_ele_j;
        const Tcalc qiqj = qi_scaled * qj_scaled;

        // Electrostatic energy: U = k * q_i * q_j / r
        const Tcalc elec_energy = lambda_nbk.coulomb_constant * qiqj * invr;

        if (eval_energy == EvaluateEnergy::YES) {
          elec_acc += std::llround(elec_energy * energy_scale);
        }

        if (eval_force == EvaluateForce::YES) {
          // Force magnitude: F = k * q_i * q_j / r^2
          // Direction: along (xj-xi, yj-yi, zj-zi)
          const Tcalc fmag = lambda_nbk.coulomb_constant * qiqj * invr * invr;
          const Tcalc fx = fmag * dx * invr * force_factor;
          const Tcalc fy = fmag * dy * invr * force_factor;
          const Tcalc fz = fmag * dz * invr * force_factor;

          // Newton's third law: F_i = -F_ij, F_j = +F_ij
          xfrc[i] -= static_cast<Tforce>(fx);
          yfrc[i] -= static_cast<Tforce>(fy);
          zfrc[i] -= static_cast<Tforce>(fz);
          xfrc[j] += static_cast<Tforce>(fx);
          yfrc[j] += static_cast<Tforce>(fy);
          zfrc[j] += static_cast<Tforce>(fz);
        }
      }

      // =================================================================
      // VDW CONTRIBUTION (lambda-scaled with softcore)
      // =================================================================
      const Tcalc lambda_vdw_ij = lambda_vdw_i * lambda_vdw_j;

      if (lambda_vdw_ij > LAMBDA_THRESHOLD) {
        // Get LJ parameters from type indices
        const int lj_type_j = lambda_nbk.lj_idx[j];
        const int lj_pair_index = lj_type_i * lambda_nbk.n_lj_types + lj_type_j;

        const Tcalc lja = lambda_nbk.lja_coeff[lj_pair_index];
        const Tcalc ljb = lambda_nbk.ljb_coeff[lj_pair_index];
        const Tcalc sigma = lambda_nbk.lj_sigma[lj_pair_index];

        // Softcore potential to prevent singularities:
        // r_eff^6 = r^6 + alpha * sigma^6 * (1 - lambda)
        // This ensures r_eff stays finite even as r -> 0 when lambda < 1
        const Tcalc sigma2 = sigma * sigma;
        const Tcalc sigma6 = sigma2 * sigma2 * sigma2;
        const Tcalc r6 = r2 * r2 * r2;
        const Tcalc softcore_offset = softcore_alpha * sigma6 * (1.0 - lambda_vdw_ij);
        const Tcalc r_eff6 = r6 + softcore_offset;
        const Tcalc invr_eff6 = 1.0 / r_eff6;
        const Tcalc invr_eff12 = invr_eff6 * invr_eff6;

        // VDW energy: U = lambda * (A/r_eff^12 - B/r_eff^6)
        const Tcalc vdw_energy = lambda_vdw_ij * (lja * invr_eff12 - ljb * invr_eff6);

        if (eval_energy == EvaluateEnergy::YES) {
          vdw_acc += std::llround(vdw_energy * energy_scale);
        }

        if (eval_force == EvaluateForce::YES) {
          // Force: F = -dU/dr
          // dU/dr = lambda * d/dr[A/r_eff^12 - B/r_eff^6]
          //
          // For softcore potential:
          // d(r_eff^6)/dr = 6 * r^5
          // dU/dr = lambda * [-12A/r_eff^13 + 6B/r_eff^7] * dr_eff/dr
          //       = lambda * [-12A/r_eff^13 + 6B/r_eff^7] * 6r^5 / (2 * r_eff^6)
          //       = lambda * 6r^5 * [-12A/(2*r_eff^19) + 6B/(2*r_eff^13)]
          //
          // Simplified for efficient computation:
          const Tcalc r5 = r2 * r2 * r;
          const Tcalc invr_eff13 = invr_eff12 * invr_eff6; // 1/r_eff^18 actually
          const Tcalc invr_eff7 = invr_eff6 * invr_eff6;   // 1/r_eff^12 actually

          // More careful calculation matching r_eff^6 definition:
          // Let y = r_eff^6 = r^6 + offset
          // dy/dr = 6r^5
          // d(1/y)/dr = -1/y^2 * dy/dr = -6r^5/y^2
          // d(1/y^2)/dr = -2/y^3 * dy/dr = -12r^5/y^3
          //
          // dU/dr = lambda * [A * d(1/y^2)/dr - B * d(1/y)/dr]
          //       = lambda * [-12*A*r^5/y^3 + 6*B*r^5/y^2]
          //       = lambda * 6*r^5 * [2*A*invr_eff18 - B*invr_eff12]

          const Tcalc fmag = lambda_vdw_ij * 6.0 * r5 *
                             (2.0 * lja / (r_eff6 * r_eff6 * r_eff6) - ljb / (r_eff6 * r_eff6));

          const Tcalc fx = fmag * dx * invr * force_factor;
          const Tcalc fy = fmag * dy * invr * force_factor;
          const Tcalc fz = fmag * dz * invr * force_factor;

          xfrc[i] -= static_cast<Tforce>(fx);
          yfrc[i] -= static_cast<Tforce>(fy);
          zfrc[i] -= static_cast<Tforce>(fz);
          xfrc[j] += static_cast<Tforce>(fx);
          yfrc[j] += static_cast<Tforce>(fy);
          zfrc[j] += static_cast<Tforce>(fz);
        }
      }
    }
  }

  // Commit energies to score card
  if (eval_energy == EvaluateEnergy::YES) {
    sc->contribute(StateVariable::ELECTROSTATIC, elec_acc, system_index);
    sc->contribute(StateVariable::VDW, vdw_acc, system_index);
    // TODO: Clash detection not currently supported in StateVariable enum
    // if (clash_count > 0 && clash_distance > 0.0) {
    //   sc->contribute(StateVariable::CLASH, clash_count, system_index);
    // }
  }
}

} // namespace energy
} // namespace stormm
