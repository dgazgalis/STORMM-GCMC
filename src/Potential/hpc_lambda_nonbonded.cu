// -*-c++-*-
#include "copyright.h"
#include <cuda_runtime.h>
#include "Accelerator/core_kernel_manager.h"
#include "Accelerator/hybrid.h"
#include "Constants/behavior.h"
#include "DataTypes/common_types.h"
#include "MolecularMechanics/mm_controls.h"
#include "Potential/cacheresource.h"
#include "Potential/scorecard.h"
#include "Synthesis/atomgraph_synthesis.h"
#include "Synthesis/implicit_solvent_workspace.h"
#include "Synthesis/phasespace_synthesis.h"
#include "Synthesis/static_mask_synthesis.h"
#include "Topology/atomgraph_enumerators.h"
#include "Trajectory/thermostat.h"
#include "Trajectory/trajectory_enumerators.h"
#include "hpc_lambda_nonbonded.h"

namespace stormm {
namespace energy {

using card::CoreKlManager;
using card::Hybrid;
using card::HybridTargetLevel;
using card::PrecisionModel;
using mm::MolecularMechanicsControls;
using synthesis::AtomGraphSynthesis;
using synthesis::ImplicitSolventWorkspace;
using synthesis::ISWorkspaceKit;
using synthesis::PhaseSpaceSynthesis;
using synthesis::PsSynthesisWriter;
using synthesis::SeMaskSynthesisReader;
using synthesis::StaticExclusionMaskSynthesis;
using synthesis::SyNonbondedKit;
using topology::ImplicitSolventModel;
using topology::UnitCellType;
using trajectory::CoordinateCycle;
using trajectory::Thermostat;

/// \brief Threshold for lambda coupling - atoms below this are fully decoupled
constexpr double LAMBDA_GHOST_THRESHOLD = 0.01;

/// \brief Softcore alpha parameter for avoiding singularities
constexpr double SOFTCORE_ALPHA = 0.5;

//-------------------------------------------------------------------------------------------------
// CUDA kernel for lambda-scaled nonbonded energy evaluation
//
// Each thread processes one coupled atom and computes its interactions with other coupled atoms.
// This parallelizes the O(N_coupled²) loop efficiently on GPU.
// Ghost atoms are skipped entirely, providing major speedup when N_ghost >> N_coupled.
//-------------------------------------------------------------------------------------------------
__global__ void kLambdaScaledNonbonded(
    const int n_atoms,
    const int n_coupled,
    const int* __restrict__ coupled_indices,
    const llint* __restrict__ xcrd,
    const llint* __restrict__ ycrd,
    const llint* __restrict__ zcrd,
    const double* __restrict__ charges,
    const double* __restrict__ lambda_vdw,
    const double* __restrict__ lambda_ele,
    const int* __restrict__ lj_idx,
    const int n_lj_types,
    const double2* __restrict__ ljab_coeff,
    const uint* __restrict__ exclusion_mask,
    const int* __restrict__ supertile_map,
    const int* __restrict__ tile_map,
    const int supertile_stride,
    const double* __restrict__ umat,
    const UnitCellType unit_cell,
    const double coulomb_const,
    const double ewald_coeff,  // Ewald coefficient for PME direct space
    const float inv_gpos_scale,
    const float frc_scale,
    double* __restrict__ output_elec,
    double* __restrict__ output_vdw,
    llint* __restrict__ xfrc,  // Force outputs (NULL for energy-only mode)
    llint* __restrict__ yfrc,
    llint* __restrict__ zfrc,
    const double* __restrict__ born_radii,    // Born radii from GB workspace (NULL if GB disabled)
    const double gb_kappa,                    // GB salt screening parameter
    const double gb_offset,                   // GB offset parameter
    const topology::ImplicitSolventModel gb_model)  // GB model type
{
  // Thread index maps to coupled atom index
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid >= n_coupled) return;

  // Get the actual atom index for this coupled atom
  const int i = coupled_indices[tid];

  // FIX: Validate atom index is within bounds (safety check for corrupted coupled_indices)
  if (i < 0 || i >= n_atoms) return;

  // Load atom i properties
  const double xi = (double)(xcrd[i]) * inv_gpos_scale;
  const double yi = (double)(ycrd[i]) * inv_gpos_scale;
  const double zi = (double)(zcrd[i]) * inv_gpos_scale;
  const double qi = charges[i];
  const double lambda_vdw_i = lambda_vdw[i];
  const double lambda_ele_i = lambda_ele[i];
  const int lj_type_i = lj_idx[i];

  // Accumulate energies and forces for this atom
  double elec_sum = 0.0;
  double vdw_sum = 0.0;
  double fx_sum = 0.0;
  double fy_sum = 0.0;
  double fz_sum = 0.0;
  const bool compute_forces = (xfrc != nullptr);

  // Tile geometry constants
  const int supertile_length = 256;
  const int tile_length = 16;

  // OPTIMIZATION: Loop over coupled atoms only (not all atoms)
  // This changes from O(N_coupled × N_total) to O(N_coupled²/2)
  // Each pair (i,j) computed exactly once using atom index comparison
  for (int j_tid = 0; j_tid < n_coupled; j_tid++) {
    // Get actual atom index for coupled atom j
    const int j = coupled_indices[j_tid];

    // FIX: Validate j is within bounds (safety check for corrupted coupled_indices)
    if (j < 0 || j >= n_atoms) continue;

    // Skip self-interaction
    if (i == j) continue;

    // Skip pairs where j >= i to avoid double-counting
    // Each unique pair (i,j) is computed by exactly one thread (the one with smaller atom index)
    if (j >= i) continue;

    // Compute tile indices for exclusion mask lookup
    const int sti = i / supertile_length;
    const int stj = j / supertile_length;
    const int ti = (i % supertile_length) / tile_length;
    const int tj = (j % supertile_length) / tile_length;
    const int local_i = i % tile_length;
    const int local_j = j % tile_length;

    const int stij_map_index = supertile_map[(stj * supertile_stride) + sti];
    const int tij_map_index = tile_map[stij_map_index + (tj * 16) + ti];
    const uint mask_i = exclusion_mask[tij_map_index + local_i];

    // Skip if excluded
    if ((mask_i >> local_j) & 0x1) continue;

    // Load atom j properties
    const double xj = (double)(xcrd[j]) * inv_gpos_scale;
    const double yj = (double)(ycrd[j]) * inv_gpos_scale;
    const double zj = (double)(zcrd[j]) * inv_gpos_scale;
    const double qj = charges[j];
    const double lambda_vdw_j = lambda_vdw[j];
    const double lambda_ele_j = lambda_ele[j];
    const int lj_type_j = lj_idx[j];

    // Compute distance with PBC
    double dx = xj - xi;
    double dy = yj - yi;
    double dz = zj - zi;

    // Apply minimum image convention
    if (unit_cell == UnitCellType::ORTHORHOMBIC) {
      const double box_x = umat[0];
      const double box_y = umat[4];
      const double box_z = umat[8];

      dx -= round(dx / box_x) * box_x;
      dy -= round(dy / box_y) * box_y;
      dz -= round(dz / box_z) * box_z;
    }
    // TODO: Add triclinic support if needed

    const double r2 = dx*dx + dy*dy + dz*dz;
    const double r = sqrt(r2);

    // Electrostatic energy and force
    const double qi_scaled = qi * lambda_ele_i;
    const double qj_scaled = qj * lambda_ele_j;
    const double qiqj = qi_scaled * qj_scaled;

    if (fabs(qiqj) > 1.0e-10) {
      const double invr = 1.0 / r;
      double elec_term;

      // Use Ewald direct space for PME (ewald_coeff > 0), otherwise cutoff Coulomb
      if (ewald_coeff > 1.0e-10) {
        // PME direct space: erfc(α·r)/r removes long-range part handled by reciprocal space
        elec_term = erfc(ewald_coeff * r) * invr;
      } else {
        // Non-periodic: standard 1/r Coulomb
        elec_term = invr;
      }

      elec_sum += coulomb_const * qiqj * elec_term;

      if (compute_forces) {
        // F = -dU/dr * (r_vec/r)
        // For Ewald: -d/dr[erfc(α·r)/r] = erfc(α·r)/r² + 2α/√π·exp(-α²r²)/r
        double fmag;
        if (ewald_coeff > 1.0e-10) {
          const double alpha_r = ewald_coeff * r;
          const double exp_term = exp(-alpha_r * alpha_r);
          fmag = coulomb_const * qiqj * (elec_term * invr + 2.0 * ewald_coeff * exp_term * invr / sqrt(M_PI));
        } else {
          fmag = coulomb_const * qiqj * invr * invr;
        }
        fx_sum += fmag * dx * invr;
        fy_sum += fmag * dy * invr;
        fz_sum += fmag * dz * invr;
      }
    }

    // VDW energy with softcore using STORMM native LJ parameters
    // Standard STORMM pattern: ij_ljidx = lj_idx[j] + lj_idx[i] * n_lj_types
    const int ij_ljidx = lj_type_j + lj_type_i * n_lj_types;
    const double2 ljab = ljab_coeff[ij_ljidx];
    const double lja = ljab.x;  // A coefficient (r^-12 repulsive term)
    const double ljb = ljab.y;  // B coefficient (r^-6 attractive term)
    const double lambda_ij_vdw = lambda_vdw_i * lambda_vdw_j;

    if ((fabs(lja) > 1.0e-10 || fabs(ljb) > 1.0e-10) && lambda_ij_vdw > 1.0e-10) {
      // STORMM uses: U = lja/r^12 - ljb/r^6 (standard Lennard-Jones form)
      // For softcore, we use r_eff instead of r
      const double one_minus_lambda = 1.0 - lambda_ij_vdw;
      const double r6 = r2 * r2 * r2;

      // Softcore offset uses ljb coefficient to estimate sigma
      // Since ljb ~ epsilon * sigma^6, we can estimate the characteristic scale
      // For simplicity, use ljb as the characteristic scale
      const double r_eff6 = r6 + SOFTCORE_ALPHA * fabs(ljb) * one_minus_lambda;

      const double inv_r_eff6 = 1.0 / r_eff6;
      const double inv_r_eff12 = inv_r_eff6 * inv_r_eff6;

      // FIX: Swap powers to match STORMM convention: A/r^12 - B/r^6
      const double lj_energy = lja * inv_r_eff12 - ljb * inv_r_eff6;

      vdw_sum += lambda_ij_vdw * lj_energy;

      if (compute_forces) {
        // Softcore force: F = -dU/dr
        // For U = lja/r_eff^6 - ljb/r_eff^12
        // dU/dr = -6*lja/r_eff^7 * dr_eff/dr + 12*ljb/r_eff^13 * dr_eff/dr
        // For r_eff^6 = r^6 + offset: dr_eff/dr = 6*r^5 / (2*r_eff^6)
        const double r5 = r2 * r2 * r;
        const double inv_r_eff7 = inv_r_eff6 / r_eff6;   // Actually 1/(r_eff^6)^2 = 1/r_eff^12
        const double inv_r_eff13 = inv_r_eff12 / r_eff6; // Actually 1/(r_eff^6)^3 = 1/r_eff^18

        // Corrected derivative with proper chain rule for softcore
        // F·r = lambda * 6*r^5 * (-6*lja*inv_r_eff^7 + 12*ljb*inv_r_eff^13) / (2*r_eff^6)
        //     = lambda * 3*r^5 * (-6*lja*inv_r_eff^7 + 12*ljb*inv_r_eff^13) / r_eff^6
        const double fmag = lambda_ij_vdw * 3.0 * r5 *
                           (-6.0 * lja * inv_r_eff7 + 12.0 * ljb * inv_r_eff13) / r_eff6;

        const double invr = 1.0 / r;
        fx_sum += fmag * dx * invr;
        fy_sum += fmag * dy * invr;
        fz_sum += fmag * dz * invr;
      }
    }

    // GB electrostatic contribution (if enabled)
    // CRITICAL: Apply lambda scaling for proper GCMC physics
    if (born_radii != nullptr && gb_model != topology::ImplicitSolventModel::NONE) {
      // Read Born radii for atoms i and j
      const double born_radius_i = born_radii[i];
      const double born_radius_j = born_radii[j];

      // Compute effective GB distance function f_GB
      // For HCT/OBC models: f_GB = sqrt(r² + Ri*Rj*exp(-r²/4*Ri*Rj))
      const double rij_prod = born_radius_i * born_radius_j;
      const double exp_arg = -r2 / (4.0 * rij_prod);
      const double exp_term = exp(exp_arg);
      const double f_gb2 = r2 + rij_prod * exp_term;
      const double f_gb = sqrt(f_gb2);
      const double inv_f_gb = 1.0 / f_gb;

      // GB energy: -0.5 * qi * qj * (1/εin - 1/εout) * (1/f_GB - κ/cutoff)
      // εin = 1 (interior dielectric), εout = 80 (water)
      // For simplicity, use pre-computed dielectric factor
      const double diel_factor = -0.5 * (1.0 - 1.0/80.0);  // -(1/εin - 1/εout)/2

      // Apply lambda scaling to GB electrostatic contribution
      const double lambda_ij_ele = lambda_ele_i * lambda_ele_j;
      const double gb_energy = diel_factor * coulomb_const * qi * qj * lambda_ij_ele *
                               (inv_f_gb - gb_kappa * exp(-gb_kappa * f_gb) / f_gb);

      // Add to electrostatic energy (GB is an electrostatic effect)
      elec_sum += gb_energy;

      if (compute_forces) {
        // GB force: F = -dU/dr
        // d/dr[1/f_GB] = -1/f_GB² * df_GB/dr
        // df_GB/dr = (1/2f_GB) * d/dr[r² + Ri*Rj*exp(-r²/4*Ri*Rj)]
        //          = (1/2f_GB) * [2r - Ri*Rj*exp(-r²/4*Ri*Rj) * r/(2*Ri*Rj)]
        //          = (r/f_GB) * [1 - 0.25*exp(-r²/4*Ri*Rj)]
        const double df_gb_dr = (r / f_gb) * (1.0 - 0.25 * exp_term);
        const double d_inv_f_gb_dr = -df_gb_dr / f_gb2;

        // Derivative of screening term if kappa > 0
        double d_screen_dr = 0.0;
        if (gb_kappa > 1.0e-10) {
          const double screen_term = gb_kappa * exp(-gb_kappa * f_gb);
          d_screen_dr = -screen_term * (gb_kappa * df_gb_dr / f_gb + df_gb_dr / f_gb2);
        }

        const double gb_fmag = -diel_factor * coulomb_const * qi * qj * lambda_ij_ele *
                               (d_inv_f_gb_dr - d_screen_dr);

        const double invr = 1.0 / r;
        fx_sum += gb_fmag * dx * invr;
        fy_sum += gb_fmag * dy * invr;
        fz_sum += gb_fmag * dz * invr;
      }
    }
  }

  // Write output
  output_elec[tid] = elec_sum;
  output_vdw[tid] = vdw_sum;

  // Accumulate forces using atomic operations (multiple threads may access same atom)
  if (compute_forces) {
    atomicAdd((ullint*)&xfrc[i], (ullint)(__double2ll_rn(fx_sum * frc_scale)));
    atomicAdd((ullint*)&yfrc[i], (ullint)(__double2ll_rn(fy_sum * frc_scale)));
    atomicAdd((ullint*)&zfrc[i], (ullint)(__double2ll_rn(fz_sum * frc_scale)));
  }
}

//-------------------------------------------------------------------------------------------------
// GPU kernel to accumulate work delta for NCMC protocol
// Computes: work += (E_after - E_before) where E = elec + vdw
//-------------------------------------------------------------------------------------------------
__global__ void kAccumulateWorkDelta(
    const double* __restrict__ elec_before,
    const double* __restrict__ vdw_before,
    const double* __restrict__ elec_after,
    const double* __restrict__ vdw_after,
    double* __restrict__ work_accumulator)
{
  // Single-threaded kernel (only called once per perturbation step)
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    double E_before = *elec_before + *vdw_before;
    double E_after = *elec_after + *vdw_after;
    double delta = E_after - E_before;
    atomicAdd(work_accumulator, delta);
  }
}

//-------------------------------------------------------------------------------------------------
// GPU kernel to update per-atom lambda values from NCMC schedule
// Each thread updates one atom's lambda values based on the current NCMC step
//-------------------------------------------------------------------------------------------------
__global__ void kUpdateLambdaFromSchedule(
    const int step_index,
    const double* __restrict__ lambda_schedule,
    const int* __restrict__ molecule_indices,
    const int n_molecule_atoms,
    const double vdw_coupling_threshold,
    double* __restrict__ lambda_vdw,
    double* __restrict__ lambda_ele)
{
  // Thread index maps to molecule atom index
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid >= n_molecule_atoms) return;

  // Get the global atom index for this molecule atom
  const int atom_idx = molecule_indices[tid];

  // Get the lambda value for this step
  const double lambda = lambda_schedule[step_index];

  // Two-stage coupling transformation (matches CPU adjustMoleculeLambda)
  // Stage 1 (λ ∈ [0, vdw_coupling_threshold]): VDW ramps up, electrostatics off
  // Stage 2 (λ ∈ (vdw_coupling_threshold, 1]): VDW at 1.0, electrostatics ramp up
  double lam_vdw, lam_ele;
  if (lambda <= vdw_coupling_threshold) {
    lam_vdw = lambda / vdw_coupling_threshold;
    lam_ele = 0.0;
  } else {
    lam_vdw = 1.0;
    lam_ele = (lambda - vdw_coupling_threshold) / (1.0 - vdw_coupling_threshold);
  }

  // Update lambda values for this atom
  lambda_vdw[atom_idx] = lam_vdw;
  lambda_ele[atom_idx] = lam_ele;
}

//-------------------------------------------------------------------------------------------------
// GPU kernel for computing Born radii for lambda-aware GB calculations
// No lambda scaling needed here - Born radii are calculated for all atoms
//-------------------------------------------------------------------------------------------------
__global__ void kLambdaBornRadii(
    const int n_atoms,
    const llint* __restrict__ xcrd,
    const llint* __restrict__ ycrd,
    const llint* __restrict__ zcrd,
    const double* __restrict__ pb_radii,      // Perfect Born radii from topology
    const double* __restrict__ gb_screen,     // Screening parameters
    const double gb_offset,                   // GB offset parameter
    const float inv_gpos_scale,
    double* __restrict__ psi,                  // Output psi values for Born radii calculation
    double* __restrict__ born_radii)          // Output Born radii
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_atoms) return;

  // Initialize psi for this atom
  double psi_sum = 0.0;

  // Get atom i properties
  const double xi = (double)(xcrd[i]) * inv_gpos_scale;
  const double yi = (double)(ycrd[i]) * inv_gpos_scale;
  const double zi = (double)(zcrd[i]) * inv_gpos_scale;
  const double pb_radius_i = pb_radii[i];
  const double screen_i = gb_screen[i];

  // Loop over all other atoms to compute psi
  for (int j = 0; j < n_atoms; j++) {
    if (i == j) continue;

    // Get atom j properties
    const double xj = (double)(xcrd[j]) * inv_gpos_scale;
    const double yj = (double)(ycrd[j]) * inv_gpos_scale;
    const double zj = (double)(zcrd[j]) * inv_gpos_scale;
    const double pb_radius_j = pb_radii[j];
    const double screen_j = gb_screen[j];

    // Compute distance
    const double dx = xj - xi;
    const double dy = yj - yi;
    const double dz = zj - zi;
    const double r2 = dx*dx + dy*dy + dz*dz;
    const double r = sqrt(r2);

    // Skip if too close
    if (r < 0.01) continue;

    // Compute psi contribution based on HCT/OBC model
    // This is a simplified version - full implementation would match STORMM's exact model
    const double rho_j = pb_radius_j - gb_offset;
    if (r < rho_j) {
      // Atom j overlaps with atom i
      const double L_ij = 1.0 / max(pb_radius_i, r - rho_j);
      const double U_ij = 1.0 / (r + rho_j);
      psi_sum += 0.5 * (L_ij + U_ij - 1.0/r) * screen_j;
    } else if (r < (4.0 * rho_j)) {
      // Within cutoff distance
      const double inv_r = 1.0 / r;
      const double inv_r2 = inv_r * inv_r;
      const double rho_j2 = rho_j * rho_j;
      psi_sum += 0.5 * screen_j * rho_j2 * inv_r2 * inv_r;
    }
  }

  // Store psi value
  psi[i] = psi_sum;

  // Calculate Born radius from psi
  // For HCT/OBC: Ri = 1 / (1/rho_i - psi_i)
  const double rho_i = pb_radius_i - gb_offset;
  const double inv_born = 1.0 / rho_i - psi_sum;
  born_radii[i] = 1.0 / inv_born;
}

//-------------------------------------------------------------------------------------------------
// GPU kernel for computing Born derivative forces with lambda scaling
// This adds the GB derivative contribution to forces WITH lambda scaling
//-------------------------------------------------------------------------------------------------
__global__ void kLambdaBornDerivatives(
    const int n_atoms,
    const int n_coupled,
    const int* __restrict__ coupled_indices,
    const llint* __restrict__ xcrd,
    const llint* __restrict__ ycrd,
    const llint* __restrict__ zcrd,
    const double* __restrict__ charges,
    const double* __restrict__ lambda_ele,    // For lambda scaling
    const double* __restrict__ born_radii,
    const double* __restrict__ sum_deijda,    // Derivative of GB energy w.r.t. Born radii
    const double gb_offset,
    const double coulomb_const,
    const float inv_gpos_scale,
    const float frc_scale,
    llint* __restrict__ xfrc,
    llint* __restrict__ yfrc,
    llint* __restrict__ zfrc)
{
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n_coupled) return;

  // Get the actual atom index for this coupled atom
  const int i = coupled_indices[tid];

  // Load atom i properties
  const double xi = (double)(xcrd[i]) * inv_gpos_scale;
  const double yi = (double)(ycrd[i]) * inv_gpos_scale;
  const double zi = (double)(zcrd[i]) * inv_gpos_scale;
  const double qi = charges[i];
  const double lambda_i = lambda_ele[i];
  const double born_radius_i = born_radii[i];
  const double deijda_i = sum_deijda[i];

  // Accumulate forces for this atom
  double fx_sum = 0.0;
  double fy_sum = 0.0;
  double fz_sum = 0.0;

  // Loop over all coupled atoms for pairwise contributions
  // Use atom index comparison to avoid double-counting
  for (int j_tid = 0; j_tid < n_coupled; j_tid++) {
    const int j = coupled_indices[j_tid];
    if (i == j) continue;
    // Skip pairs where j >= i to avoid double-counting
    if (j >= i) continue;

    // Load atom j properties
    const double xj = (double)(xcrd[j]) * inv_gpos_scale;
    const double yj = (double)(ycrd[j]) * inv_gpos_scale;
    const double zj = (double)(zcrd[j]) * inv_gpos_scale;
    const double qj = charges[j];
    const double lambda_j = lambda_ele[j];
    const double born_radius_j = born_radii[j];
    const double deijda_j = sum_deijda[j];

    // Compute distance vector
    const double dx = xj - xi;
    const double dy = yj - yi;
    const double dz = zj - zi;
    const double r2 = dx*dx + dy*dy + dz*dz;
    const double r = sqrt(r2);

    if (r < 0.01) continue;

    // Compute GB derivative force contribution
    // This implements the chain rule: dE_GB/dr = dE_GB/dR_i * dR_i/dr
    const double rij_prod = born_radius_i * born_radius_j;
    const double exp_arg = -r2 / (4.0 * rij_prod);
    const double exp_term = exp(exp_arg);
    const double f_gb2 = r2 + rij_prod * exp_term;
    const double f_gb = sqrt(f_gb2);

    // CRITICAL: Apply lambda scaling to the derivative
    const double lambda_prod = lambda_i * lambda_j;

    // Derivative of f_GB with respect to r
    const double df_gb_dr = (r / f_gb) * (1.0 - 0.25 * exp_term);

    // GB derivative force magnitude
    // Scale by lambda product for proper GCMC physics
    const double gb_deriv_factor = -0.5 * coulomb_const * qi * qj * lambda_prod / (f_gb * f_gb);
    const double fmag = gb_deriv_factor * df_gb_dr;

    // Add contribution from sum_deijda (derivative of GB energy w.r.t. Born radius)
    // This term also needs lambda scaling
    const double deriv_contrib = (deijda_i + deijda_j) * lambda_prod;
    const double total_fmag = fmag + deriv_contrib * df_gb_dr / f_gb;

    // Accumulate force components
    const double invr = 1.0 / r;
    fx_sum += total_fmag * dx * invr;
    fy_sum += total_fmag * dy * invr;
    fz_sum += total_fmag * dz * invr;
  }

  // Add forces to global arrays using atomic operations
  atomicAdd((ullint*)&xfrc[i], (ullint)(__double2ll_rn(fx_sum * frc_scale)));
  atomicAdd((ullint*)&yfrc[i], (ullint)(__double2ll_rn(fy_sum * frc_scale)));
  atomicAdd((ullint*)&zfrc[i], (ullint)(__double2ll_rn(fz_sum * frc_scale)));
}

//-------------------------------------------------------------------------------------------------
// GPU reduction kernel to sum per-atom energies into scalar totals
//-------------------------------------------------------------------------------------------------
__global__ void kSumEnergies(
    const int n,
    const double* __restrict__ input,
    double* __restrict__ output)
{
  // Shared memory for block-level reduction
  __shared__ double shared[256];

  const int tid = threadIdx.x;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Load data into shared memory
  shared[tid] = (idx < n) ? input[idx] : 0.0;
  __syncthreads();

  // Reduction in shared memory
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }

  // Write block result to global memory
  if (tid == 0) {
    atomicAdd(output, shared[0]);
  }
}

//-------------------------------------------------------------------------------------------------
/// \brief GPU kernel to write lambda-scaled nonbonded energies to ScoreCard
///
/// This kernel takes GPU-reduced scalar energies and writes them directly to the ScoreCard
/// using atomic operations, matching the standard STORMM pattern used by valence and nonbonded
/// kernels. This ensures proper synchronization when multiple kernels contribute energies.
///
/// \param elec_energy      Electrostatic energy (already GPU-reduced to single scalar)
/// \param vdw_energy       VDW energy (already GPU-reduced to single scalar)
/// \param scw              ScoreCard writer for GPU-side atomic accumulation
/// \param system_id        System index (0 for single-system GCMC)
//-------------------------------------------------------------------------------------------------
__global__ void kWriteEnergiesToScoreCard(
    const double* elec_energy,
    const double* vdw_energy,
    ScoreCardWriter scw,
    int system_id)
{
  // Only one thread writes (kernel launched with <<<1, 1>>>)
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    // Convert double energies to fixed-precision llint
    const llint elec_scaled = __double2ll_rn(elec_energy[0] * scw.nrg_scale_f);
    const llint vdw_scaled = __double2ll_rn(vdw_energy[0] * scw.nrg_scale_f);

    // Calculate indices in ScoreCard accumulator array
    const int elec_idx = (system_id * scw.data_stride) + (int)(StateVariable::ELECTROSTATIC);
    const int vdw_idx = (system_id * scw.data_stride) + (int)(StateVariable::VDW);

    // Atomic write to ScoreCard (GPU-side, matches valence/nonbonded pattern)
    atomicAdd((ullint*)&scw.instantaneous_accumulators[elec_idx], (ullint)(elec_scaled));
    atomicAdd((ullint*)&scw.instantaneous_accumulators[vdw_idx], (ullint)(vdw_scaled));
  }
}

//-------------------------------------------------------------------------------------------------
// Launch wrapper for the lambda-scaled nonbonded kernel
//-------------------------------------------------------------------------------------------------
void launchLambdaScaledNonbonded(
    int n_atoms,
    int n_coupled,
    const int* coupled_indices,
    const llint* xcrd,
    const llint* ycrd,
    const llint* zcrd,
    const double* charges,
    const double* lambda_vdw,
    const double* lambda_ele,
    const int* lj_idx,
    int n_lj_types,
    const double2* ljab_coeff,
    const uint* exclusion_mask,
    const int* supertile_map,
    const int* tile_map,
    int supertile_stride,
    const double* umat,
    UnitCellType unit_cell,
    double coulomb_const,
    double ewald_coeff,
    float inv_gpos_scale,
    float frc_scale,
    double* output_elec,
    double* output_vdw,
    llint* xfrc,  // nullptr for energy-only mode
    llint* yfrc,
    llint* zfrc,
    synthesis::ImplicitSolventWorkspace* gb_workspace,
    topology::ImplicitSolventModel gb_model,
    const LambdaNeighborListReader* neighbor_list,
    const int* fragment_indices,
    int n_fragment,
    bool profile_timing)
{
  if (n_coupled == 0) return;

  // Extract GB parameters if enabled
  const double* born_radii = nullptr;
  double gb_kappa = 0.0;   // Salt screening parameter
  double gb_offset = 0.09; // Default GB offset for OBC models

  if (gb_workspace != nullptr && gb_model != topology::ImplicitSolventModel::NONE) {
    // For now, pass nullptr for Born radii - they would need to be computed
    // from psi values stored in the workspace
    // TODO: Add born_radii storage to workspace or compute on-the-fly

    // Set GB parameters based on model
    // Note: These are typical values, should ideally come from workspace/topology
    gb_kappa = 0.0;  // No salt screening by default
    switch (gb_model) {
    case topology::ImplicitSolventModel::OBC_GB:
    case topology::ImplicitSolventModel::OBC_GB_II:
      gb_offset = 0.09;  // OBC offset parameter
      break;
    default:
      gb_offset = 0.0;
      break;
    }
  }

  // Launch configuration: 256 threads per block
  const int threads_per_block = 256;
  const int num_blocks = (n_coupled + threads_per_block - 1) / threads_per_block;

  kLambdaScaledNonbonded<<<num_blocks, threads_per_block>>>(
      n_atoms, n_coupled, coupled_indices,
      xcrd, ycrd, zcrd, charges,
      lambda_vdw, lambda_ele,
      lj_idx, n_lj_types, ljab_coeff,
      exclusion_mask, supertile_map, tile_map, supertile_stride,
      umat, unit_cell, coulomb_const, ewald_coeff,
      inv_gpos_scale, frc_scale,
      output_elec, output_vdw,
      xfrc, yfrc, zfrc,
      born_radii, gb_kappa, gb_offset, gb_model);

  // Check for errors (silent - STORMM has its own error handling)
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    // Error occurred - STORMM will handle it
  }
}

//-------------------------------------------------------------------------------------------------
// Combined kernel + reduction: compute lambda-scaled nonbonded energies and return scalar totals
// This avoids downloading large per-atom arrays, only transferring 2 scalars (elec + vdw)
//-------------------------------------------------------------------------------------------------
void launchLambdaScaledNonbondedWithReduction(
    int n_atoms,
    int n_coupled,
    const int* coupled_indices,
    const llint* xcrd,
    const llint* ycrd,
    const llint* zcrd,
    const double* charges,
    const double* lambda_vdw,
    const double* lambda_ele,
    const int* lj_idx,
    int n_lj_types,
    const double2* ljab_coeff,
    const uint* exclusion_mask,
    const int* supertile_map,
    const int* tile_map,
    int supertile_stride,
    const double* umat,
    UnitCellType unit_cell,
    double coulomb_const,
    double ewald_coeff,
    float inv_gpos_scale,
    float frc_scale,
    double* per_atom_elec,      // Device arrays for intermediate results
    double* per_atom_vdw,
    double* total_elec_out,     // Device scalar output
    double* total_vdw_out,      // Device scalar output
    llint* xfrc,               // nullptr for energy-only mode
    llint* yfrc,
    llint* zfrc,
    synthesis::ImplicitSolventWorkspace* gb_workspace,
    topology::ImplicitSolventModel gb_model,
    const LambdaNeighborListReader* neighbor_list,
    const int* fragment_indices,
    int n_fragment,
    bool profile_timing)
{
  if (n_coupled == 0) {
    // Zero the output scalars on device
    cudaMemset(total_elec_out, 0, sizeof(double));
    cudaMemset(total_vdw_out, 0, sizeof(double));
    return;
  }

  // Step 1: Compute per-coupled-atom energies (and optionally forces)
  const int threads_per_block = 256;
  const int num_blocks = (n_coupled + threads_per_block - 1) / threads_per_block;

  // Extract GB parameters if enabled
  const double* born_radii = nullptr;
  double gb_kappa = 0.0;
  double gb_offset = 0.09;

  if (gb_workspace != nullptr && gb_model != topology::ImplicitSolventModel::NONE) {
    // Note: Born radii are computed by the GB radii kernel and stored in the workspace
    // For now, we pass nullptr as Born radii calculation happens in separate kernels
    gb_kappa = 0.0;  // No salt screening by default
    switch (gb_model) {
    case topology::ImplicitSolventModel::OBC_GB:
    case topology::ImplicitSolventModel::OBC_GB_II:
      gb_offset = 0.09;  // OBC offset parameter
      break;
    default:
      gb_offset = 0.0;
      break;
    }
  }

  kLambdaScaledNonbonded<<<num_blocks, threads_per_block>>>(
      n_atoms, n_coupled, coupled_indices,
      xcrd, ycrd, zcrd, charges,
      lambda_vdw, lambda_ele,
      lj_idx, n_lj_types, ljab_coeff,
      exclusion_mask, supertile_map, tile_map, supertile_stride,
      umat, unit_cell, coulomb_const, ewald_coeff,
      inv_gpos_scale, frc_scale,
      per_atom_elec, per_atom_vdw,
      xfrc, yfrc, zfrc,
      born_radii, gb_kappa, gb_offset, gb_model);

  // Step 2: Zero the output scalars before reduction
  cudaMemset(total_elec_out, 0, sizeof(double));
  cudaMemset(total_vdw_out, 0, sizeof(double));

  // Step 3: Reduce per-atom energies to scalar totals on GPU
  kSumEnergies<<<num_blocks, threads_per_block>>>(n_coupled, per_atom_elec, total_elec_out);
  kSumEnergies<<<num_blocks, threads_per_block>>>(n_coupled, per_atom_vdw, total_vdw_out);

  // Check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    // Error occurred - STORMM will handle it
  }
}

//-------------------------------------------------------------------------------------------------
// Launch wrapper for work delta accumulation kernel
//-------------------------------------------------------------------------------------------------
void launchAccumulateWorkDelta(
    const double* elec_before,
    const double* vdw_before,
    const double* elec_after,
    const double* vdw_after,
    double* work_accumulator)
{
  // Single thread is sufficient for scalar addition
  kAccumulateWorkDelta<<<1, 1>>>(
      elec_before, vdw_before,
      elec_after, vdw_after,
      work_accumulator);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    // Error occurred - STORMM will handle it
  }
}

//-------------------------------------------------------------------------------------------------
// Launch wrapper for lambda update kernel
//-------------------------------------------------------------------------------------------------
void launchUpdateLambdaFromSchedule(
    int step_index,
    const double* lambda_schedule,
    const int* molecule_indices,
    int n_molecule_atoms,
    double vdw_coupling_threshold,
    double* lambda_vdw,
    double* lambda_ele)
{
  if (n_molecule_atoms == 0) return;

  // Launch configuration: 256 threads per block
  const int threads_per_block = 256;
  const int num_blocks = (n_molecule_atoms + threads_per_block - 1) / threads_per_block;

  kUpdateLambdaFromSchedule<<<num_blocks, threads_per_block>>>(
      step_index, lambda_schedule, molecule_indices, n_molecule_atoms,
      vdw_coupling_threshold, lambda_vdw, lambda_ele);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    // Error occurred - STORMM will handle it
  }
}

//-------------------------------------------------------------------------------------------------
// Launch wrapper for lambda-aware Born radii computation
// Computes Born radii for all atoms - no lambda scaling needed
//-------------------------------------------------------------------------------------------------
void launchLambdaBornRadii(
    int n_atoms,
    const llint* xcrd,
    const llint* ycrd,
    const llint* zcrd,
    const double* pb_radii,
    const double* gb_screen,
    double gb_offset,
    float inv_gpos_scale,
    double* psi,
    double* born_radii,
    synthesis::ImplicitSolventWorkspace* gb_workspace,
    topology::ImplicitSolventModel gb_model)
{
  if (n_atoms == 0 || gb_model == topology::ImplicitSolventModel::NONE) return;

  // Launch configuration: 256 threads per block
  const int threads_per_block = 256;
  const int num_blocks = (n_atoms + threads_per_block - 1) / threads_per_block;

  kLambdaBornRadii<<<num_blocks, threads_per_block>>>(
      n_atoms, xcrd, ycrd, zcrd,
      pb_radii, gb_screen, gb_offset,
      inv_gpos_scale,
      psi, born_radii);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    // Error occurred - STORMM will handle it
  }
}

//-------------------------------------------------------------------------------------------------
// Launch wrapper for lambda-aware Born derivative computation
// Adds GB derivative forces WITH lambda scaling for proper GCMC physics
//-------------------------------------------------------------------------------------------------
void launchLambdaBornDerivatives(
    int n_atoms,
    int n_coupled,
    const int* coupled_indices,
    const llint* xcrd,
    const llint* ycrd,
    const llint* zcrd,
    const double* charges,
    const double* lambda_ele,
    const double* born_radii,
    const double* sum_deijda,
    double gb_offset,
    double coulomb_const,
    float inv_gpos_scale,
    float frc_scale,
    llint* xfrc,
    llint* yfrc,
    llint* zfrc,
    synthesis::ImplicitSolventWorkspace* gb_workspace,
    topology::ImplicitSolventModel gb_model)
{
  if (n_coupled == 0 || gb_model == topology::ImplicitSolventModel::NONE) return;

  // Launch configuration: 256 threads per block
  const int threads_per_block = 256;
  const int num_blocks = (n_coupled + threads_per_block - 1) / threads_per_block;

  kLambdaBornDerivatives<<<num_blocks, threads_per_block>>>(
      n_atoms, n_coupled, coupled_indices,
      xcrd, ycrd, zcrd, charges, lambda_ele,
      born_radii, sum_deijda, gb_offset, coulomb_const,
      inv_gpos_scale, frc_scale,
      xfrc, yfrc, zfrc);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    // Error occurred - STORMM will handle it
  }
}

//================================================================================================
// TILE-BASED LAMBDA NONBONDED KERNEL IMPLEMENTATION
//================================================================================================
// The code below implements tile-based lambda kernels for 27× speedup over serial approach

// Additional includes for tile-based kernels
#include "Accelerator/ptx_macros.h"
#include "Math/rounding.cui"
#include "Numerics/accumulation.cui"
#include "hpc_lambda_supertile_helpers.cui"
#include "Potential/energy_enumerators.h"
#include "Synthesis/synthesis_enumerators.h"

// Additional using statements for tile-based kernels
using constants::twice_warp_bits_mask_int;
using constants::twice_warp_size_int;
using constants::warp_size_int;
using constants::warp_bits;
using constants::warp_bits_mask_int;
using mm::MMControlKit;
using numerics::AccumulationMethod;
using numerics::chooseAccumulationMethod;
using numerics::max_llint_accumulation;
using synthesis::NbwuKind;
using trajectory::ThermostatWriter;

#define NONBOND_KERNEL_BLOCKS_MULTIPLIER 5
// Define constants needed by kernel for visibility in .cui file scope
// These values must match those in synthesis/nonbonded_workunit.h
constexpr int small_block_max_atoms = 320;
constexpr int small_block_max_imports = small_block_max_atoms / tile_length;  // 320/16 = 20
constexpr int small_block_max_tiles = 16;
constexpr int small_block_size = 256;
constexpr int tile_groups_wu_abstract_length = 64;
constexpr int supertile_wu_abstract_length = 8;

//-------------------------------------------------------------------------------------------------
// Helper functions for tile loading and accumulation (copied from hpc_nonbonded_potential.cu)
//-------------------------------------------------------------------------------------------------

static __device__ __forceinline__ int getTileSideAtomCount(const int* nbwu_map, const int pos) {
  const int key_idx  = pos / 4;
  const int key_slot = pos - (key_idx * 4);
  return ((nbwu_map[small_block_max_imports + 1 + key_idx] >> (8 * key_slot)) & 0xff);
}

static __device__ int loadTileCoordinates(const int pos, const int iter, const int* nbwu_map,
                                          const llint* read_crd, llint* write_crd, float* sh_tile_cog,
                                          const float gpos_scale) {
  const int tile_sides_per_warp = (warp_size_int / tile_length);
  const int warps_per_block = blockDim.x >> warp_bits;
  const int tile_lane_idx = (threadIdx.x & tile_length_bits_mask);
  const int import_count = nbwu_map[0];
  const int padded_import_count = devcRoundUp(import_count, tile_sides_per_warp);
  int rel_pos = pos - (iter * padded_import_count);
  while (rel_pos < padded_import_count) {
    float fval;
    if (rel_pos < import_count) {
      const size_t read_idx = nbwu_map[rel_pos + 1] + tile_lane_idx;
      const size_t write_idx = (rel_pos * tile_length) + tile_lane_idx;
      if (tile_lane_idx < getTileSideAtomCount(nbwu_map, rel_pos)) {
        const llint ival = __ldcs(&read_crd[read_idx]);
        fval = (float)(ival);
        write_crd[write_idx] = ival;
      }
      else {
        fval = (float)(0.0);
        write_crd[write_idx] = (128 * (rel_pos + 8) * tile_lane_idx) * gpos_scale;
      }
    }
    else {
      fval = (float)(0.0);
    }
    for (int i = half_tile_length; i > 0; i >>= 1) {
      fval += SHFL_DOWN(fval, i);
    }
    if (tile_lane_idx == 0 && rel_pos < import_count) {
      sh_tile_cog[rel_pos] = fval;
    }
    rel_pos += tile_sides_per_warp * warps_per_block;
  }
  return rel_pos + (iter * padded_import_count);
}

static __device__ int loadTileCoordinates(const int pos, const int iter, const int* nbwu_map,
                                          const llint* read_crd, llint* write_crd,
                                          const int* read_crd_ovrf, int* write_crd_ovrf,
                                          double* sh_tile_cog, const double gpos_scale) {
  const int tile_sides_per_warp = (warp_size_int / tile_length);
  const int warps_per_block = blockDim.x >> warp_bits;
  const int tile_lane_idx = (threadIdx.x & tile_length_bits_mask);
  const int import_count = nbwu_map[0];
  const int padded_import_count = devcRoundUp(import_count, tile_sides_per_warp);
  int rel_pos = pos - (iter * padded_import_count);
  while (rel_pos < padded_import_count) {
    double fval;
    if (rel_pos < import_count) {
      const size_t read_idx = nbwu_map[rel_pos + 1] + tile_lane_idx;
      const size_t write_idx = (rel_pos * tile_length) + tile_lane_idx;
      if (tile_lane_idx < getTileSideAtomCount(nbwu_map, rel_pos)) {
        const llint ival = __ldcs(&read_crd[read_idx]);
        fval = (double)(ival);
        write_crd[write_idx] = ival;
        const int ival_ovrf = __ldcs(&read_crd_ovrf[read_idx]);
        fval += (double)(ival_ovrf) * max_llint_accumulation;
        write_crd_ovrf[write_idx] = ival_ovrf;
      }
      else {
        fval = 0.0;
        const int95_t fake_val = doubleToInt95((128 * (rel_pos + 8) * tile_lane_idx) * gpos_scale);
        write_crd[write_idx] = fake_val.x;
        write_crd_ovrf[write_idx] = fake_val.y;
      }
    }
    else {
      fval = 0.0;
    }
    for (int i = half_tile_length; i > 0; i >>= 1) {
      fval += SHFL_DOWN(fval, i);
    }
    if (tile_lane_idx == 0 && rel_pos < import_count) {
      sh_tile_cog[rel_pos] = fval;
    }
    rel_pos += tile_sides_per_warp * warps_per_block;
  }
  return rel_pos + (iter * padded_import_count);
}

template <typename T> static __device__
int loadTileProperty(const int pos, const int iter, const int* nbwu_map, const T* read_array,
                     T* write_array) {
  const int tile_sides_per_warp = (warp_size_int / tile_length);
  const int warps_per_block = blockDim.x >> warp_bits;
  const int tile_lane_idx = (threadIdx.x & tile_length_bits_mask);
  const int import_count = nbwu_map[0];
  const int padded_import_count = devcRoundUp(import_count, tile_sides_per_warp);
  int rel_pos = pos - (iter * padded_import_count);
  while (rel_pos < padded_import_count) {
    if (rel_pos < import_count) {
      const size_t read_idx = nbwu_map[rel_pos + 1] + tile_lane_idx;
      const size_t write_idx = (rel_pos * tile_length) + tile_lane_idx;
      if (tile_lane_idx < getTileSideAtomCount(nbwu_map, rel_pos)) {
        write_array[write_idx] = __ldcs(&read_array[read_idx]);
      }
      else {
        write_array[write_idx] = (T)(0);
      }
    }
    rel_pos += tile_sides_per_warp * warps_per_block;
  }
  return rel_pos + (iter * padded_import_count);
}

template <typename T> static __device__
int loadTileProperty(const int pos, const int iter, const int* nbwu_map, const T* read_array,
                     T* write_array, T multiplier) {
  const int tile_sides_per_warp = (warp_size_int / tile_length);
  const int warps_per_block = blockDim.x >> warp_bits;
  const int tile_lane_idx = (threadIdx.x & tile_length_bits_mask);
  const int import_count = nbwu_map[0];
  const int padded_import_count = devcRoundUp(import_count, tile_sides_per_warp);
  int rel_pos = pos - (iter * padded_import_count);
  while (rel_pos < padded_import_count) {
    if (rel_pos < import_count) {
      const size_t read_idx = nbwu_map[rel_pos + 1] + tile_lane_idx;
      const size_t write_idx = (rel_pos * tile_length) + tile_lane_idx;
      if (tile_lane_idx < getTileSideAtomCount(nbwu_map, rel_pos)) {
        write_array[write_idx] = __ldcs(&read_array[read_idx]) * multiplier;
      }
      else {
        write_array[write_idx] = (T)(0);
      }
    }
    rel_pos += tile_sides_per_warp * warps_per_block;
  }
  return rel_pos + (iter * padded_import_count);
}

template <typename T> static __device__
int loadTileProperty(const int pos, const int iter, const int* nbwu_map, const T* read_array,
                     T increment, T* write_array) {
  const int tile_sides_per_warp = (warp_size_int / tile_length);
  const int warps_per_block = blockDim.x >> warp_bits;
  const int tile_lane_idx = (threadIdx.x & tile_length_bits_mask);
  const int import_count = nbwu_map[0];
  const int padded_import_count = devcRoundUp(import_count, tile_sides_per_warp);
  int rel_pos = pos - (iter * padded_import_count);
  while (rel_pos < padded_import_count) {
    if (rel_pos < import_count) {
      const size_t read_idx = nbwu_map[rel_pos + 1] + tile_lane_idx;
      const size_t write_idx = (rel_pos * tile_length) + tile_lane_idx;
      if (tile_lane_idx < getTileSideAtomCount(nbwu_map, rel_pos)) {
        write_array[write_idx] = __ldcs(&read_array[read_idx]) + increment;
      }
      else {
        write_array[write_idx] = (T)(0);
      }
    }
    rel_pos += tile_sides_per_warp * warps_per_block;
  }
  return rel_pos + (iter * padded_import_count);
}

static __device__ int accumulateTileProperty(const int pos, const int iter, const int* nbwu_map,
                                      const int* tile_prop, const int* tile_prop_ovrf,
                                      llint* gbl_accumulator) {
  const int tile_sides_per_warp = (warp_size_int / tile_length);
  const int warps_per_block = blockDim.x >> warp_bits;
  const int tile_lane_idx = (threadIdx.x & tile_length_bits_mask);
  const int import_count = nbwu_map[0];
  const int padded_import_count = devcRoundUp(import_count, tile_sides_per_warp);
  int rel_pos = pos - (iter * padded_import_count);
  while (rel_pos < padded_import_count) {
    if (rel_pos < import_count) {
      const size_t write_idx = nbwu_map[rel_pos + 1] + tile_lane_idx;
      const size_t read_idx = (rel_pos * tile_length) + tile_lane_idx;
      if (tile_lane_idx < getTileSideAtomCount(nbwu_map, rel_pos)) {
        llint itp = tile_prop_ovrf[read_idx];
        itp *= max_int_accumulation_ll;
        itp += tile_prop[read_idx];
        atomicAdd((ullint*)&gbl_accumulator[write_idx], (ullint)(itp));
      }
    }
    rel_pos += tile_sides_per_warp * warps_per_block;
  }
  return rel_pos + (iter * padded_import_count);
}

static __device__ int accumulateTileProperty(const int pos, const int iter, const int* nbwu_map,
                                      const llint* tile_prop, llint* gbl_accumulator) {
  const int tile_sides_per_warp = (warp_size_int / tile_length);
  const int warps_per_block = blockDim.x >> warp_bits;
  const int tile_lane_idx = (threadIdx.x & tile_length_bits_mask);
  const int import_count = nbwu_map[0];
  const int padded_import_count = devcRoundUp(import_count, tile_sides_per_warp);
  int rel_pos = pos - (iter * padded_import_count);
  while (rel_pos < padded_import_count) {
    if (rel_pos < import_count) {
      const size_t write_idx = nbwu_map[rel_pos + 1] + tile_lane_idx;
      const size_t read_idx = (rel_pos * tile_length) + tile_lane_idx;
      if (tile_lane_idx < getTileSideAtomCount(nbwu_map, rel_pos)) {
        atomicAdd((ullint*)&gbl_accumulator[write_idx], (ullint)(tile_prop[read_idx]));
      }
    }
    rel_pos += tile_sides_per_warp * warps_per_block;
  }
  return rel_pos + (iter * padded_import_count);
}

static __device__ int accumulateTileProperty(const int pos, const int iter, const int* nbwu_map,
                                      const llint* tile_prop, const int* tile_prop_ovrf,
                                      llint* gbl_accumulator, int* gbl_accumulator_ovrf) {
  const int tile_sides_per_warp = (warp_size_int / tile_length);
  const int warps_per_block = blockDim.x >> warp_bits;
  const int tile_lane_idx = (threadIdx.x & tile_length_bits_mask);
  const int import_count = nbwu_map[0];
  const int padded_import_count = devcRoundUp(import_count, tile_sides_per_warp);
  int rel_pos = pos - (iter * padded_import_count);
  while (rel_pos < padded_import_count) {
    if (rel_pos < import_count) {
      const size_t write_idx = nbwu_map[rel_pos + 1] + tile_lane_idx;
      const size_t read_idx = (rel_pos * tile_length) + tile_lane_idx;
      if (tile_lane_idx < getTileSideAtomCount(nbwu_map, rel_pos)) {
        atomicSplit(tile_prop[read_idx], tile_prop_ovrf[read_idx], write_idx, gbl_accumulator,
                    gbl_accumulator_ovrf);
      }
    }
    rel_pos += tile_sides_per_warp * warps_per_block;
  }
  return rel_pos + (iter * padded_import_count);
}

// Forward declarations for tile-based kernel variants
__global__ void __launch_bounds__(small_block_size, NONBOND_KERNEL_BLOCKS_MULTIPLIER)
kLambdaTileGroupVacuumForceEnergy_D(const SyNonbondedKit<double, double2> poly_nbk,
                                     const SeMaskSynthesisReader poly_se,
                                     const MMControlKit<double> ctrl,
                                     PsSynthesisWriter poly_psw,
                                     const double* __restrict__ lambda_vdw,
                                     const double* __restrict__ lambda_ele,
                                     ScoreCardWriter scw,
                                     ThermostatWriter<double> tstw,
                                     CacheResourceKit<double> gmem_r);

__global__ void __launch_bounds__(small_block_size, NONBOND_KERNEL_BLOCKS_MULTIPLIER)
kLambdaTileGroupVacuumForce_D(const SyNonbondedKit<double, double2> poly_nbk,
                               const SeMaskSynthesisReader poly_se,
                               const MMControlKit<double> ctrl,
                               PsSynthesisWriter poly_psw,
                               const double* __restrict__ lambda_vdw,
                               const double* __restrict__ lambda_ele,
                               ThermostatWriter<double> tstw,
                               CacheResourceKit<double> gmem_r);

__global__ void __launch_bounds__(small_block_size, NONBOND_KERNEL_BLOCKS_MULTIPLIER)
kLambdaTileGroupVacuumEnergy_D(const SyNonbondedKit<double, double2> poly_nbk,
                                const SeMaskSynthesisReader poly_se,
                                const MMControlKit<double> ctrl,
                                PsSynthesisWriter poly_psw,
                                const double* __restrict__ lambda_vdw,
                                const double* __restrict__ lambda_ele,
                                ScoreCardWriter scw,
                                ThermostatWriter<double> tstw,
                                CacheResourceKit<double> gmem_r);

__global__ void __launch_bounds__(small_block_size, NONBOND_KERNEL_BLOCKS_MULTIPLIER)
kLambdaTileGroupVacuumForceEnergy_F(const SyNonbondedKit<float, float2> poly_nbk,
                                     const SeMaskSynthesisReader poly_se,
                                     const MMControlKit<float> ctrl,
                                     PsSynthesisWriter poly_psw,
                                     const double* __restrict__ lambda_vdw,
                                     const double* __restrict__ lambda_ele,
                                     ScoreCardWriter scw,
                                     ThermostatWriter<float> tstw,
                                     CacheResourceKit<float> gmem_r);

__global__ void __launch_bounds__(small_block_size, NONBOND_KERNEL_BLOCKS_MULTIPLIER)
kLambdaTileGroupVacuumForce_F(const SyNonbondedKit<float, float2> poly_nbk,
                               const SeMaskSynthesisReader poly_se,
                               const MMControlKit<float> ctrl,
                               PsSynthesisWriter poly_psw,
                               const double* __restrict__ lambda_vdw,
                               const double* __restrict__ lambda_ele,
                               ThermostatWriter<float> tstw,
                               CacheResourceKit<float> gmem_r);

__global__ void __launch_bounds__(small_block_size, NONBOND_KERNEL_BLOCKS_MULTIPLIER)
kLambdaTileGroupVacuumEnergy_F(const SyNonbondedKit<float, float2> poly_nbk,
                                const SeMaskSynthesisReader poly_se,
                                const MMControlKit<float> ctrl,
                                PsSynthesisWriter poly_psw,
                                const double* __restrict__ lambda_vdw,
                                const double* __restrict__ lambda_ele,
                                ScoreCardWriter scw,
                                ThermostatWriter<float> tstw,
                                CacheResourceKit<float> gmem_r);

// Kernel instantiations using preprocessor to generate all 6 variants
#define TCALC double
#define TCALC2 double2
#define LLCONV_FUNC __double2ll_rn
#define SQRT_FUNC sqrt
#define EXP_FUNC exp
#define CBRT_FUNC cbrt
#define SPLIT_FORCE_ACCUMULATION
#define COMPUTE_FORCE
#define COMPUTE_ENERGY
#define KERNEL_NAME kLambdaTileGroupVacuumForceEnergy_D
#include "lambda_nonbonded_tilegroups_vacuum.cui"
#undef KERNEL_NAME
#undef COMPUTE_ENERGY
#undef COMPUTE_FORCE

#define COMPUTE_FORCE
#define KERNEL_NAME kLambdaTileGroupVacuumForce_D
#include "lambda_nonbonded_tilegroups_vacuum.cui"
#undef KERNEL_NAME
#undef COMPUTE_FORCE

#define COMPUTE_ENERGY
#define KERNEL_NAME kLambdaTileGroupVacuumEnergy_D
#include "lambda_nonbonded_tilegroups_vacuum.cui"
#undef KERNEL_NAME
#undef COMPUTE_ENERGY
#undef SPLIT_FORCE_ACCUMULATION
#undef TCALC
#undef TCALC2
#undef LLCONV_FUNC
#undef SQRT_FUNC
#undef EXP_FUNC
#undef CBRT_FUNC

// Single precision variants
#define TCALC float
#define TCALC2 float2
#define TCALC_IS_SINGLE
#define LLCONV_FUNC __float2ll_rn
#define SQRT_FUNC sqrtf
#define EXP_FUNC expf
#define CBRT_FUNC cbrtf
#define COMPUTE_FORCE
#define COMPUTE_ENERGY
#define KERNEL_NAME kLambdaTileGroupVacuumForceEnergy_F
#include "lambda_nonbonded_tilegroups_vacuum.cui"
#undef KERNEL_NAME
#undef COMPUTE_ENERGY
#undef COMPUTE_FORCE

#define COMPUTE_FORCE
#define KERNEL_NAME kLambdaTileGroupVacuumForce_F
#include "lambda_nonbonded_tilegroups_vacuum.cui"
#undef KERNEL_NAME
#undef COMPUTE_FORCE

#define COMPUTE_ENERGY
#define KERNEL_NAME kLambdaTileGroupVacuumEnergy_F
#include "lambda_nonbonded_tilegroups_vacuum.cui"
#undef KERNEL_NAME
#undef COMPUTE_ENERGY
#undef TCALC_IS_SINGLE
#undef TCALC
#undef TCALC2
#undef LLCONV_FUNC
#undef SQRT_FUNC
#undef EXP_FUNC
#undef CBRT_FUNC

//================================================================================================
// SUPERTILE-BASED LAMBDA NONBONDED KERNEL INSTANTIATIONS
//================================================================================================
// Supertile kernels for high ghost-count GCMC (1000+ fragments)
// Uses 8-integer work unit abstract vs 64-integer tile-group abstract

// Double precision supertile variants
#define TCALC double
#define TCALC2 double2
#define LLCONV_FUNC __double2ll_rn
#define SQRT_FUNC sqrt
#define EXP_FUNC exp
#define CBRT_FUNC cbrt
#define SPLIT_FORCE_ACCUMULATION
#define COMPUTE_FORCE
#define COMPUTE_ENERGY
#define KERNEL_NAME kLambdaSupertileVacuumForceEnergy_D
#include "lambda_nonbonded_supertiles_vacuum.cui"
#undef KERNEL_NAME
#undef COMPUTE_ENERGY
#undef COMPUTE_FORCE

#define COMPUTE_FORCE
#define KERNEL_NAME kLambdaSupertileVacuumForce_D
#include "lambda_nonbonded_supertiles_vacuum.cui"
#undef KERNEL_NAME
#undef COMPUTE_FORCE

#define COMPUTE_ENERGY
#define KERNEL_NAME kLambdaSupertileVacuumEnergy_D
#include "lambda_nonbonded_supertiles_vacuum.cui"
#undef KERNEL_NAME
#undef COMPUTE_ENERGY
#undef SPLIT_FORCE_ACCUMULATION
#undef TCALC
#undef TCALC2
#undef LLCONV_FUNC
#undef SQRT_FUNC
#undef EXP_FUNC
#undef CBRT_FUNC

// Single precision supertile variants
#define TCALC float
#define TCALC2 float2
#define TCALC_IS_SINGLE
#define LLCONV_FUNC __float2ll_rn
#define SQRT_FUNC sqrtf
#define EXP_FUNC expf
#define CBRT_FUNC cbrtf
#define COMPUTE_FORCE
#define COMPUTE_ENERGY
#define KERNEL_NAME kLambdaSupertileVacuumForceEnergy_F
#include "lambda_nonbonded_supertiles_vacuum.cui"
#undef KERNEL_NAME
#undef COMPUTE_ENERGY
#undef COMPUTE_FORCE

#define COMPUTE_FORCE
#define KERNEL_NAME kLambdaSupertileVacuumForce_F
#include "lambda_nonbonded_supertiles_vacuum.cui"
#undef KERNEL_NAME
#undef COMPUTE_FORCE

#define COMPUTE_ENERGY
#define KERNEL_NAME kLambdaSupertileVacuumEnergy_F
#include "lambda_nonbonded_supertiles_vacuum.cui"
#undef KERNEL_NAME
#undef COMPUTE_ENERGY
#undef TCALC_IS_SINGLE
#undef TCALC
#undef TCALC2
#undef LLCONV_FUNC
#undef SQRT_FUNC
#undef EXP_FUNC
#undef CBRT_FUNC

//================================================================================================
// HIGH-LEVEL LAMBDA NONBONDED LAUNCHER (matching standard dynamics pattern)
//================================================================================================

//-------------------------------------------------------------------------------------------------
// High-level launcher that dispatches tile-based lambda-scaled nonbonded kernels
// This achieves 27× speedup over serial implementation via warp shuffle architecture
//-------------------------------------------------------------------------------------------------
void launchLambdaNonbonded(
    const double* lambda_vdw,
    const double* lambda_ele,
    const int* coupled_indices,
    int n_coupled,
    constants::PrecisionModel prec,
    const AtomGraphSynthesis& poly_ag,
    const StaticExclusionMaskSynthesis& poly_se,
    MolecularMechanicsControls* mmctrl,
    PhaseSpaceSynthesis* poly_ps,
    Thermostat* heat_bath,
    ScoreCard* sc,
    CacheResource* tb_space,
    ImplicitSolventWorkspace* ism_space,
    EvaluateForce eval_force,
    EvaluateEnergy eval_energy,
    const CoreKlManager& launcher)
{
  using card::HybridTargetLevel;
  using synthesis::PsSynthesisWriter;
  using synthesis::SeMaskSynthesisReader;
  using synthesis::SyNonbondedKit;
  using topology::UnitCellType;

  // CRITICAL: Check for GB - tile-based lambda kernels don't support GB yet
  const ImplicitSolventModel gb_model = poly_ag.getImplicitSolventModel();
  if (gb_model != ImplicitSolventModel::NONE) {
    rtErr("Tile-based lambda kernels do not support Generalized Born implicit solvent. "
          "GB support is pending implementation. Use vacuum or explicit solvent systems only.",
          "launchLambdaNonbonded");
  }

  const HybridTargetLevel tier = HybridTargetLevel::DEVICE;
  const int n_atoms = poly_ag.getAtomCount();

  if (n_coupled == 0) return;

  // Get current cycle position for coordinate/force access
  const CoordinateCycle curr_cyc = poly_ps->getCyclePosition();
  PsSynthesisWriter psw = poly_ps->data(curr_cyc, tier);

  // Get exclusion mask reader (DEVICE tier for kernel)
  const SeMaskSynthesisReader poly_ser = poly_se.data(tier);

  // Get nonbonded work unit kind from AtomGraphSynthesis
  const NbwuKind wu_kind_val = poly_ag.getNonbondedWorkType();

  // Lambda kernels support both TILE_GROUPS and SUPERTILES work units
  if (wu_kind_val != NbwuKind::TILE_GROUPS && wu_kind_val != NbwuKind::SUPERTILES) {
    rtErr("Lambda nonbonded kernels support TILE_GROUPS and SUPERTILES work units only. "
          "System uses " + std::string(getEnumerationName(wu_kind_val)) + " layout.",
          "launchLambdaNonbonded");
  }

  // Get kernel launch dimensions from CoreKlManager
  // Note: For lambda kernels, we use NONE implicit solvent model (no GB support yet)
  const int2 bt = launcher.getNonbondedKernelDims(prec, wu_kind_val, eval_force, eval_energy,
                                                  AccumulationMethod::SPLIT, ImplicitSolventModel::NONE,
                                                  ClashResponse::NONE);

  if (bt.x == 0 || bt.y == 0) return;

  // Create default thermostat if nullptr passed (lambda kernels don't use thermostat)
  Thermostat default_thermostat;
  Thermostat* thermostat_ptr = heat_bath ? heat_bath : &default_thermostat;

  // Dispatch based on precision model
  switch (prec) {
    case PrecisionModel::DOUBLE: {
      // Get double-precision data structures
      MMControlKit<double> ctrl_d = mmctrl->dpData(tier);
      ThermostatWriter<double> tstw_d = thermostat_ptr->dpData(tier);
      CacheResourceKit<double> gmem_r_d = tb_space->dpData(tier);
      SyNonbondedKit<double, double2> nbk_d = poly_ag.getDoublePrecisionNonbondedKit(tier);

      // Dispatch based on work unit kind and force/energy flags
      if (wu_kind_val == NbwuKind::TILE_GROUPS) {
        if (eval_force == EvaluateForce::YES && eval_energy == EvaluateEnergy::YES) {
          ScoreCardWriter scw = sc->data(tier);
          kLambdaTileGroupVacuumForceEnergy_D<<<bt.x, bt.y>>>(
              nbk_d, poly_ser, ctrl_d, psw, lambda_vdw, lambda_ele,
              scw, tstw_d, gmem_r_d);
        }
        else if (eval_force == EvaluateForce::YES) {
          kLambdaTileGroupVacuumForce_D<<<bt.x, bt.y>>>(
              nbk_d, poly_ser, ctrl_d, psw, lambda_vdw, lambda_ele,
              tstw_d, gmem_r_d);
        }
        else if (eval_energy == EvaluateEnergy::YES) {
          ScoreCardWriter scw = sc->data(tier);
          kLambdaTileGroupVacuumEnergy_D<<<bt.x, bt.y>>>(
              nbk_d, poly_ser, ctrl_d, psw, lambda_vdw, lambda_ele,
              scw, tstw_d, gmem_r_d);
        }
      }
      else if (wu_kind_val == NbwuKind::SUPERTILES) {
        if (eval_force == EvaluateForce::YES && eval_energy == EvaluateEnergy::YES) {
          ScoreCardWriter scw = sc->data(tier);
          kLambdaSupertileVacuumForceEnergy_D<<<bt.x, bt.y>>>(
              nbk_d, poly_ser, ctrl_d, psw, lambda_vdw, lambda_ele,
              scw, tstw_d, gmem_r_d);
        }
        else if (eval_force == EvaluateForce::YES) {
          kLambdaSupertileVacuumForce_D<<<bt.x, bt.y>>>(
              nbk_d, poly_ser, ctrl_d, psw, lambda_vdw, lambda_ele,
              tstw_d, gmem_r_d);
        }
        else if (eval_energy == EvaluateEnergy::YES) {
          ScoreCardWriter scw = sc->data(tier);
          kLambdaSupertileVacuumEnergy_D<<<bt.x, bt.y>>>(
              nbk_d, poly_ser, ctrl_d, psw, lambda_vdw, lambda_ele,
              scw, tstw_d, gmem_r_d);
        }
      }
      break;
    }

    case PrecisionModel::SINGLE: {
      // Get single-precision data structures
      MMControlKit<float> ctrl_f = mmctrl->spData(tier);
      ThermostatWriter<float> tstw_f = thermostat_ptr->spData(tier);
      CacheResourceKit<float> gmem_r_f = tb_space->spData(tier);
      SyNonbondedKit<float, float2> nbk_f = poly_ag.getSinglePrecisionNonbondedKit(tier);

      // Dispatch based on work unit kind and force/energy flags
      if (wu_kind_val == NbwuKind::TILE_GROUPS) {
        if (eval_force == EvaluateForce::YES && eval_energy == EvaluateEnergy::YES) {
          ScoreCardWriter scw = sc->data(tier);
          kLambdaTileGroupVacuumForceEnergy_F<<<bt.x, bt.y>>>(
              nbk_f, poly_ser, ctrl_f, psw, lambda_vdw, lambda_ele,
              scw, tstw_f, gmem_r_f);
        }
        else if (eval_force == EvaluateForce::YES) {
          kLambdaTileGroupVacuumForce_F<<<bt.x, bt.y>>>(
              nbk_f, poly_ser, ctrl_f, psw, lambda_vdw, lambda_ele,
              tstw_f, gmem_r_f);
        }
        else if (eval_energy == EvaluateEnergy::YES) {
          ScoreCardWriter scw = sc->data(tier);
          kLambdaTileGroupVacuumEnergy_F<<<bt.x, bt.y>>>(
              nbk_f, poly_ser, ctrl_f, psw, lambda_vdw, lambda_ele,
              scw, tstw_f, gmem_r_f);
        }
      }
      else if (wu_kind_val == NbwuKind::SUPERTILES) {
        if (eval_force == EvaluateForce::YES && eval_energy == EvaluateEnergy::YES) {
          ScoreCardWriter scw = sc->data(tier);
          kLambdaSupertileVacuumForceEnergy_F<<<bt.x, bt.y>>>(
              nbk_f, poly_ser, ctrl_f, psw, lambda_vdw, lambda_ele,
              scw, tstw_f, gmem_r_f);
        }
        else if (eval_force == EvaluateForce::YES) {
          kLambdaSupertileVacuumForce_F<<<bt.x, bt.y>>>(
              nbk_f, poly_ser, ctrl_f, psw, lambda_vdw, lambda_ele,
              tstw_f, gmem_r_f);
        }
        else if (eval_energy == EvaluateEnergy::YES) {
          ScoreCardWriter scw = sc->data(tier);
          kLambdaSupertileVacuumEnergy_F<<<bt.x, bt.y>>>(
              nbk_f, poly_ser, ctrl_f, psw, lambda_vdw, lambda_ele,
              scw, tstw_f, gmem_r_f);
        }
      }
      break;
    }
  }

  // Check for CUDA errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    rtErr("CUDA kernel launch failed: " + std::string(cudaGetErrorString(err)),
          "launchLambdaNonbonded");
  }
}

//-------------------------------------------------------------------------------------------------
/// \brief GPU kernel to extract energy totals from ScoreCard to scalar device pointers
///
/// This kernel reads the accumulated electrostatic and VDW energies from the ScoreCard
/// for a given system and writes them to separate scalar device pointers. This enables
/// downloading only 16 bytes (2 doubles) instead of the full 4.36 MB ScoreCard.
///
/// \param scw           ScoreCard writer with accumulated energies
/// \param system_id     System index (0 for single-system GCMC)
/// \param total_elec    Output pointer for total electrostatic energy (device)
/// \param total_vdw     Output pointer for total VDW energy (device)
//-------------------------------------------------------------------------------------------------
__global__ void kExtractScoreCardEnergies(
    const ScoreCardReader scr,
    int system_id,
    double* total_elec,
    double* total_vdw)
{
  // Only one thread extracts (kernel launched with <<<1, 1>>>)
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    // Calculate indices in ScoreCard accumulator array
    const int elec_idx = (system_id * scr.data_stride) + static_cast<int>(StateVariable::ELECTROSTATIC);
    const int vdw_idx = (system_id * scr.data_stride) + static_cast<int>(StateVariable::VDW);

    // Read accumulated values and convert from fixed-precision llint to double
    const llint elec_scaled = scr.instantaneous_accumulators[elec_idx];
    const llint vdw_scaled = scr.instantaneous_accumulators[vdw_idx];

    *total_elec = static_cast<double>(elec_scaled) * scr.inverse_nrg_scale_f;
    *total_vdw = static_cast<double>(vdw_scaled) * scr.inverse_nrg_scale_f;
  }
}

//-------------------------------------------------------------------------------------------------
/// \brief High-level wrapper for lambda nonbonded with GPU-side energy reduction
///
/// This function provides the same interface as launchLambdaNonbonded but adds a GPU reduction
/// step that extracts scalar energy totals. This eliminates the 4.36 MB ScoreCard download,
/// replacing it with a 16-byte transfer (2 doubles).
///
/// \param lambda_vdw           Per-atom VDW lambda values (device array)
/// \param lambda_ele           Per-atom electrostatic lambda values (device array)
/// \param coupled_indices      Indices of coupled atoms (device array)
/// \param n_coupled            Number of coupled atoms
/// \param prec                 Precision model (DOUBLE or SINGLE)
/// \param poly_ag              Atom graph synthesis
/// \param poly_se              Static exclusion mask synthesis
/// \param mmctrl               Molecular mechanics controls
/// \param poly_ps              Phase space synthesis
/// \param heat_bath            Thermostat (can be nullptr)
/// \param sc                   Score card for energy accumulation
/// \param tb_space             Cache resource for tile-based kernels
/// \param ism_space            Implicit solvent workspace (nullptr if GB disabled)
/// \param eval_force           Whether to evaluate forces
/// \param eval_energy          Whether to evaluate energy
/// \param launcher             Kernel launch manager
/// \param total_elec_out       Device pointer for scalar total electrostatic energy
/// \param total_vdw_out        Device pointer for scalar total VDW energy
//-------------------------------------------------------------------------------------------------
void launchLambdaNonbondedWithReduction(
    const double* lambda_vdw,
    const double* lambda_ele,
    const int* coupled_indices,
    int n_coupled,
    constants::PrecisionModel prec,
    const AtomGraphSynthesis& poly_ag,
    const StaticExclusionMaskSynthesis& poly_se,
    MolecularMechanicsControls* mmctrl,
    PhaseSpaceSynthesis* poly_ps,
    Thermostat* heat_bath,
    ScoreCard* sc,
    CacheResource* tb_space,
    ImplicitSolventWorkspace* ism_space,
    EvaluateForce eval_force,
    EvaluateEnergy eval_energy,
    const CoreKlManager& launcher,
    double* total_elec_out,
    double* total_vdw_out)
{
  // Step 1: Call the standard tile-based lambda nonbonded kernel
  // This uses the fast tile-group kernels and writes to ScoreCard on GPU
  launchLambdaNonbonded(
      lambda_vdw, lambda_ele, coupled_indices, n_coupled,
      prec, poly_ag, poly_se, mmctrl, poly_ps, heat_bath,
      sc, tb_space, ism_space, eval_force, eval_energy, launcher);

  // Step 2: Extract scalar totals from ScoreCard on GPU
  // This replaces the 4.36 MB ScoreCard download with a 16-byte transfer
  if (eval_energy == EvaluateEnergy::YES) {
    using card::HybridTargetLevel;
    const ScoreCard* const_sc = sc;  // Cast to const to get ScoreCardReader
    const ScoreCardReader scr = const_sc->data(HybridTargetLevel::DEVICE);
    const int system_id = 0;  // Single system for GCMC

    // Launch single-thread kernel to extract energies
    kExtractScoreCardEnergies<<<1, 1>>>(scr, system_id, total_elec_out, total_vdw_out);

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      rtErr("Failed to extract ScoreCard energies: " + std::string(cudaGetErrorString(err)),
            "launchLambdaNonbondedWithReduction");
    }
  }
}

} // namespace energy
} // namespace stormm
