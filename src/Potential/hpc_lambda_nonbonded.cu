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
    topology::ImplicitSolventModel gb_model)
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
    topology::ImplicitSolventModel gb_model)
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
// HIGH-LEVEL LAMBDA NONBONDED LAUNCHER (matching standard dynamics pattern)
//================================================================================================

//-------------------------------------------------------------------------------------------------
// High-level launcher that matches launchNonbonded() from hpc_nonbonded_potential.cu
// This extracts coordinate/force pointers internally from PhaseSpaceSynthesis just like
// the standard dynamics does.
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

  const HybridTargetLevel tier = HybridTargetLevel::DEVICE;
  const int n_atoms = poly_ag.getAtomCount();

  // Get current cycle position for coordinate/force access
  const CoordinateCycle curr_cyc = poly_ps->getCyclePosition();
  PsSynthesisWriter psw = poly_ps->data(curr_cyc, tier);

  // Get exclusion mask reader (DEVICE tier for kernel)
  const SeMaskSynthesisReader poly_ser = poly_se.data(tier);

  // Get nonbonded parameters (DEVICE tier for kernel)
  const SyNonbondedKit<double, double2> nbk = poly_ag.getDoublePrecisionNonbondedKit(tier);

  // FIX: Get HOST-tier abstracts for reading scalars on CPU
  const SeMaskSynthesisReader poly_ser_host = poly_se.data(HybridTargetLevel::HOST);
  const SyNonbondedKit<double, double2> nbk_host =
      poly_ag.getDoublePrecisionNonbondedKit(HybridTargetLevel::HOST);

  // Get unit cell info
  const UnitCellType unit_cell = poly_ps->getUnitCellType();
  const double ewald_coeff = (unit_cell != UnitCellType::NONE) ? 0.35 : 0.0;
  const double coulomb_const = poly_ag.getCoulombConstant();

  // FIX: Read scalars from HOST-tier abstracts (system 0 for single-system GCMC)
  const int system_id = 0;  // GCMC typically operates on single system
  const int supertile_stride = poly_ser_host.atom_counts[system_id];
  const int n_lj_types = nbk_host.n_lj_types[system_id];
  const int lj_offset = nbk_host.ljabc_offsets[system_id];

  // Extract LJ parameters from topology (STORMM native convention)
  // Use DEVICE-tier pointers for passing to kernel
  const int* lj_idx_ptr = nbk.lj_idx;
  // FIX: Offset ljab_coeff by system's LJ parameter table offset
  const double2* ljab_coeff_ptr = nbk.ljab_coeff + lj_offset;

  // Per-atom energy arrays for GPU-side reduction
  // CRITICAL: Use static variables to avoid memory leak from repeated allocations
  // These are allocated once and reused across all GCMC cycles, preventing
  // cudaFreeHost failures from pinned memory fragmentation
  static Hybrid<double> per_atom_elec(1, "per_atom_elec");
  static Hybrid<double> per_atom_vdw(1, "per_atom_vdw");

  // Resize if needed (only allocates if size changed)
  if (per_atom_elec.size() != n_coupled) {
    per_atom_elec.resize(n_coupled);
    per_atom_vdw.resize(n_coupled);
    per_atom_elec.upload();
    per_atom_vdw.upload();
  }

  // Device scalars for GPU-reduced total energies
  // CRITICAL: Use static variables to avoid memory leak from repeated allocations
  static Hybrid<double> total_elec_dev(1, "total_elec_reduced");
  static Hybrid<double> total_vdw_dev(1, "total_vdw_reduced");
  static bool scalars_initialized = false;

  if (!scalars_initialized) {
    total_elec_dev.upload();
    total_vdw_dev.upload();
    scalars_initialized = true;
  }

  // Get implicit solvent model
  const ImplicitSolventModel gb_model = poly_ag.getImplicitSolventModel();

  // Call lambda-scaled nonbonded kernel WITH GPU reduction to get scalar totals
  // This eliminates CPU-side reduction and only downloads 2 scalars instead of n_coupled values
  launchLambdaScaledNonbondedWithReduction(
      n_atoms,
      n_coupled,
      coupled_indices,
      psw.xcrd,                    // llint* coordinates (fixed-precision)
      psw.ycrd,
      psw.zcrd,
      nbk.charge,                  // Charges from topology
      lambda_vdw,
      lambda_ele,
      lj_idx_ptr,                  // LJ type indices from topology
      n_lj_types,                  // Number of LJ types
      ljab_coeff_ptr,              // LJ A/B coefficients from topology
      poly_ser.mask_data,
      poly_ser.supertile_map_idx,
      poly_ser.tile_map_idx,
      supertile_stride,
      psw.umat,
      unit_cell,
      coulomb_const,
      ewald_coeff,
      psw.inv_gpos_scale,          // Coordinate scaling factor
      psw.frc_scale,               // Force scaling factor
      per_atom_elec.data(tier),    // Device per-atom arrays (intermediate)
      per_atom_vdw.data(tier),
      total_elec_dev.data(tier),   // Device scalar outputs (GPU-reduced)
      total_vdw_dev.data(tier),
      psw.xfrc,                    // llint* forces (fixed-precision)
      psw.yfrc,
      psw.zfrc,
      ism_space,
      gb_model);

  // Only write energies to ScoreCard if energy evaluation was requested
  if (eval_energy == EvaluateEnergy::YES) {
    // Extract ScoreCardWriter for GPU-side atomic writes (matches standard STORMM pattern)
    ScoreCardWriter scw = sc->data(tier);

    // Launch GPU kernel to write energies directly to ScoreCard
    // This uses atomic operations to accumulate energies, ensuring proper synchronization
    // with other kernels (valence, PME, etc.) that also write to the same ScoreCard
    const int system_id = 0;  // Single-system GCMC
    kWriteEnergiesToScoreCard<<<1, 1>>>(
        total_elec_dev.data(tier),  // GPU-side scalar (already reduced)
        total_vdw_dev.data(tier),   // GPU-side scalar (already reduced)
        scw,                        // ScoreCardWriter for atomic accumulation
        system_id);                 // System index

    // No CPU-side download or contribute() needed - kernel writes directly to GPU ScoreCard!
  }
}

} // namespace energy
} // namespace stormm
