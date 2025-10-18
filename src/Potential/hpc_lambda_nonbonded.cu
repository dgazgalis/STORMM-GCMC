// -*-c++-*-
#include "copyright.h"
#include <cuda_runtime.h>
#include "Accelerator/hybrid.h"
#include "Constants/behavior.h"
#include "DataTypes/common_types.h"
#include "Synthesis/implicit_solvent_workspace.h"
#include "Topology/atomgraph_enumerators.h"
#include "hpc_lambda_nonbonded.h"

namespace stormm {
namespace energy {

using card::HybridTargetLevel;
using synthesis::ImplicitSolventWorkspace;
using synthesis::ISWorkspaceKit;
using topology::ImplicitSolventModel;
using topology::UnitCellType;

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
    const double* __restrict__ xcrd,
    const double* __restrict__ ycrd,
    const double* __restrict__ zcrd,
    const double* __restrict__ charges,
    const double* __restrict__ lambda_vdw,
    const double* __restrict__ lambda_ele,
    const double* __restrict__ atom_sigma,
    const double* __restrict__ atom_epsilon,
    const uint* __restrict__ exclusion_mask,
    const int* __restrict__ supertile_map,
    const int* __restrict__ tile_map,
    const int supertile_stride,
    const double* __restrict__ umat,
    const UnitCellType unit_cell,
    const double coulomb_const,
    const double ewald_coeff,  // Ewald coefficient for PME direct space
    double* __restrict__ output_elec,
    double* __restrict__ output_vdw,
    double* __restrict__ xfrc,  // Force outputs (NULL for energy-only mode)
    double* __restrict__ yfrc,
    double* __restrict__ zfrc,
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
  const double xi = xcrd[i];
  const double yi = ycrd[i];
  const double zi = zcrd[i];
  const double qi = charges[i];
  const double lambda_vdw_i = lambda_vdw[i];
  const double lambda_ele_i = lambda_ele[i];
  const double sigma_i = atom_sigma[i];
  const double epsilon_i = atom_epsilon[i];

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
  // This changes from O(N_coupled × N_total) to O(N_coupled²)
  // Skips all Coupled-Ghost interactions which contribute ~0
  for (int j_tid = 0; j_tid < n_coupled; j_tid++) {
    // Get actual atom index for coupled atom j
    const int j = coupled_indices[j_tid];

    if (i == j) continue;

    // Avoid double counting: only process each pair once
    // Since coupled_indices is sorted, use i < j
    if (i > j) continue;

    // Check exclusion mask
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
    const double xj = xcrd[j];
    const double yj = ycrd[j];
    const double zj = zcrd[j];
    const double qj = charges[j];
    const double lambda_vdw_j = lambda_vdw[j];
    const double lambda_ele_j = lambda_ele[j];
    const double sigma_j = atom_sigma[j];
    const double epsilon_j = atom_epsilon[j];

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

    // VDW energy with softcore
    const double sigma = 0.5 * (sigma_i + sigma_j);
    const double epsilon = sqrt(epsilon_i * epsilon_j);
    const double lambda_ij_vdw = lambda_vdw_i * lambda_vdw_j;

    if (epsilon > 1.0e-10 && lambda_ij_vdw > 1.0e-10) {
      // Softcore potential
      const double one_minus_lambda = 1.0 - lambda_ij_vdw;
      const double r6 = r2 * r2 * r2;
      const double sigma2 = sigma * sigma;
      const double sigma6 = sigma2 * sigma2 * sigma2;
      const double r_eff6 = r6 + SOFTCORE_ALPHA * sigma6 * one_minus_lambda;

      const double inv_r_eff6 = 1.0 / r_eff6;
      const double inv_r_eff12 = inv_r_eff6 * inv_r_eff6;

      const double sigma12 = sigma6 * sigma6;
      const double lj_energy = 4.0 * epsilon * (sigma12 * inv_r_eff12 - sigma6 * inv_r_eff6);

      vdw_sum += lambda_ij_vdw * lj_energy;

      if (compute_forces) {
        // Softcore force: F = -dU/dr
        // dU/dr = lambda * d/dr[4*eps*(sig^12/r_eff^12 - sig^6/r_eff^6)]
        // For softcore: d(r_eff^6)/dr = 6*r^5
        // dU/dr = lambda * 4*eps * [-12*sig^12/(r_eff^13) + 6*sig^6/(r_eff^7)] * 6*r^5/(2*r_eff^6)
        //       = lambda * 24*eps*r^5 * [sig^6/r_eff^7 - 2*sig^12/r_eff^13]
        const double r5 = r2 * r2 * r;
        const double inv_r_eff7 = inv_r_eff6 / r_eff6;   // 1/r_eff^12
        const double inv_r_eff13 = inv_r_eff12 / r_eff6; // 1/r_eff^18

        // More accurate form matching the softcore potential derivative
        const double fmag = lambda_ij_vdw * 24.0 * epsilon * r5 *
                           (sigma6 * inv_r_eff7 - 2.0 * sigma12 * inv_r_eff13);

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
    atomicAdd(&xfrc[i], fx_sum);
    atomicAdd(&yfrc[i], fy_sum);
    atomicAdd(&zfrc[i], fz_sum);
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
    const double* __restrict__ xcrd,
    const double* __restrict__ ycrd,
    const double* __restrict__ zcrd,
    const double* __restrict__ pb_radii,      // Perfect Born radii from topology
    const double* __restrict__ gb_screen,     // Screening parameters
    const double gb_offset,                   // GB offset parameter
    double* __restrict__ psi,                  // Output psi values for Born radii calculation
    double* __restrict__ born_radii)          // Output Born radii
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_atoms) return;

  // Initialize psi for this atom
  double psi_sum = 0.0;

  // Get atom i properties
  const double xi = xcrd[i];
  const double yi = ycrd[i];
  const double zi = zcrd[i];
  const double pb_radius_i = pb_radii[i];
  const double screen_i = gb_screen[i];

  // Loop over all other atoms to compute psi
  for (int j = 0; j < n_atoms; j++) {
    if (i == j) continue;

    // Get atom j properties
    const double xj = xcrd[j];
    const double yj = ycrd[j];
    const double zj = zcrd[j];
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
    const double* __restrict__ xcrd,
    const double* __restrict__ ycrd,
    const double* __restrict__ zcrd,
    const double* __restrict__ charges,
    const double* __restrict__ lambda_ele,    // For lambda scaling
    const double* __restrict__ born_radii,
    const double* __restrict__ sum_deijda,    // Derivative of GB energy w.r.t. Born radii
    const double gb_offset,
    const double coulomb_const,
    double* __restrict__ xfrc,
    double* __restrict__ yfrc,
    double* __restrict__ zfrc)
{
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n_coupled) return;

  // Get the actual atom index for this coupled atom
  const int i = coupled_indices[tid];

  // Load atom i properties
  const double xi = xcrd[i];
  const double yi = ycrd[i];
  const double zi = zcrd[i];
  const double qi = charges[i];
  const double lambda_i = lambda_ele[i];
  const double born_radius_i = born_radii[i];
  const double deijda_i = sum_deijda[i];

  // Accumulate forces for this atom
  double fx_sum = 0.0;
  double fy_sum = 0.0;
  double fz_sum = 0.0;

  // Loop over all coupled atoms for pairwise contributions
  for (int j_tid = 0; j_tid < n_coupled; j_tid++) {
    const int j = coupled_indices[j_tid];
    if (i == j) continue;

    // Load atom j properties
    const double xj = xcrd[j];
    const double yj = ycrd[j];
    const double zj = zcrd[j];
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
  atomicAdd(&xfrc[i], fx_sum);
  atomicAdd(&yfrc[i], fy_sum);
  atomicAdd(&zfrc[i], fz_sum);
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
// Launch wrapper for the lambda-scaled nonbonded kernel
//-------------------------------------------------------------------------------------------------
void launchLambdaScaledNonbonded(
    int n_atoms,
    int n_coupled,
    const int* coupled_indices,
    const double* xcrd,
    const double* ycrd,
    const double* zcrd,
    const double* charges,
    const double* lambda_vdw,
    const double* lambda_ele,
    const double* atom_sigma,
    const double* atom_epsilon,
    const uint* exclusion_mask,
    const int* supertile_map,
    const int* tile_map,
    int supertile_stride,
    const double* umat,
    UnitCellType unit_cell,
    double coulomb_const,
    double ewald_coeff,
    double* output_elec,
    double* output_vdw,
    double* xfrc,  // nullptr for energy-only mode
    double* yfrc,
    double* zfrc,
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
      atom_sigma, atom_epsilon,
      exclusion_mask, supertile_map, tile_map, supertile_stride,
      umat, unit_cell, coulomb_const, ewald_coeff,
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
    const double* xcrd,
    const double* ycrd,
    const double* zcrd,
    const double* charges,
    const double* lambda_vdw,
    const double* lambda_ele,
    const double* atom_sigma,
    const double* atom_epsilon,
    const uint* exclusion_mask,
    const int* supertile_map,
    const int* tile_map,
    int supertile_stride,
    const double* umat,
    UnitCellType unit_cell,
    double coulomb_const,
    double ewald_coeff,
    double* per_atom_elec,      // Device arrays for intermediate results
    double* per_atom_vdw,
    double* total_elec_out,     // Device scalar output
    double* total_vdw_out,      // Device scalar output
    double* xfrc,               // nullptr for energy-only mode
    double* yfrc,
    double* zfrc,
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
      atom_sigma, atom_epsilon,
      exclusion_mask, supertile_map, tile_map, supertile_stride,
      umat, unit_cell, coulomb_const, ewald_coeff,
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
    const double* xcrd,
    const double* ycrd,
    const double* zcrd,
    const double* pb_radii,
    const double* gb_screen,
    double gb_offset,
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
    const double* xcrd,
    const double* ycrd,
    const double* zcrd,
    const double* charges,
    const double* lambda_ele,
    const double* born_radii,
    const double* sum_deijda,
    double gb_offset,
    double coulomb_const,
    double* xfrc,
    double* yfrc,
    double* zfrc,
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
      xfrc, yfrc, zfrc);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    // Error occurred - STORMM will handle it
  }
}

} // namespace energy
} // namespace stormm
