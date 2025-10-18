#include <algorithm>
#include <cmath>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include "copyright.h"
#include "Accelerator/hybrid.h"
#include "Math/vector_ops.h"
#include "Reporting/error_format.h"
#include "Trajectory/trajectory_enumerators.h"
#include "mc_mover.h"
#include "gcmc_sampler.h"
#ifdef STORMM_USE_HPC
#  include "hpc_mc_moves.h"
#endif

namespace stormm {
namespace sampling {

using card::Hybrid;
using card::HybridTargetLevel;
using stmath::crossProduct;
using stmath::normalize;
using symbols::pi;
using trajectory::CartesianDimension;

//-------------------------------------------------------------------------------------------------
// MCMoveStatistics implementation
//-------------------------------------------------------------------------------------------------
MCMoveStatistics::MCMoveStatistics() :
  n_attempted{0},
  n_accepted{0},
  n_rejected{0},
  total_energy_change{0.0}
{}

//-------------------------------------------------------------------------------------------------
double MCMoveStatistics::getAcceptanceRate() const {
  if (n_attempted == 0) return 0.0;
  return static_cast<double>(n_accepted) / static_cast<double>(n_attempted);
}

//-------------------------------------------------------------------------------------------------
void MCMoveStatistics::reset() {
  n_attempted = 0;
  n_accepted = 0;
  n_rejected = 0;
  total_energy_change = 0.0;
}

//-------------------------------------------------------------------------------------------------
// MCMover base class implementation
//-------------------------------------------------------------------------------------------------
MCMover::MCMover(GCMCSampler* sampler, double beta, Xoshiro256ppGenerator* rng) :
  sampler_{sampler},
  beta_{beta},
  rng_{rng},
  stats_{}
{}

//-------------------------------------------------------------------------------------------------
bool MCMover::acceptMove(double delta_E) {
  // Metropolis acceptance criterion
  if (delta_E <= 0.0) {
    return true;  // Always accept moves that lower energy
  }

  const double acceptance_prob = std::exp(-beta_ * delta_E);
  const double random_num = rng_->uniformRandomNumber();
  return random_num < acceptance_prob;
}

//-------------------------------------------------------------------------------------------------
// TranslationMover implementation
//-------------------------------------------------------------------------------------------------
TranslationMover::TranslationMover(GCMCSampler* sampler, double beta,
                                   Xoshiro256ppGenerator* rng, double max_displacement) :
  MCMover(sampler, beta, rng),
  max_displacement_{max_displacement}
{}

//-------------------------------------------------------------------------------------------------
bool TranslationMover::attemptMove(GCMCMolecule& mol) {
  stats_.n_attempted++;

  // Skip ghost molecules (lambda = 0)
  if (mol.isGhost()) {
    stats_.n_rejected++;
    return false;
  }

  // Validate molecule has atoms
  if (mol.atom_indices.empty()) {
    stats_.n_rejected++;
    return false;
  }

  PhaseSpace* ps = sampler_->getPhaseSpace();

  // Evaluate energy before move
  const double E_old = sampler_->evaluateTotalEnergy();

  // Generate random displacement (on CPU - trivial computation)
  const double dx = (rng_->uniformRandomNumber() - 0.5) * 2.0 * max_displacement_;
  const double dy = (rng_->uniformRandomNumber() - 0.5) * 2.0 * max_displacement_;
  const double dz = (rng_->uniformRandomNumber() - 0.5) * 2.0 * max_displacement_;

#ifdef STORMM_USE_HPC
  // GPU path - no coordinate download/upload!
  PhaseSpaceWriter psw = ps->data();

  // Defensive check: ensure workspace is large enough
  if (mol.atom_indices.size() > sampler_->getMCSavedX().size()) {
    rtErr("MC move workspace undersized: need " +
          std::to_string(mol.atom_indices.size()) + " but have " +
          std::to_string(sampler_->getMCSavedX().size()),
          "TranslationMover::attemptMove");
  }

  // 1. Prepare atom indices on GPU
  Hybrid<int>& atom_indices = sampler_->getMCAtomIndices();
  atom_indices.resize(mol.atom_indices.size());
  for (size_t i = 0; i < mol.atom_indices.size(); i++) {
    atom_indices.data()[i] = mol.atom_indices[i];
  }
  atom_indices.upload();

  // 2. Backup coordinates on GPU
  Hybrid<double>& saved_x = sampler_->getMCSavedX();
  Hybrid<double>& saved_y = sampler_->getMCSavedY();
  Hybrid<double>& saved_z = sampler_->getMCSavedZ();
  saved_x.resize(mol.atom_indices.size());
  saved_y.resize(mol.atom_indices.size());
  saved_z.resize(mol.atom_indices.size());

  launchBackupCoordinates(
      mol.atom_indices.size(),
      atom_indices.data(HybridTargetLevel::DEVICE),
      psw.xcrd, psw.ycrd, psw.zcrd,
      saved_x.data(HybridTargetLevel::DEVICE),
      saved_y.data(HybridTargetLevel::DEVICE),
      saved_z.data(HybridTargetLevel::DEVICE));

  // 3. Apply translation on GPU
  launchTranslateMolecule(
      mol.atom_indices.size(),
      atom_indices.data(HybridTargetLevel::DEVICE),
      dx, dy, dz,
      psw.xcrd, psw.ycrd, psw.zcrd);

  // 4. Apply PBC (GPU-aware if implemented, otherwise CPU fallback)
  sampler_->applyPBC(mol);

  // 5. Invalidate energy cache and evaluate energy (already on GPU!)
  sampler_->invalidateEnergyCache();
  const double E_new = sampler_->evaluateTotalEnergy();
  const double delta_E = E_new - E_old;

  // 6. Accept/reject based on Metropolis criterion
  if (acceptMove(delta_E)) {
    stats_.n_accepted++;
    stats_.total_energy_change += delta_E;
    return true;
  } else {
    // Restore coordinates on GPU
    launchRestoreCoordinates(
        mol.atom_indices.size(),
        atom_indices.data(HybridTargetLevel::DEVICE),
        saved_x.data(HybridTargetLevel::DEVICE),
        saved_y.data(HybridTargetLevel::DEVICE),
        saved_z.data(HybridTargetLevel::DEVICE),
        psw.xcrd, psw.ycrd, psw.zcrd);

    stats_.n_rejected++;
    return false;
  }
#else
  // CPU fallback - original implementation
  auto saved_coords = sampler_->saveCoordinates(mol);

  PhaseSpaceWriter psw = ps->data();
  double* xcrd = psw.xcrd;
  double* ycrd = psw.ycrd;
  double* zcrd = psw.zcrd;

  for (int atom_idx : mol.atom_indices) {
    xcrd[atom_idx] += dx;
    ycrd[atom_idx] += dy;
    zcrd[atom_idx] += dz;
  }

  sampler_->applyPBC(mol);
  sampler_->invalidateEnergyCache();

  const double E_new = sampler_->evaluateTotalEnergy();
  const double delta_E = E_new - E_old;

  if (acceptMove(delta_E)) {
    stats_.n_accepted++;
    stats_.total_energy_change += delta_E;
    return true;
  } else {
    sampler_->restoreCoordinates(mol, saved_coords);
    stats_.n_rejected++;
    return false;
  }
#endif
}

//-------------------------------------------------------------------------------------------------
// RotationMover implementation
//-------------------------------------------------------------------------------------------------
RotationMover::RotationMover(GCMCSampler* sampler, double beta,
                             Xoshiro256ppGenerator* rng, double max_angle) :
  MCMover(sampler, beta, rng),
  max_angle_{max_angle}
{}

//-------------------------------------------------------------------------------------------------
std::vector<double> RotationMover::generateRandomQuaternion(bool use_limited) {
  std::vector<double> quat(4);

  if (use_limited && max_angle_ > 0.0) {
    // Limited rotation: generate quaternion for rotation up to max_angle
    // Random axis (uniform on sphere)
    const double theta = rng_->uniformRandomNumber() * 2.0 * pi;
    const double phi = std::acos(2.0 * rng_->uniformRandomNumber() - 1.0);
    const double axis_x = std::sin(phi) * std::cos(theta);
    const double axis_y = std::sin(phi) * std::sin(theta);
    const double axis_z = std::cos(phi);

    // Random angle up to max_angle
    const double angle = (rng_->uniformRandomNumber() - 0.5) * 2.0 * max_angle_;
    const double half_angle = angle * 0.5;
    const double sin_half = std::sin(half_angle);

    quat[0] = std::cos(half_angle);  // w
    quat[1] = axis_x * sin_half;     // x
    quat[2] = axis_y * sin_half;     // y
    quat[3] = axis_z * sin_half;     // z
  } else {
    // Uniform random rotation using Shoemake's method
    const double u1 = rng_->uniformRandomNumber();
    const double u2 = rng_->uniformRandomNumber();
    const double u3 = rng_->uniformRandomNumber();

    const double sqrt1_u1 = std::sqrt(1.0 - u1);
    const double sqrtu1 = std::sqrt(u1);
    const double two_pi_u2 = 2.0 * pi * u2;
    const double two_pi_u3 = 2.0 * pi * u3;

    quat[0] = sqrt1_u1 * std::sin(two_pi_u2);  // w
    quat[1] = sqrt1_u1 * std::cos(two_pi_u2);  // x
    quat[2] = sqrtu1 * std::sin(two_pi_u3);    // y
    quat[3] = sqrtu1 * std::cos(two_pi_u3);    // z
  }

  return quat;
}

//-------------------------------------------------------------------------------------------------
double3 RotationMover::rotatePoint(const double3& point, const std::vector<double>& quat,
                                   const double3& center) {
  // Translate point to origin
  double3 p;
  p.x = point.x - center.x;
  p.y = point.y - center.y;
  p.z = point.z - center.z;

  // Apply quaternion rotation
  const double qw = quat[0];
  const double qx = quat[1];
  const double qy = quat[2];
  const double qz = quat[3];

  // Rotation matrix from quaternion
  const double xx = qx * qx;
  const double yy = qy * qy;
  const double zz = qz * qz;
  const double xy = qx * qy;
  const double xz = qx * qz;
  const double yz = qy * qz;
  const double xw = qx * qw;
  const double yw = qy * qw;
  const double zw = qz * qw;

  double3 rotated;
  rotated.x = p.x * (1.0 - 2.0 * (yy + zz)) +
              p.y * 2.0 * (xy - zw) +
              p.z * 2.0 * (xz + yw);
  rotated.y = p.x * 2.0 * (xy + zw) +
              p.y * (1.0 - 2.0 * (xx + zz)) +
              p.z * 2.0 * (yz - xw);
  rotated.z = p.x * 2.0 * (xz - yw) +
              p.y * 2.0 * (yz + xw) +
              p.z * (1.0 - 2.0 * (xx + yy));

  // Translate back
  rotated.x += center.x;
  rotated.y += center.y;
  rotated.z += center.z;

  return rotated;
}

//-------------------------------------------------------------------------------------------------
bool RotationMover::attemptMove(GCMCMolecule& mol) {
  stats_.n_attempted++;

  // Skip ghost molecules
  if (mol.isGhost()) {
    stats_.n_rejected++;
    return false;
  }

  // Validate molecule has at least 2 atoms for rotation
  if (mol.atom_indices.size() < 2) {
    stats_.n_rejected++;
    return false;
  }

  PhaseSpace* ps = sampler_->getPhaseSpace();

  // Evaluate energy before move
  const double E_old = sampler_->evaluateTotalEnergy();

  // Calculate center of geometry
  const double3 cog = sampler_->calculateMoleculeCOG(mol);

  // Generate random rotation quaternion (on CPU - trivial computation)
  const bool use_limited = (max_angle_ > 0.0);
  std::vector<double> quat = generateRandomQuaternion(use_limited);

  // Convert quaternion to rotation matrix (CPU)
  const double qw = quat[0];
  const double qx = quat[1];
  const double qy = quat[2];
  const double qz = quat[3];

  const double xx = qx * qx;
  const double yy = qy * qy;
  const double zz = qz * qz;
  const double xy = qx * qy;
  const double xz = qx * qz;
  const double yz = qy * qz;
  const double xw = qx * qw;
  const double yw = qy * qw;
  const double zw = qz * qw;

  double rot_matrix[9];
  rot_matrix[0] = 1.0 - 2.0 * (yy + zz);
  rot_matrix[1] = 2.0 * (xy - zw);
  rot_matrix[2] = 2.0 * (xz + yw);
  rot_matrix[3] = 2.0 * (xy + zw);
  rot_matrix[4] = 1.0 - 2.0 * (xx + zz);
  rot_matrix[5] = 2.0 * (yz - xw);
  rot_matrix[6] = 2.0 * (xz - yw);
  rot_matrix[7] = 2.0 * (yz + xw);
  rot_matrix[8] = 1.0 - 2.0 * (xx + yy);

#ifdef STORMM_USE_HPC
  // GPU path - no coordinate download/upload!
  PhaseSpaceWriter psw = ps->data();

  // Defensive check: ensure workspace is large enough
  if (mol.atom_indices.size() > sampler_->getMCSavedX().size()) {
    rtErr("MC move workspace undersized: need " +
          std::to_string(mol.atom_indices.size()) + " but have " +
          std::to_string(sampler_->getMCSavedX().size()),
          "RotationMover::attemptMove");
  }

  // 1. Prepare atom indices on GPU
  Hybrid<int>& atom_indices = sampler_->getMCAtomIndices();
  atom_indices.resize(mol.atom_indices.size());
  for (size_t i = 0; i < mol.atom_indices.size(); i++) {
    atom_indices.data()[i] = mol.atom_indices[i];
  }
  atom_indices.upload();

  // 2. Upload rotation matrix to GPU
  Hybrid<double>& gpu_rot_matrix = sampler_->getMCRotationMatrix();
  gpu_rot_matrix.resize(9);
  for (int i = 0; i < 9; i++) {
    gpu_rot_matrix.data()[i] = rot_matrix[i];
  }
  gpu_rot_matrix.upload();

  // 3. Backup coordinates on GPU
  Hybrid<double>& saved_x = sampler_->getMCSavedX();
  Hybrid<double>& saved_y = sampler_->getMCSavedY();
  Hybrid<double>& saved_z = sampler_->getMCSavedZ();
  saved_x.resize(mol.atom_indices.size());
  saved_y.resize(mol.atom_indices.size());
  saved_z.resize(mol.atom_indices.size());

  launchBackupCoordinates(
      mol.atom_indices.size(),
      atom_indices.data(HybridTargetLevel::DEVICE),
      psw.xcrd, psw.ycrd, psw.zcrd,
      saved_x.data(HybridTargetLevel::DEVICE),
      saved_y.data(HybridTargetLevel::DEVICE),
      saved_z.data(HybridTargetLevel::DEVICE));

  // 4. Apply rotation on GPU
  launchRotateMolecule(
      mol.atom_indices.size(),
      atom_indices.data(HybridTargetLevel::DEVICE),
      cog.x, cog.y, cog.z,
      gpu_rot_matrix.data(HybridTargetLevel::DEVICE),
      psw.xcrd, psw.ycrd, psw.zcrd);

  // 5. Apply PBC
  sampler_->applyPBC(mol);

  // 6. Invalidate energy cache and evaluate energy (already on GPU!)
  sampler_->invalidateEnergyCache();
  const double E_new = sampler_->evaluateTotalEnergy();
  const double delta_E = E_new - E_old;

  // 7. Accept/reject
  if (acceptMove(delta_E)) {
    stats_.n_accepted++;
    stats_.total_energy_change += delta_E;
    return true;
  } else {
    // Restore coordinates on GPU
    launchRestoreCoordinates(
        mol.atom_indices.size(),
        atom_indices.data(HybridTargetLevel::DEVICE),
        saved_x.data(HybridTargetLevel::DEVICE),
        saved_y.data(HybridTargetLevel::DEVICE),
        saved_z.data(HybridTargetLevel::DEVICE),
        psw.xcrd, psw.ycrd, psw.zcrd);

    stats_.n_rejected++;
    return false;
  }
#else
  // CPU fallback - original implementation
  auto saved_coords = sampler_->saveCoordinates(mol);

  PhaseSpaceWriter psw = ps->data();
  double* xcrd = psw.xcrd;
  double* ycrd = psw.ycrd;
  double* zcrd = psw.zcrd;

  for (int atom_idx : mol.atom_indices) {
    double3 current;
    current.x = xcrd[atom_idx];
    current.y = ycrd[atom_idx];
    current.z = zcrd[atom_idx];

    double3 rotated = rotatePoint(current, quat, cog);

    xcrd[atom_idx] = rotated.x;
    ycrd[atom_idx] = rotated.y;
    zcrd[atom_idx] = rotated.z;
  }

  sampler_->applyPBC(mol);
  sampler_->invalidateEnergyCache();

  const double E_new = sampler_->evaluateTotalEnergy();
  const double delta_E = E_new - E_old;

  if (acceptMove(delta_E)) {
    stats_.n_accepted++;
    stats_.total_energy_change += delta_E;
    return true;
  } else {
    sampler_->restoreCoordinates(mol, saved_coords);
    stats_.n_rejected++;
    return false;
  }
#endif
}

//-------------------------------------------------------------------------------------------------
// TorsionMover implementation
//-------------------------------------------------------------------------------------------------
TorsionMover::TorsionMover(GCMCSampler* sampler, double beta,
                           Xoshiro256ppGenerator* rng,
                           const AtomGraph* topology, double max_angle) :
  MCMover(sampler, beta, rng),
  topology_{topology},
  max_angle_{max_angle},
  rotatable_bonds_cache_{}
{
  if (topology == nullptr) {
    rtErr("TorsionMover requires non-null topology pointer", "TorsionMover");
  }
}

//-------------------------------------------------------------------------------------------------
bool TorsionMover::isBondRotatable(int atom1, int atom2) const {
  // Terminal bonds (like methyl groups) are typically rotatable
  // but we may want to exclude them for efficiency

  // Check if either atom is hydrogen - skip H-X bonds
  const double mass1 = topology_->getAtomicMass<double>(atom1);
  const double mass2 = topology_->getAtomicMass<double>(atom2);
  if (mass1 < 1.5 || mass2 < 1.5) {
    return false;
  }

  // Check if bond is in a ring - these are generally not rotatable
  // (Would require ring detection algorithm - simplified for now)

  // For now, consider all non-H bonds rotatable
  return true;
}

//-------------------------------------------------------------------------------------------------
std::vector<RotatableBond> TorsionMover::identifyRotatableBonds(const GCMCMolecule& mol) {
  std::vector<RotatableBond> rotatable_bonds;

  // Check cache first using residue ID as key
  const int cache_key = mol.resid;
  auto it = rotatable_bonds_cache_.find(cache_key);
  if (it != rotatable_bonds_cache_.end()) {
    return it->second;
  }

  // Build set of molecule atoms for fast lookup
  std::unordered_set<int> mol_atoms(mol.atom_indices.begin(), mol.atom_indices.end());

  // Get bond information from topology using ValenceKit
  const topology::ValenceKit<double> vk = topology_->getDoublePrecisionValenceKit();
  const int n_bonds = vk.nbond;

  // Find bonds within the molecule
  for (int i = 0; i < n_bonds; i++) {
    int atom1 = vk.bond_i_atoms[i];
    int atom2 = vk.bond_j_atoms[i];

    // Check if both atoms are in the molecule
    if (mol_atoms.count(atom1) && mol_atoms.count(atom2)) {
      // Check if bond is rotatable
      if (isBondRotatable(atom1, atom2)) {
        RotatableBond bond;
        bond.atom1 = atom1;
        bond.atom2 = atom2;

        // Identify which atoms rotate with this bond using BFS graph traversal
        // Start from atom2, excluding paths through atom1
        bond.rotating_atoms.clear();

        std::unordered_set<int> visited;
        std::queue<int> to_visit;

        to_visit.push(atom2);
        visited.insert(atom2);
        visited.insert(atom1);  // Block traversal through atom1

        while (!to_visit.empty()) {
          int current = to_visit.front();
          to_visit.pop();

          // Find bonded neighbors from ValenceKit
          for (int j = 0; j < n_bonds; j++) {
            int neighbor = -1;
            if (vk.bond_i_atoms[j] == current) {
              neighbor = vk.bond_j_atoms[j];
            } else if (vk.bond_j_atoms[j] == current) {
              neighbor = vk.bond_i_atoms[j];
            }

            if (neighbor >= 0 && visited.find(neighbor) == visited.end()) {
              // Check if neighbor is in this molecule
              if (mol_atoms.count(neighbor)) {
                visited.insert(neighbor);
                to_visit.push(neighbor);
                bond.rotating_atoms.push_back(neighbor);
              }
            }
          }
        }

        // Only add if there are atoms to rotate
        if (!bond.rotating_atoms.empty()) {
          bond.is_terminal = (bond.rotating_atoms.size() <= 3);
          rotatable_bonds.push_back(bond);
        }
      }
    }
  }

  // Cache the result using residue ID as key
  rotatable_bonds_cache_[cache_key] = rotatable_bonds;

  return rotatable_bonds;
}

//-------------------------------------------------------------------------------------------------
void TorsionMover::rotateBondedAtoms(const std::vector<int>& atom_indices,
                                     const double3& axis_start,
                                     const double3& axis_end,
                                     double angle,
                                     PhaseSpace* ps) {
  // Get coordinate arrays
  PhaseSpaceWriter psw = ps->data();
  double* xcrd = psw.xcrd;
  double* ycrd = psw.ycrd;
  double* zcrd = psw.zcrd;

  // Calculate rotation axis
  double3 axis;
  axis.x = axis_end.x - axis_start.x;
  axis.y = axis_end.y - axis_start.y;
  axis.z = axis_end.z - axis_start.z;

  // Normalize axis
  const double axis_length = std::sqrt(axis.x * axis.x + axis.y * axis.y + axis.z * axis.z);
  if (axis_length < 1e-6) return;  // Degenerate axis

  axis.x /= axis_length;
  axis.y /= axis_length;
  axis.z /= axis_length;

  // Rotation matrix using Rodrigues' formula
  const double cos_angle = std::cos(angle);
  const double sin_angle = std::sin(angle);
  const double one_minus_cos = 1.0 - cos_angle;

  // Build rotation matrix
  const double r11 = cos_angle + axis.x * axis.x * one_minus_cos;
  const double r12 = axis.x * axis.y * one_minus_cos - axis.z * sin_angle;
  const double r13 = axis.x * axis.z * one_minus_cos + axis.y * sin_angle;
  const double r21 = axis.y * axis.x * one_minus_cos + axis.z * sin_angle;
  const double r22 = cos_angle + axis.y * axis.y * one_minus_cos;
  const double r23 = axis.y * axis.z * one_minus_cos - axis.x * sin_angle;
  const double r31 = axis.z * axis.x * one_minus_cos - axis.y * sin_angle;
  const double r32 = axis.z * axis.y * one_minus_cos + axis.x * sin_angle;
  const double r33 = cos_angle + axis.z * axis.z * one_minus_cos;

  // Apply rotation to each atom
  for (int atom_idx : atom_indices) {
    // Translate to origin (axis_start)
    double px = xcrd[atom_idx] - axis_start.x;
    double py = ycrd[atom_idx] - axis_start.y;
    double pz = zcrd[atom_idx] - axis_start.z;

    // Apply rotation
    double new_x = r11 * px + r12 * py + r13 * pz;
    double new_y = r21 * px + r22 * py + r23 * pz;
    double new_z = r31 * px + r32 * py + r33 * pz;

    // Translate back
    xcrd[atom_idx] = new_x + axis_start.x;
    ycrd[atom_idx] = new_y + axis_start.y;
    zcrd[atom_idx] = new_z + axis_start.z;
  }
}

//-------------------------------------------------------------------------------------------------
bool TorsionMover::attemptMove(GCMCMolecule& mol) {
  stats_.n_attempted++;

  // Skip ghost molecules
  if (mol.isGhost()) {
    stats_.n_rejected++;
    return false;
  }

  // Validate molecule has at least 4 atoms for torsion
  if (mol.atom_indices.size() < 4) {
    stats_.n_rejected++;
    return false;
  }

  // Identify rotatable bonds
  std::vector<RotatableBond> rotatable_bonds = identifyRotatableBonds(mol);

  if (rotatable_bonds.empty()) {
    // No rotatable bonds - reject move
    stats_.n_rejected++;
    return false;
  }

  PhaseSpace* ps = sampler_->getPhaseSpace();

  // Evaluate energy before move
  const double E_old = sampler_->evaluateTotalEnergy();

  // Select random bond (CPU)
  const int bond_idx = static_cast<int>(rng_->uniformRandomNumber() * rotatable_bonds.size());
  const RotatableBond& bond = rotatable_bonds[bond_idx];

  // Generate random torsion angle change (CPU)
  const double angle_change = (rng_->uniformRandomNumber() - 0.5) * 2.0 * max_angle_;

  // Get bond axis points
  PhaseSpaceWriter psw = ps->data();

  double3 axis_start;
  axis_start.x = psw.xcrd[bond.atom1];
  axis_start.y = psw.ycrd[bond.atom1];
  axis_start.z = psw.zcrd[bond.atom1];

  double3 axis_end;
  axis_end.x = psw.xcrd[bond.atom2];
  axis_end.y = psw.ycrd[bond.atom2];
  axis_end.z = psw.zcrd[bond.atom2];

  // Calculate rotation axis and matrix (CPU - trivial computation)
  double3 axis;
  axis.x = axis_end.x - axis_start.x;
  axis.y = axis_end.y - axis_start.y;
  axis.z = axis_end.z - axis_start.z;

  const double axis_length = std::sqrt(axis.x * axis.x + axis.y * axis.y + axis.z * axis.z);
  if (axis_length < 1e-6) {
    // Degenerate axis - reject move
    stats_.n_rejected++;
    return false;
  }

  axis.x /= axis_length;
  axis.y /= axis_length;
  axis.z /= axis_length;

  // Build rotation matrix using Rodrigues' formula
  const double cos_angle = std::cos(angle_change);
  const double sin_angle = std::sin(angle_change);
  const double one_minus_cos = 1.0 - cos_angle;

  double rot_matrix[9];
  rot_matrix[0] = cos_angle + axis.x * axis.x * one_minus_cos;
  rot_matrix[1] = axis.x * axis.y * one_minus_cos - axis.z * sin_angle;
  rot_matrix[2] = axis.x * axis.z * one_minus_cos + axis.y * sin_angle;
  rot_matrix[3] = axis.y * axis.x * one_minus_cos + axis.z * sin_angle;
  rot_matrix[4] = cos_angle + axis.y * axis.y * one_minus_cos;
  rot_matrix[5] = axis.y * axis.z * one_minus_cos - axis.x * sin_angle;
  rot_matrix[6] = axis.z * axis.x * one_minus_cos - axis.y * sin_angle;
  rot_matrix[7] = axis.z * axis.y * one_minus_cos + axis.x * sin_angle;
  rot_matrix[8] = cos_angle + axis.z * axis.z * one_minus_cos;

#ifdef STORMM_USE_HPC
  // GPU path - no coordinate download/upload!

  // Defensive check: ensure workspace is large enough
  if (mol.atom_indices.size() > sampler_->getMCSavedX().size()) {
    rtErr("MC move workspace undersized: need " +
          std::to_string(mol.atom_indices.size()) + " but have " +
          std::to_string(sampler_->getMCSavedX().size()),
          "TorsionMover::attemptMove");
  }

  // 1. Prepare rotating atom indices on GPU
  Hybrid<int>& rotating_atoms = sampler_->getMCRotatingAtoms();
  rotating_atoms.resize(bond.rotating_atoms.size());
  for (size_t i = 0; i < bond.rotating_atoms.size(); i++) {
    rotating_atoms.data()[i] = bond.rotating_atoms[i];
  }
  rotating_atoms.upload();

  // 2. Upload rotation matrix to GPU
  Hybrid<double>& gpu_rot_matrix = sampler_->getMCRotationMatrix();
  gpu_rot_matrix.resize(9);
  for (int i = 0; i < 9; i++) {
    gpu_rot_matrix.data()[i] = rot_matrix[i];
  }
  gpu_rot_matrix.upload();

  // 3. Backup coordinates on GPU (backup all molecule atoms)
  Hybrid<int>& atom_indices = sampler_->getMCAtomIndices();
  atom_indices.resize(mol.atom_indices.size());
  for (size_t i = 0; i < mol.atom_indices.size(); i++) {
    atom_indices.data()[i] = mol.atom_indices[i];
  }
  atom_indices.upload();

  Hybrid<double>& saved_x = sampler_->getMCSavedX();
  Hybrid<double>& saved_y = sampler_->getMCSavedY();
  Hybrid<double>& saved_z = sampler_->getMCSavedZ();
  saved_x.resize(mol.atom_indices.size());
  saved_y.resize(mol.atom_indices.size());
  saved_z.resize(mol.atom_indices.size());

  launchBackupCoordinates(
      mol.atom_indices.size(),
      atom_indices.data(HybridTargetLevel::DEVICE),
      psw.xcrd, psw.ycrd, psw.zcrd,
      saved_x.data(HybridTargetLevel::DEVICE),
      saved_y.data(HybridTargetLevel::DEVICE),
      saved_z.data(HybridTargetLevel::DEVICE));

  // 4. Apply torsion rotation on GPU
  launchRotateTorsion(
      bond.rotating_atoms.size(),
      rotating_atoms.data(HybridTargetLevel::DEVICE),
      axis_start.x, axis_start.y, axis_start.z,
      gpu_rot_matrix.data(HybridTargetLevel::DEVICE),
      psw.xcrd, psw.ycrd, psw.zcrd);

  // 5. Apply PBC
  sampler_->applyPBC(mol);

  // 6. Invalidate energy cache and evaluate energy (already on GPU!)
  sampler_->invalidateEnergyCache();
  const double E_new = sampler_->evaluateTotalEnergy();
  const double delta_E = E_new - E_old;

  // 7. Accept/reject
  if (acceptMove(delta_E)) {
    stats_.n_accepted++;
    stats_.total_energy_change += delta_E;
    return true;
  } else {
    // Restore coordinates on GPU
    launchRestoreCoordinates(
        mol.atom_indices.size(),
        atom_indices.data(HybridTargetLevel::DEVICE),
        saved_x.data(HybridTargetLevel::DEVICE),
        saved_y.data(HybridTargetLevel::DEVICE),
        saved_z.data(HybridTargetLevel::DEVICE),
        psw.xcrd, psw.ycrd, psw.zcrd);

    stats_.n_rejected++;
    return false;
  }
#else
  // CPU fallback - original implementation
  auto saved_coords = sampler_->saveCoordinates(mol);

  rotateBondedAtoms(bond.rotating_atoms, axis_start, axis_end, angle_change, ps);

  sampler_->applyPBC(mol);
  sampler_->invalidateEnergyCache();

  const double E_new = sampler_->evaluateTotalEnergy();
  const double delta_E = E_new - E_old;

  if (acceptMove(delta_E)) {
    stats_.n_accepted++;
    stats_.total_energy_change += delta_E;
    return true;
  } else {
    sampler_->restoreCoordinates(mol, saved_coords);
    stats_.n_rejected++;
    return false;
  }
#endif
}

//-------------------------------------------------------------------------------------------------
// Factory function
//-------------------------------------------------------------------------------------------------
std::unique_ptr<MCMover> createMCMover(const std::string& move_type,
                                       GCMCSampler* sampler,
                                       double beta,
                                       Xoshiro256ppGenerator* rng,
                                       const AtomGraph* topology,
                                       double parameter) {
  std::string lower_type = move_type;
  std::transform(lower_type.begin(), lower_type.end(), lower_type.begin(), ::tolower);

  if (lower_type == "translation" || lower_type == "translate") {
    return std::make_unique<TranslationMover>(sampler, beta, rng, parameter);
  } else if (lower_type == "rotation" || lower_type == "rotate") {
    // Convert degrees to radians if needed
    double max_angle_rad = parameter * pi / 180.0;
    return std::make_unique<RotationMover>(sampler, beta, rng, max_angle_rad);
  } else if (lower_type == "torsion" || lower_type == "dihedral") {
    // Convert degrees to radians
    double max_angle_rad = parameter * pi / 180.0;
    return std::make_unique<TorsionMover>(sampler, beta, rng, topology, max_angle_rad);
  } else {
    throw std::invalid_argument("Unknown MC move type: " + move_type);
  }
}

} // namespace sampling
} // namespace stormm