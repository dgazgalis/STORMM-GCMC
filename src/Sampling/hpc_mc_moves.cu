// -*-c++-*-
#include "copyright.h"
#include <cuda_runtime.h>
#include "Reporting/error_format.h"
#include "hpc_mc_moves.h"

namespace stormm {
namespace sampling {

//-------------------------------------------------------------------------------------------------
// GPU kernel: Translate molecule by displacement vector
//
// Applies a uniform displacement (dx, dy, dz) to all atoms in the atom_indices array.
// Thread-parallel over atoms: each thread handles one atom.
//-------------------------------------------------------------------------------------------------
__global__ void kTranslateMolecule(
    const int n_atoms,
    const int* __restrict__ atom_indices,
    const double dx,
    const double dy,
    const double dz,
    double* __restrict__ xcrd,
    double* __restrict__ ycrd,
    double* __restrict__ zcrd)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_atoms) return;

  const int atom = atom_indices[idx];
  xcrd[atom] += dx;
  ycrd[atom] += dy;
  zcrd[atom] += dz;
}

//-------------------------------------------------------------------------------------------------
// GPU kernel: Rotate molecule about center of geometry
//
// Applies a 3x3 rotation matrix to each atom about the specified center of geometry.
// Rotation matrix is in row-major order: [R00, R01, R02, R10, R11, R12, R20, R21, R22]
// Thread-parallel over atoms: each thread handles one atom.
//-------------------------------------------------------------------------------------------------
__global__ void kRotateMolecule(
    const int n_atoms,
    const int* __restrict__ atom_indices,
    const double cogx,
    const double cogy,
    const double cogz,
    const double* __restrict__ rot_matrix,
    double* __restrict__ xcrd,
    double* __restrict__ ycrd,
    double* __restrict__ zcrd)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_atoms) return;

  const int atom = atom_indices[idx];

  // Translate to origin (center of geometry)
  const double x = xcrd[atom] - cogx;
  const double y = ycrd[atom] - cogy;
  const double z = zcrd[atom] - cogz;

  // Apply rotation matrix (row-major)
  const double rx = rot_matrix[0] * x + rot_matrix[1] * y + rot_matrix[2] * z;
  const double ry = rot_matrix[3] * x + rot_matrix[4] * y + rot_matrix[5] * z;
  const double rz = rot_matrix[6] * x + rot_matrix[7] * y + rot_matrix[8] * z;

  // Translate back
  xcrd[atom] = rx + cogx;
  ycrd[atom] = ry + cogy;
  zcrd[atom] = rz + cogz;
}

//-------------------------------------------------------------------------------------------------
// GPU kernel: Rotate atoms about an arbitrary axis (for torsion moves)
//
// Applies a 3x3 rotation matrix about an arbitrary axis defined by axis_start.
// Used for torsion angle modifications where rotation axis is a molecular bond.
// Thread-parallel over atoms: each thread handles one atom.
//-------------------------------------------------------------------------------------------------
__global__ void kRotateTorsion(
    const int n_rotating_atoms,
    const int* __restrict__ rotating_atoms,
    const double axis_start_x,
    const double axis_start_y,
    const double axis_start_z,
    const double* __restrict__ rot_matrix,
    double* __restrict__ xcrd,
    double* __restrict__ ycrd,
    double* __restrict__ zcrd)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_rotating_atoms) return;

  const int atom = rotating_atoms[idx];

  // Translate to rotation axis origin
  const double x = xcrd[atom] - axis_start_x;
  const double y = ycrd[atom] - axis_start_y;
  const double z = zcrd[atom] - axis_start_z;

  // Apply rotation matrix (row-major)
  const double rx = rot_matrix[0] * x + rot_matrix[1] * y + rot_matrix[2] * z;
  const double ry = rot_matrix[3] * x + rot_matrix[4] * y + rot_matrix[5] * z;
  const double rz = rot_matrix[6] * x + rot_matrix[7] * y + rot_matrix[8] * z;

  // Translate back
  xcrd[atom] = rx + axis_start_x;
  ycrd[atom] = ry + axis_start_y;
  zcrd[atom] = rz + axis_start_z;
}

//-------------------------------------------------------------------------------------------------
// GPU kernel: Backup coordinates for a set of atoms
//
// Copies coordinates from main arrays to backup arrays for later restoration if MC move rejected.
// Thread-parallel over atoms: each thread handles one atom.
//-------------------------------------------------------------------------------------------------
__global__ void kBackupCoordinates(
    const int n_atoms,
    const int* __restrict__ atom_indices,
    const double* __restrict__ xcrd,
    const double* __restrict__ ycrd,
    const double* __restrict__ zcrd,
    double* __restrict__ saved_xcrd,
    double* __restrict__ saved_ycrd,
    double* __restrict__ saved_zcrd)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_atoms) return;

  const int atom = atom_indices[idx];
  saved_xcrd[idx] = xcrd[atom];
  saved_ycrd[idx] = ycrd[atom];
  saved_zcrd[idx] = zcrd[atom];
}

//-------------------------------------------------------------------------------------------------
// GPU kernel: Restore coordinates for a set of atoms
//
// Copies coordinates from backup arrays back to main arrays to reject MC move.
// Thread-parallel over atoms: each thread handles one atom.
//-------------------------------------------------------------------------------------------------
__global__ void kRestoreCoordinates(
    const int n_atoms,
    const int* __restrict__ atom_indices,
    const double* __restrict__ saved_xcrd,
    const double* __restrict__ saved_ycrd,
    const double* __restrict__ saved_zcrd,
    double* __restrict__ xcrd,
    double* __restrict__ ycrd,
    double* __restrict__ zcrd)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_atoms) return;

  const int atom = atom_indices[idx];
  xcrd[atom] = saved_xcrd[idx];
  ycrd[atom] = saved_ycrd[idx];
  zcrd[atom] = saved_zcrd[idx];
}

//-------------------------------------------------------------------------------------------------
// Launch wrapper: Translate molecule
//-------------------------------------------------------------------------------------------------
void launchTranslateMolecule(
    int n_atoms,
    const int* atom_indices,
    double dx, double dy, double dz,
    double* xcrd, double* ycrd, double* zcrd)
{
  const int threads_per_block = 256;
  const int num_blocks = (n_atoms + threads_per_block - 1) / threads_per_block;

  kTranslateMolecule<<<num_blocks, threads_per_block>>>(
      n_atoms, atom_indices, dx, dy, dz, xcrd, ycrd, zcrd);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    rtErr("CUDA kernel launch failed in launchTranslateMolecule: " +
          std::string(cudaGetErrorString(err)), "launchTranslateMolecule");
  }

  // Synchronize to ensure completion before subsequent operations
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    rtErr("CUDA kernel execution failed in launchTranslateMolecule: " +
          std::string(cudaGetErrorString(err)), "launchTranslateMolecule");
  }
}

//-------------------------------------------------------------------------------------------------
// Launch wrapper: Rotate molecule about center of geometry
//-------------------------------------------------------------------------------------------------
void launchRotateMolecule(
    int n_atoms,
    const int* atom_indices,
    double cogx, double cogy, double cogz,
    const double* rot_matrix,
    double* xcrd, double* ycrd, double* zcrd)
{
  const int threads_per_block = 256;
  const int num_blocks = (n_atoms + threads_per_block - 1) / threads_per_block;

  kRotateMolecule<<<num_blocks, threads_per_block>>>(
      n_atoms, atom_indices, cogx, cogy, cogz, rot_matrix, xcrd, ycrd, zcrd);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    rtErr("CUDA kernel launch failed in launchRotateMolecule: " +
          std::string(cudaGetErrorString(err)), "launchRotateMolecule");
  }

  // Synchronize to ensure completion before subsequent operations
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    rtErr("CUDA kernel execution failed in launchRotateMolecule: " +
          std::string(cudaGetErrorString(err)), "launchRotateMolecule");
  }
}

//-------------------------------------------------------------------------------------------------
// Launch wrapper: Rotate atoms about arbitrary axis (torsion)
//-------------------------------------------------------------------------------------------------
void launchRotateTorsion(
    int n_rotating_atoms,
    const int* rotating_atoms,
    double axis_start_x, double axis_start_y, double axis_start_z,
    const double* rot_matrix,
    double* xcrd, double* ycrd, double* zcrd)
{
  const int threads_per_block = 256;
  const int num_blocks = (n_rotating_atoms + threads_per_block - 1) / threads_per_block;

  kRotateTorsion<<<num_blocks, threads_per_block>>>(
      n_rotating_atoms, rotating_atoms, axis_start_x, axis_start_y, axis_start_z,
      rot_matrix, xcrd, ycrd, zcrd);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    rtErr("CUDA kernel launch failed in launchRotateTorsion: " +
          std::string(cudaGetErrorString(err)), "launchRotateTorsion");
  }

  // Synchronize to ensure completion before subsequent operations
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    rtErr("CUDA kernel execution failed in launchRotateTorsion: " +
          std::string(cudaGetErrorString(err)), "launchRotateTorsion");
  }
}

//-------------------------------------------------------------------------------------------------
// Launch wrapper: Backup coordinates
//-------------------------------------------------------------------------------------------------
void launchBackupCoordinates(
    int n_atoms,
    const int* atom_indices,
    const double* xcrd, const double* ycrd, const double* zcrd,
    double* saved_xcrd, double* saved_ycrd, double* saved_zcrd)
{
  const int threads_per_block = 256;
  const int num_blocks = (n_atoms + threads_per_block - 1) / threads_per_block;

  kBackupCoordinates<<<num_blocks, threads_per_block>>>(
      n_atoms, atom_indices, xcrd, ycrd, zcrd, saved_xcrd, saved_ycrd, saved_zcrd);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    rtErr("CUDA kernel launch failed in launchBackupCoordinates: " +
          std::string(cudaGetErrorString(err)), "launchBackupCoordinates");
  }

  // Synchronize to ensure completion before subsequent operations
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    rtErr("CUDA kernel execution failed in launchBackupCoordinates: " +
          std::string(cudaGetErrorString(err)), "launchBackupCoordinates");
  }
}

//-------------------------------------------------------------------------------------------------
// Launch wrapper: Restore coordinates
//-------------------------------------------------------------------------------------------------
void launchRestoreCoordinates(
    int n_atoms,
    const int* atom_indices,
    const double* saved_xcrd, const double* saved_ycrd, const double* saved_zcrd,
    double* xcrd, double* ycrd, double* zcrd)
{
  const int threads_per_block = 256;
  const int num_blocks = (n_atoms + threads_per_block - 1) / threads_per_block;

  kRestoreCoordinates<<<num_blocks, threads_per_block>>>(
      n_atoms, atom_indices, saved_xcrd, saved_ycrd, saved_zcrd, xcrd, ycrd, zcrd);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    rtErr("CUDA kernel launch failed in launchRestoreCoordinates: " +
          std::string(cudaGetErrorString(err)), "launchRestoreCoordinates");
  }

  // Synchronize to ensure completion before subsequent operations
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    rtErr("CUDA kernel execution failed in launchRestoreCoordinates: " +
          std::string(cudaGetErrorString(err)), "launchRestoreCoordinates");
  }
}

} // namespace sampling
} // namespace stormm
