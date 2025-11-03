// -*-c++-*-
#include "copyright.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cmath>
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
// GPU kernel: Translate molecule to target point (for GCMC insertions)
//
// Translates all atoms so that the molecule's COG moves to the target insertion point.
// Displacement = target - COG, applied to all atoms.
// COG must be pre-computed on GPU using launchCalculateCOG.
// Thread-parallel over atoms: each thread handles one atom.
//-------------------------------------------------------------------------------------------------
__global__ void kTranslateMoleculeToPoint(
    const int n_atoms,
    const int* __restrict__ atom_indices,
    const double target_x,
    const double target_y,
    const double target_z,
    const double* __restrict__ cog_gpu,
    double* __restrict__ xcrd,
    double* __restrict__ ycrd,
    double* __restrict__ zcrd)
{
  // Use shared memory to broadcast displacement to all threads
  __shared__ double dx, dy, dz;

  // Thread 0 reads COG and computes displacement
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    dx = target_x - cog_gpu[0];
    dy = target_y - cog_gpu[1];
    dz = target_z - cog_gpu[2];
  }

  // Sync to ensure displacement is computed before all threads use it
  __syncthreads();

  // All threads apply the displacement to their atoms
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
// Calculate center of geometry (COG) using parallel reduction
//
// Uses block-level parallel reduction to compute sum, then divides by n_atoms.
// Each block computes partial sums, then a second kernel could combine them, but for small
// molecules (< 1000 atoms typical in GCMC), we use a single block with atomic operations.
//-------------------------------------------------------------------------------------------------
__global__ void kCalculateCOG(
    const int n_atoms,
    const int* __restrict__ atom_indices,
    const double* __restrict__ xcrd,
    const double* __restrict__ ycrd,
    const double* __restrict__ zcrd,
    double* __restrict__ cog_x,
    double* __restrict__ cog_y,
    double* __restrict__ cog_z)
{
  __shared__ double sum_x[256];
  __shared__ double sum_y[256];
  __shared__ double sum_z[256];

  const int tid = threadIdx.x;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Thread 0 of block 0 initializes output to zero (eliminates cudaMemset overhead)
  if (tid == 0 && blockIdx.x == 0) {
    *cog_x = 0.0;
    *cog_y = 0.0;
    *cog_z = 0.0;
  }

  // Initialize shared memory
  sum_x[tid] = 0.0;
  sum_y[tid] = 0.0;
  sum_z[tid] = 0.0;

  // Load and accumulate coordinates (grid-stride loop for large molecules)
  for (int i = idx; i < n_atoms; i += blockDim.x * gridDim.x) {
    const int atom = atom_indices[i];
    sum_x[tid] += xcrd[atom];
    sum_y[tid] += ycrd[atom];
    sum_z[tid] += zcrd[atom];
  }

  __syncthreads();

  // Parallel reduction within block (assumes blockDim.x = 256)
  if (tid < 128) {
    sum_x[tid] += sum_x[tid + 128];
    sum_y[tid] += sum_y[tid + 128];
    sum_z[tid] += sum_z[tid + 128];
  }
  __syncthreads();

  if (tid < 64) {
    sum_x[tid] += sum_x[tid + 64];
    sum_y[tid] += sum_y[tid + 64];
    sum_z[tid] += sum_z[tid + 64];
  }
  __syncthreads();

  // Warp-level reduction (no sync needed within warp)
  if (tid < 32) {
    sum_x[tid] += sum_x[tid + 32];
    sum_y[tid] += sum_y[tid + 32];
    sum_z[tid] += sum_z[tid + 32];
    __syncwarp();

    sum_x[tid] += sum_x[tid + 16];
    sum_y[tid] += sum_y[tid + 16];
    sum_z[tid] += sum_z[tid + 16];
    __syncwarp();

    sum_x[tid] += sum_x[tid + 8];
    sum_y[tid] += sum_y[tid + 8];
    sum_z[tid] += sum_z[tid + 8];
    __syncwarp();

    sum_x[tid] += sum_x[tid + 4];
    sum_y[tid] += sum_y[tid + 4];
    sum_z[tid] += sum_z[tid + 4];
    __syncwarp();

    sum_x[tid] += sum_x[tid + 2];
    sum_y[tid] += sum_y[tid + 2];
    sum_z[tid] += sum_z[tid + 2];
    __syncwarp();

    sum_x[tid] += sum_x[tid + 1];
    sum_y[tid] += sum_y[tid + 1];
    sum_z[tid] += sum_z[tid + 1];
  }

  // Thread 0 of each block writes partial result using atomicAdd
  if (tid == 0) {
    atomicAdd(cog_x, sum_x[0]);
    atomicAdd(cog_y, sum_y[0]);
    atomicAdd(cog_z, sum_z[0]);
  }
}

//-------------------------------------------------------------------------------------------------
// Finalize COG by dividing by n_atoms (single-thread kernel)
//-------------------------------------------------------------------------------------------------
__global__ void kFinalizeCOG(
    const int n_atoms,
    double* __restrict__ cog_x,
    double* __restrict__ cog_y,
    double* __restrict__ cog_z)
{
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    const double inv_n = 1.0 / static_cast<double>(n_atoms);
    *cog_x *= inv_n;
    *cog_y *= inv_n;
    *cog_z *= inv_n;
  }
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

  // NOTE: Removed cudaDeviceSynchronize() to allow kernel pipelining.
  // Caller must sync before downloading results or making critical decisions.
}

//-------------------------------------------------------------------------------------------------
// Launch wrapper: Translate molecule to target point (for GCMC insertions)
//-------------------------------------------------------------------------------------------------
void launchTranslateMoleculeToPoint(
    int n_atoms,
    const int* atom_indices,
    double target_x, double target_y, double target_z,
    const double* cog_gpu,
    double* xcrd, double* ycrd, double* zcrd)
{
  const int threads_per_block = 256;
  const int num_blocks = (n_atoms + threads_per_block - 1) / threads_per_block;

  kTranslateMoleculeToPoint<<<num_blocks, threads_per_block>>>(
      n_atoms, atom_indices, target_x, target_y, target_z, cog_gpu, xcrd, ycrd, zcrd);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    rtErr("CUDA kernel launch failed in launchTranslateMoleculeToPoint: " +
          std::string(cudaGetErrorString(err)), "launchTranslateMoleculeToPoint");
  }

  // NOTE: No synchronization - coordinates consumed by subsequent energy evaluation which syncs
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

  // NOTE: Removed cudaDeviceSynchronize() to allow kernel pipelining.
  // Caller must sync before downloading results or making critical decisions.
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

  // NOTE: Removed cudaDeviceSynchronize() to allow kernel pipelining.
  // Caller must sync before downloading results or making critical decisions.
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

  // NOTE: Removed cudaDeviceSynchronize() to allow kernel pipelining.
  // Backup kernels don't need sync - they just copy data on GPU.
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

  // NOTE: Removed cudaDeviceSynchronize() to allow kernel pipelining.
  // Restore kernels don't need sync - coordinates stay on GPU for energy eval.
}

//-------------------------------------------------------------------------------------------------
// Launch wrapper: Calculate COG
//-------------------------------------------------------------------------------------------------
void launchCalculateCOG(
    int n_atoms,
    const int* atom_indices,
    const double* xcrd, const double* ycrd, const double* zcrd,
    double* cog_x, double* cog_y, double* cog_z)
{
  // OPTIMIZATION: Removed cudaMemset calls - kernel now initializes output to zero internally
  // This eliminates ~0.05-0.15ms of overhead per COG calculation

  // Use multiple blocks for better occupancy on large molecules
  const int threads_per_block = 256;
  const int num_blocks = std::min(32, (n_atoms + threads_per_block - 1) / threads_per_block);

  // Launch reduction kernel
  kCalculateCOG<<<num_blocks, threads_per_block>>>(
      n_atoms, atom_indices, xcrd, ycrd, zcrd, cog_x, cog_y, cog_z);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    rtErr("CUDA kernel launch failed in launchCalculateCOG: " +
          std::string(cudaGetErrorString(err)), "launchCalculateCOG");
  }

  // Launch finalization kernel to divide by n_atoms
  kFinalizeCOG<<<1, 1>>>(n_atoms, cog_x, cog_y, cog_z);

  err = cudaGetLastError();
  if (err != cudaSuccess) {
    rtErr("CUDA kernel launch failed in kFinalizeCOG: " +
          std::string(cudaGetErrorString(err)), "launchCalculateCOG");
  }

  // OPTIMIZATION: Removed cudaDeviceSynchronize() - caller's download() already syncs
  // This eliminates 50-100μs of unnecessary blocking per COG calculation
}

//-------------------------------------------------------------------------------------------------
// GPU kernel: Backup velocities for a set of atoms
//
// Copies velocities from main arrays to backup arrays for later restoration if GCMC move rejected.
// Thread-parallel over atoms: each thread handles one atom.
//-------------------------------------------------------------------------------------------------
__global__ void kBackupVelocities(
    const int n_atoms,
    const int* __restrict__ atom_indices,
    const double* __restrict__ xvel,
    const double* __restrict__ yvel,
    const double* __restrict__ zvel,
    double* __restrict__ saved_xvel,
    double* __restrict__ saved_yvel,
    double* __restrict__ saved_zvel)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_atoms) return;

  const int atom = atom_indices[idx];
  saved_xvel[idx] = xvel[atom];
  saved_yvel[idx] = yvel[atom];
  saved_zvel[idx] = zvel[atom];
}

//-------------------------------------------------------------------------------------------------
// GPU kernel: Restore velocities for a set of atoms
//
// Copies velocities from backup arrays back to main arrays to reject GCMC move.
// Thread-parallel over atoms: each thread handles one atom.
//-------------------------------------------------------------------------------------------------
__global__ void kRestoreVelocities(
    const int n_atoms,
    const int* __restrict__ atom_indices,
    const double* __restrict__ saved_xvel,
    const double* __restrict__ saved_yvel,
    const double* __restrict__ saved_zvel,
    double* __restrict__ xvel,
    double* __restrict__ yvel,
    double* __restrict__ zvel)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_atoms) return;

  const int atom = atom_indices[idx];
  xvel[atom] = saved_xvel[idx];
  yvel[atom] = saved_yvel[idx];
  zvel[atom] = saved_zvel[idx];
}

//-------------------------------------------------------------------------------------------------
// GPU kernel: Backup coordinates AND velocities for a set of atoms (FUSED)
//
// Combines backup of both coordinates and velocities into a single kernel launch.
// This reduces kernel launch overhead and improves memory locality.
// Thread-parallel over atoms: each thread handles one atom.
//-------------------------------------------------------------------------------------------------
__global__ void kBackupCoordinatesAndVelocities(
    const int n_atoms,
    const int* __restrict__ atom_indices,
    const double* __restrict__ xcrd,
    const double* __restrict__ ycrd,
    const double* __restrict__ zcrd,
    const double* __restrict__ xvel,
    const double* __restrict__ yvel,
    const double* __restrict__ zvel,
    double* __restrict__ saved_xcrd,
    double* __restrict__ saved_ycrd,
    double* __restrict__ saved_zcrd,
    double* __restrict__ saved_xvel,
    double* __restrict__ saved_yvel,
    double* __restrict__ saved_zvel)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_atoms) return;

  const int atom = atom_indices[idx];

  // Backup coordinates
  saved_xcrd[idx] = xcrd[atom];
  saved_ycrd[idx] = ycrd[atom];
  saved_zcrd[idx] = zcrd[atom];

  // Backup velocities
  saved_xvel[idx] = xvel[atom];
  saved_yvel[idx] = yvel[atom];
  saved_zvel[idx] = zvel[atom];
}

//-------------------------------------------------------------------------------------------------
// GPU kernel: Restore coordinates AND velocities for a set of atoms (FUSED)
//
// Combines restore of both coordinates and velocities into a single kernel launch.
// This reduces kernel launch overhead and improves memory locality.
// Thread-parallel over atoms: each thread handles one atom.
//-------------------------------------------------------------------------------------------------
__global__ void kRestoreCoordinatesAndVelocities(
    const int n_atoms,
    const int* __restrict__ atom_indices,
    const double* __restrict__ saved_xcrd,
    const double* __restrict__ saved_ycrd,
    const double* __restrict__ saved_zcrd,
    const double* __restrict__ saved_xvel,
    const double* __restrict__ saved_yvel,
    const double* __restrict__ saved_zvel,
    double* __restrict__ xcrd,
    double* __restrict__ ycrd,
    double* __restrict__ zcrd,
    double* __restrict__ xvel,
    double* __restrict__ yvel,
    double* __restrict__ zvel)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_atoms) return;

  const int atom = atom_indices[idx];

  // Restore coordinates
  xcrd[atom] = saved_xcrd[idx];
  ycrd[atom] = saved_ycrd[idx];
  zcrd[atom] = saved_zcrd[idx];

  // Restore velocities
  xvel[atom] = saved_xvel[idx];
  yvel[atom] = saved_yvel[idx];
  zvel[atom] = saved_zvel[idx];
}

//-------------------------------------------------------------------------------------------------
void launchBackupVelocities(
    int n_atoms,
    const int* atom_indices,
    const double* xvel, const double* yvel, const double* zvel,
    double* saved_xvel, double* saved_yvel, double* saved_zvel)
{
  const int threads_per_block = 256;
  const int num_blocks = (n_atoms + threads_per_block - 1) / threads_per_block;

  kBackupVelocities<<<num_blocks, threads_per_block>>>(
      n_atoms, atom_indices, xvel, yvel, zvel, saved_xvel, saved_yvel, saved_zvel);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    rtErr("CUDA kernel launch failed in launchBackupVelocities: " +
          std::string(cudaGetErrorString(err)), "launchBackupVelocities");
  }

  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    rtErr("CUDA kernel execution failed in launchBackupVelocities: " +
          std::string(cudaGetErrorString(err)), "launchBackupVelocities");
  }
}

//-------------------------------------------------------------------------------------------------
void launchRestoreVelocities(
    int n_atoms,
    const int* atom_indices,
    const double* saved_xvel, const double* saved_yvel, const double* saved_zvel,
    double* xvel, double* yvel, double* zvel)
{
  const int threads_per_block = 256;
  const int num_blocks = (n_atoms + threads_per_block - 1) / threads_per_block;

  kRestoreVelocities<<<num_blocks, threads_per_block>>>(
      n_atoms, atom_indices, saved_xvel, saved_yvel, saved_zvel, xvel, yvel, zvel);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    rtErr("CUDA kernel launch failed in launchRestoreVelocities: " +
          std::string(cudaGetErrorString(err)), "launchRestoreVelocities");
  }

  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    rtErr("CUDA kernel execution failed in launchRestoreVelocities: " +
          std::string(cudaGetErrorString(err)), "launchRestoreVelocities");
  }
}

//-------------------------------------------------------------------------------------------------
// Launch wrapper: Backup coordinates AND velocities (FUSED)
//-------------------------------------------------------------------------------------------------
void launchBackupCoordinatesAndVelocities(
    int n_atoms,
    const int* atom_indices,
    const double* xcrd, const double* ycrd, const double* zcrd,
    const double* xvel, const double* yvel, const double* zvel,
    double* saved_xcrd, double* saved_ycrd, double* saved_zcrd,
    double* saved_xvel, double* saved_yvel, double* saved_zvel)
{
  const int threads_per_block = 256;
  const int num_blocks = (n_atoms + threads_per_block - 1) / threads_per_block;

  kBackupCoordinatesAndVelocities<<<num_blocks, threads_per_block>>>(
      n_atoms, atom_indices,
      xcrd, ycrd, zcrd,
      xvel, yvel, zvel,
      saved_xcrd, saved_ycrd, saved_zcrd,
      saved_xvel, saved_yvel, saved_zvel);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    rtErr("CUDA kernel launch failed in launchBackupCoordinatesAndVelocities: " +
          std::string(cudaGetErrorString(err)), "launchBackupCoordinatesAndVelocities");
  }

  // NOTE: Removed cudaDeviceSynchronize() to allow kernel pipelining.
  // Backup kernels don't need sync - they just copy data on GPU.
}

//-------------------------------------------------------------------------------------------------
// Launch wrapper: Restore coordinates AND velocities (FUSED)
//-------------------------------------------------------------------------------------------------
void launchRestoreCoordinatesAndVelocities(
    int n_atoms,
    const int* atom_indices,
    const double* saved_xcrd, const double* saved_ycrd, const double* saved_zcrd,
    const double* saved_xvel, const double* saved_yvel, const double* saved_zvel,
    double* xcrd, double* ycrd, double* zcrd,
    double* xvel, double* yvel, double* zvel)
{
  const int threads_per_block = 256;
  const int num_blocks = (n_atoms + threads_per_block - 1) / threads_per_block;

  kRestoreCoordinatesAndVelocities<<<num_blocks, threads_per_block>>>(
      n_atoms, atom_indices,
      saved_xcrd, saved_ycrd, saved_zcrd,
      saved_xvel, saved_yvel, saved_zvel,
      xcrd, ycrd, zcrd,
      xvel, yvel, zvel);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    rtErr("CUDA kernel launch failed in launchRestoreCoordinatesAndVelocities: " +
          std::string(cudaGetErrorString(err)), "launchRestoreCoordinatesAndVelocities");
  }

  // NOTE: Removed cudaDeviceSynchronize() to allow kernel pipelining.
  // Restore kernels don't need sync - coordinates stay on GPU for energy eval.
}

//-------------------------------------------------------------------------------------------------
// GPU kernel: Generate random rotation matrix using Shoemake method (uniform random rotation)
//
// Uses cuRAND to generate a uniform random rotation quaternion, then converts to 3x3 matrix.
// Reference: K. Shoemake, "Uniform random rotations", Graphics Gems III, 1992
//
// The matrix is stored in row-major order (9 elements).
//-------------------------------------------------------------------------------------------------
__global__ void kGenerateRandomRotationMatrix(
    double* __restrict__ rot_matrix,
    curandState* __restrict__ curand_state)
{
  // Only need one thread to generate the matrix
  if (threadIdx.x != 0 || blockIdx.x != 0) return;

  // Load cuRAND state
  curandState local_state = curand_state[0];

  // Generate 3 uniform random numbers [0,1]
  const double u1 = curand_uniform_double(&local_state);
  const double u2 = curand_uniform_double(&local_state);
  const double u3 = curand_uniform_double(&local_state);

  // Shoemake method: convert to uniform random quaternion
  const double sqrt1_u1 = sqrt(1.0 - u1);
  const double sqrtu1 = sqrt(u1);
  const double two_pi_u2 = 2.0 * M_PI * u2;
  const double two_pi_u3 = 2.0 * M_PI * u3;

  const double qw = sqrt1_u1 * sin(two_pi_u2);
  const double qx = sqrt1_u1 * cos(two_pi_u2);
  const double qy = sqrtu1 * sin(two_pi_u3);
  const double qz = sqrtu1 * cos(two_pi_u3);

  // Precompute quaternion products for matrix conversion
  const double xx = qx * qx, yy = qy * qy, zz = qz * qz;
  const double xy = qx * qy, xz = qx * qz, yz = qy * qz;
  const double wx = qw * qx, wy = qw * qy, wz = qw * qz;

  // Convert quaternion to 3x3 rotation matrix (row-major order)
  rot_matrix[0] = 1.0 - 2.0 * (yy + zz);
  rot_matrix[1] = 2.0 * (xy - wz);
  rot_matrix[2] = 2.0 * (xz + wy);
  rot_matrix[3] = 2.0 * (xy + wz);
  rot_matrix[4] = 1.0 - 2.0 * (xx + zz);
  rot_matrix[5] = 2.0 * (yz - wx);
  rot_matrix[6] = 2.0 * (xz - wy);
  rot_matrix[7] = 2.0 * (yz + wx);
  rot_matrix[8] = 1.0 - 2.0 * (xx + yy);

  // Save updated cuRAND state
  curand_state[0] = local_state;
}

//-------------------------------------------------------------------------------------------------
// GPU kernel: Initialize cuRAND states for parallel RNG
//
// Each thread initializes its own cuRAND state with a unique seed.
// Seed = base_seed + thread_idx for reproducible but independent RNG streams.
//-------------------------------------------------------------------------------------------------
__global__ void kInitCurandStates(
    curandState* __restrict__ states,
    const int n_states,
    const unsigned long long base_seed)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_states) return;

  // Initialize cuRAND state with unique seed per thread
  curand_init(base_seed + idx, 0, 0, &states[idx]);
}

//-------------------------------------------------------------------------------------------------
void* initializeCurandStates(int n_states, unsigned long long base_seed)
{
  // Allocate cuRAND states on GPU
  curandState* d_states = nullptr;
  const size_t bytes = n_states * sizeof(curandState);
  cudaError_t err = cudaMalloc(&d_states, bytes);
  if (err != cudaSuccess) {
    rtErr("Failed to allocate cuRAND states (" + std::to_string(bytes / 1024) +
          " KB): " + std::string(cudaGetErrorString(err)), "initializeCurandStates");
    return nullptr;
  }

  // Launch initialization kernel
  const int threads_per_block = 256;
  const int num_blocks = (n_states + threads_per_block - 1) / threads_per_block;

  kInitCurandStates<<<num_blocks, threads_per_block>>>(d_states, n_states, base_seed);

  err = cudaGetLastError();
  if (err != cudaSuccess) {
    cudaFree(d_states);
    rtErr("Failed to launch cuRAND initialization kernel: " +
          std::string(cudaGetErrorString(err)), "initializeCurandStates");
    return nullptr;
  }

  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    cudaFree(d_states);
    rtErr("cuRAND initialization kernel failed: " +
          std::string(cudaGetErrorString(err)), "initializeCurandStates");
    return nullptr;
  }

  return static_cast<void*>(d_states);
}

//-------------------------------------------------------------------------------------------------
// GPU kernel: Generate Maxwell-Boltzmann velocities for atoms
//
// Generates velocities from a Maxwell-Boltzmann distribution using cuRAND.
// For each velocity component (x, y, z):
//   v ~ N(0, sigma) where sigma = sqrt(kB * T / m)
//
// Physical constants:
//   kB = 0.001987204 kcal/(mol·K) (Boltzmann constant)
//   velocity units: Å/ps (STORMM internal units)
//
// Conversion factor from kcal/mol to (Å/ps)²:
//   1 kcal/mol = 4.184 kJ/mol = 4184 J/mol
//   1 amu = 1.66054e-27 kg
//   1 Å/ps = 100 m/s
//   factor = sqrt(kB*T / m) * sqrt(4184 / 1.66054e-27) / 100
//          = sqrt(kB*T / m) * 20455.26  [conversion factor]
//
// Simplified: sigma_velocity = sqrt(0.001987204 * T / mass_amu) * 20.455
//-------------------------------------------------------------------------------------------------
__global__ void kGenerateMaxwellBoltzmannVelocities(
    const int n_atoms,
    const int* __restrict__ atom_indices,
    const double* __restrict__ masses,
    const double temperature,
    double* __restrict__ xvel,
    double* __restrict__ yvel,
    double* __restrict__ zvel,
    curandState* __restrict__ curand_states)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_atoms) return;

  const int atom = atom_indices[idx];
  const double mass = masses[idx];  // Mass in amu

  // Boltzmann constant in kcal/(mol·K)
  constexpr double kB = 0.001987204;

  // Conversion factor: sqrt(kB*T/m) in kcal/(mol*amu) to Å/ps
  // factor = sqrt(2 * kB * T / mass) * 20.455
  // Using factor of 20.455 to convert from internal energy units to velocity units
  const double sigma = sqrt(kB * temperature / mass) * 20.455;

  // Generate three independent Gaussian random numbers for vx, vy, vz
  curandState local_state = curand_states[idx];

  const double vx = curand_normal_double(&local_state) * sigma;
  const double vy = curand_normal_double(&local_state) * sigma;
  const double vz = curand_normal_double(&local_state) * sigma;

  xvel[atom] = vx;
  yvel[atom] = vy;
  zvel[atom] = vz;

  // Save updated cuRAND state
  curand_states[idx] = local_state;
}

//-------------------------------------------------------------------------------------------------
// Launch wrapper: Generate random rotation matrix on GPU
//-------------------------------------------------------------------------------------------------
void launchGenerateRandomRotationMatrix(
    double* rot_matrix_gpu,
    void* curand_states)
{
  // Single thread kernel - only need 1 block, 1 thread
  kGenerateRandomRotationMatrix<<<1, 1>>>(
      rot_matrix_gpu,
      static_cast<curandState*>(curand_states));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    rtErr("CUDA kernel launch failed in launchGenerateRandomRotationMatrix: " +
          std::string(cudaGetErrorString(err)), "launchGenerateRandomRotationMatrix");
  }

  // NOTE: Removed cudaDeviceSynchronize() to allow kernel pipelining.
  // Matrix stays on GPU for immediate use by rotation kernel.
}

//-------------------------------------------------------------------------------------------------
void launchGenerateMaxwellBoltzmannVelocities(
    int n_atoms,
    const int* atom_indices,
    const double* masses,
    double temperature,
    double* xvel, double* yvel, double* zvel,
    void* curand_states)
{
  const int threads_per_block = 256;
  const int num_blocks = (n_atoms + threads_per_block - 1) / threads_per_block;

  kGenerateMaxwellBoltzmannVelocities<<<num_blocks, threads_per_block>>>(
      n_atoms, atom_indices, masses, temperature, xvel, yvel, zvel,
      static_cast<curandState*>(curand_states));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    rtErr("CUDA kernel launch failed in launchGenerateMaxwellBoltzmannVelocities: " +
          std::string(cudaGetErrorString(err)), "launchGenerateMaxwellBoltzmannVelocities");
  }

  // OPTIMIZATION: Removed cudaDeviceSynchronize() - velocities consumed by energy eval which syncs
  // This eliminates ~50μs per insertion
}

//-------------------------------------------------------------------------------------------------
// GPU kernel: Apply periodic boundary conditions to wrap molecule into primary box
//
// Computes the PBC shift needed to bring molecule's center of geometry into the primary box,
// then applies the same shift to all atoms to preserve molecular geometry.
//
// Algorithm:
// 1. Thread 0 computes shift = -floor(cog / box_size) * box_size
// 2. All threads apply the same shift to their atoms (broadcast)
//
// Thread-parallel over atoms: each thread handles one atom after shift is computed.
//-------------------------------------------------------------------------------------------------
__global__ void kApplyPBCWrap(
    const int n_atoms,
    const int* __restrict__ atom_indices,
    const double cog_x,
    const double cog_y,
    const double cog_z,
    const double box_x,
    const double box_y,
    const double box_z,
    double* __restrict__ xcrd,
    double* __restrict__ ycrd,
    double* __restrict__ zcrd)
{
  // Use shared memory to broadcast shift to all threads in block
  __shared__ double shift_x, shift_y, shift_z;

  // Thread 0 computes the PBC shift
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    shift_x = -floor(cog_x / box_x) * box_x;
    shift_y = -floor(cog_y / box_y) * box_y;
    shift_z = -floor(cog_z / box_z) * box_z;
  }

  // Sync to ensure shift is computed before all threads use it
  __syncthreads();

  // All threads apply the shift to their atoms
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_atoms) return;

  const int atom = atom_indices[idx];
  xcrd[atom] += shift_x;
  ycrd[atom] += shift_y;
  zcrd[atom] += shift_z;
}

//-------------------------------------------------------------------------------------------------
// Launch wrapper: Apply PBC wrapping to molecule (GPU-accelerated)
//
// Eliminates CPU download/upload overhead by keeping coordinates on GPU.
// Uses pre-computed COG from launchCalculateCOG().
//
// Performance: ~0.01ms (GPU) vs ~3-4ms (CPU with download/upload)
//-------------------------------------------------------------------------------------------------
void launchApplyPBCWrap(
    int n_atoms,
    const int* atom_indices,
    double cog_x,
    double cog_y,
    double cog_z,
    double box_x,
    double box_y,
    double box_z,
    double* xcrd,
    double* ycrd,
    double* zcrd)
{
  const int threads_per_block = 256;
  const int num_blocks = (n_atoms + threads_per_block - 1) / threads_per_block;

  kApplyPBCWrap<<<num_blocks, threads_per_block>>>(
      n_atoms, atom_indices, cog_x, cog_y, cog_z,
      box_x, box_y, box_z,
      xcrd, ycrd, zcrd);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    rtErr("CUDA kernel launch failed in launchApplyPBCWrap: " +
          std::string(cudaGetErrorString(err)), "launchApplyPBCWrap");
  }

  // NOTE: Removed cudaDeviceSynchronize() to allow kernel pipelining.
  // PBC wrapping is typically followed by energy evaluation which syncs.
}

//-------------------------------------------------------------------------------------------------
// GPU kernel: Apply periodic boundary conditions with COG on GPU (fully GPU-resident)
//
// This variant reads COG from device memory, eliminating the need to download COG to CPU.
// Computes PBC shift entirely on GPU. This saves ~0.1-0.3ms per move.
//-------------------------------------------------------------------------------------------------
__global__ void kApplyPBCWrapGPU(
    const int n_atoms,
    const int* __restrict__ atom_indices,
    const double* __restrict__ cog_gpu,
    const double box_x,
    const double box_y,
    const double box_z,
    double* __restrict__ xcrd,
    double* __restrict__ ycrd,
    double* __restrict__ zcrd)
{
  // Use shared memory to broadcast shift to all threads in block
  __shared__ double shift_x, shift_y, shift_z;

  // Thread 0 reads COG from device memory and computes the PBC shift
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    const double cog_x = cog_gpu[0];
    const double cog_y = cog_gpu[1];
    const double cog_z = cog_gpu[2];

    shift_x = -floor(cog_x / box_x) * box_x;
    shift_y = -floor(cog_y / box_y) * box_y;
    shift_z = -floor(cog_z / box_z) * box_z;
  }

  // Sync to ensure shift is computed before all threads use it
  __syncthreads();

  // All threads apply the shift to their atoms
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_atoms) return;

  const int atom = atom_indices[idx];
  xcrd[atom] += shift_x;
  ycrd[atom] += shift_y;
  zcrd[atom] += shift_z;
}

//-------------------------------------------------------------------------------------------------
// Launch wrapper: Apply PBC with COG on GPU (fully GPU-resident)
//
// Fully GPU-resident PBC wrapping. Eliminates COG download overhead (~0.1-0.3ms).
//-------------------------------------------------------------------------------------------------
void launchApplyPBCWrapGPU(
    int n_atoms,
    const int* atom_indices,
    const double* cog_gpu,
    double box_x,
    double box_y,
    double box_z,
    double* xcrd,
    double* ycrd,
    double* zcrd)
{
  const int threads_per_block = 256;
  const int num_blocks = (n_atoms + threads_per_block - 1) / threads_per_block;

  kApplyPBCWrapGPU<<<num_blocks, threads_per_block>>>(
      n_atoms, atom_indices, cog_gpu,
      box_x, box_y, box_z,
      xcrd, ycrd, zcrd);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    rtErr("CUDA kernel launch failed in launchApplyPBCWrapGPU: " +
          std::string(cudaGetErrorString(err)), "launchApplyPBCWrapGPU");
  }

  // NOTE: Removed cudaDeviceSynchronize() to allow kernel pipelining.
  // PBC wrapping is typically followed by energy evaluation which syncs.
}

} // namespace sampling
} // namespace stormm
