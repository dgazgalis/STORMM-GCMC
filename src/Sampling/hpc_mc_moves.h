// -*-c++-*-
#ifndef STORMM_HPC_MC_MOVES_H
#define STORMM_HPC_MC_MOVES_H

#include "copyright.h"

namespace stormm {
namespace sampling {

#ifdef STORMM_USE_HPC

/// \brief Launch GPU kernel to translate a molecule by a displacement vector
///
/// Applies dx, dy, dz translation to all atoms in the atom_indices array.
/// All operations occur on GPU with no CPU transfers.
///
/// NOTE: This function includes cudaDeviceSynchronize() to ensure the kernel
/// completes before returning. Modified coordinates are safe to use immediately
/// after this call returns.
///
/// \param n_atoms        Number of atoms to translate
/// \param atom_indices   Device pointer to atom indices (size n_atoms)
/// \param dx             X displacement (Angstroms)
/// \param dy             Y displacement (Angstroms)
/// \param dz             Z displacement (Angstroms)
/// \param xcrd           Device pointer to X coordinates (modified in-place)
/// \param ycrd           Device pointer to Y coordinates (modified in-place)
/// \param zcrd           Device pointer to Z coordinates (modified in-place)
void launchTranslateMolecule(
    int n_atoms,
    const int* atom_indices,
    double dx, double dy, double dz,
    double* xcrd, double* ycrd, double* zcrd);

/// \brief Launch GPU kernel to translate molecule to target point (for GCMC insertions)
///
/// Translates all atoms so the molecule's COG moves to the target insertion point.
/// The COG must be pre-computed on GPU using launchCalculateCOG before calling this.
/// This eliminates the need to download coordinates to CPU for insertion operations.
///
/// NOTE: This function does NOT synchronize. Coordinates are consumed by subsequent
/// energy evaluation which will ensure synchronization.
///
/// \param n_atoms        Number of atoms to translate
/// \param atom_indices   Device pointer to atom indices (size n_atoms)
/// \param target_x       Target X coordinate (insertion site)
/// \param target_y       Target Y coordinate (insertion site)
/// \param target_z       Target Z coordinate (insertion site)
/// \param cog_gpu        Device pointer to pre-computed COG [x, y, z] (3 doubles)
/// \param xcrd           Device pointer to X coordinates (modified in-place)
/// \param ycrd           Device pointer to Y coordinates (modified in-place)
/// \param zcrd           Device pointer to Z coordinates (modified in-place)
void launchTranslateMoleculeToPoint(
    int n_atoms,
    const int* atom_indices,
    double target_x, double target_y, double target_z,
    const double* cog_gpu,
    double* xcrd, double* ycrd, double* zcrd);

/// \brief Launch GPU kernel to rotate a molecule about its center of geometry
///
/// Applies a rotation matrix to all atoms about the specified center point.
/// The rotation matrix must be stored in row-major order (9 elements).
///
/// NOTE: This function includes cudaDeviceSynchronize() to ensure the kernel
/// completes before returning. Modified coordinates are safe to use immediately
/// after this call returns.
///
/// \param n_atoms        Number of atoms to rotate
/// \param atom_indices   Device pointer to atom indices (size n_atoms)
/// \param cogx           Center of geometry X coordinate (Angstroms)
/// \param cogy           Center of geometry Y coordinate (Angstroms)
/// \param cogz           Center of geometry Z coordinate (Angstroms)
/// \param rot_matrix     Device pointer to 3x3 rotation matrix (row-major, 9 elements)
/// \param xcrd           Device pointer to X coordinates (modified in-place)
/// \param ycrd           Device pointer to Y coordinates (modified in-place)
/// \param zcrd           Device pointer to Z coordinates (modified in-place)
void launchRotateMolecule(
    int n_atoms,
    const int* atom_indices,
    double cogx, double cogy, double cogz,
    const double* rot_matrix,
    double* xcrd, double* ycrd, double* zcrd);

/// \brief Launch GPU kernel to rotate atoms about an arbitrary axis (for torsion moves)
///
/// Applies Rodrigues' rotation formula to rotate atoms about a specified axis.
/// Used for torsion angle modifications in flexible molecules.
///
/// NOTE: This function includes cudaDeviceSynchronize() to ensure the kernel
/// completes before returning. Modified coordinates are safe to use immediately
/// after this call returns.
///
/// \param n_rotating_atoms  Number of atoms to rotate
/// \param rotating_atoms    Device pointer to rotating atom indices (size n_rotating_atoms)
/// \param axis_start_x      Rotation axis start X coordinate (Angstroms)
/// \param axis_start_y      Rotation axis start Y coordinate (Angstroms)
/// \param axis_start_z      Rotation axis start Z coordinate (Angstroms)
/// \param rot_matrix        Device pointer to 3x3 rotation matrix (row-major, 9 elements)
/// \param xcrd              Device pointer to X coordinates (modified in-place)
/// \param ycrd              Device pointer to Y coordinates (modified in-place)
/// \param zcrd              Device pointer to Z coordinates (modified in-place)
void launchRotateTorsion(
    int n_rotating_atoms,
    const int* rotating_atoms,
    double axis_start_x, double axis_start_y, double axis_start_z,
    const double* rot_matrix,
    double* xcrd, double* ycrd, double* zcrd);

/// \brief Launch GPU kernel to backup coordinates for a set of atoms
///
/// Copies coordinates from the main arrays to backup arrays.
/// Used before MC moves to enable rejection/restoration.
///
/// NOTE: This function includes cudaDeviceSynchronize() to ensure the kernel
/// completes before returning. Backed up coordinates are safe to use immediately
/// after this call returns.
///
/// \param n_atoms        Number of atoms to backup
/// \param atom_indices   Device pointer to atom indices (size n_atoms)
/// \param xcrd           Device pointer to X coordinates (source)
/// \param ycrd           Device pointer to Y coordinates (source)
/// \param zcrd           Device pointer to Z coordinates (source)
/// \param saved_xcrd     Device pointer to backup X coordinates (destination, size n_atoms)
/// \param saved_ycrd     Device pointer to backup Y coordinates (destination, size n_atoms)
/// \param saved_zcrd     Device pointer to backup Z coordinates (destination, size n_atoms)
void launchBackupCoordinates(
    int n_atoms,
    const int* atom_indices,
    const double* xcrd, const double* ycrd, const double* zcrd,
    double* saved_xcrd, double* saved_ycrd, double* saved_zcrd);

/// \brief Launch GPU kernel to restore coordinates for a set of atoms
///
/// Copies coordinates from backup arrays back to main arrays.
/// Used to reject MC moves by restoring previous state.
///
/// NOTE: This function includes cudaDeviceSynchronize() to ensure the kernel
/// completes before returning. Restored coordinates are safe to use immediately
/// after this call returns.
///
/// \param n_atoms        Number of atoms to restore
/// \param atom_indices   Device pointer to atom indices (size n_atoms)
/// \param saved_xcrd     Device pointer to backup X coordinates (source, size n_atoms)
/// \param saved_ycrd     Device pointer to backup Y coordinates (source, size n_atoms)
/// \param saved_zcrd     Device pointer to backup Z coordinates (source, size n_atoms)
/// \param xcrd           Device pointer to X coordinates (destination)
/// \param ycrd           Device pointer to Y coordinates (destination)
/// \param zcrd           Device pointer to Z coordinates (destination)
void launchRestoreCoordinates(
    int n_atoms,
    const int* atom_indices,
    const double* saved_xcrd, const double* saved_ycrd, const double* saved_zcrd,
    double* xcrd, double* ycrd, double* zcrd);

/// \brief Launch GPU kernel to backup velocities for a set of atoms
///
/// Copies velocities from the main arrays to backup arrays.
/// Used before GCMC moves to enable rejection/restoration.
///
/// \param n_atoms        Number of atoms to backup
/// \param atom_indices   Device pointer to atom indices (size n_atoms)
/// \param xvel           Device pointer to X velocities (source)
/// \param yvel           Device pointer to Y velocities (source)
/// \param zvel           Device pointer to Z velocities (source)
/// \param saved_xvel     Device pointer to backup X velocities (destination, size n_atoms)
/// \param saved_yvel     Device pointer to backup Y velocities (destination, size n_atoms)
/// \param saved_zvel     Device pointer to backup Z velocities (destination, size n_atoms)
void launchBackupVelocities(
    int n_atoms,
    const int* atom_indices,
    const double* xvel, const double* yvel, const double* zvel,
    double* saved_xvel, double* saved_yvel, double* saved_zvel);

/// \brief Launch GPU kernel to restore velocities for a set of atoms
///
/// Copies velocities from backup arrays back to main arrays.
/// Used to reject GCMC moves by restoring previous state.
///
/// \param n_atoms        Number of atoms to restore
/// \param atom_indices   Device pointer to atom indices (size n_atoms)
/// \param saved_xvel     Device pointer to backup X velocities (source, size n_atoms)
/// \param saved_yvel     Device pointer to backup Y velocities (source, size n_atoms)
/// \param saved_zvel     Device pointer to backup Z velocities (source, size n_atoms)
/// \param xvel           Device pointer to X velocities (destination)
/// \param yvel           Device pointer to Y velocities (destination)
/// \param zvel           Device pointer to Z velocities (destination)
void launchRestoreVelocities(
    int n_atoms,
    const int* atom_indices,
    const double* saved_xvel, const double* saved_yvel, const double* saved_zvel,
    double* xvel, double* yvel, double* zvel);

/// \brief Launch GPU kernel to backup coordinates AND velocities (FUSED)
///
/// Combines backup of both coordinates and velocities into a single kernel launch.
/// This reduces kernel launch overhead and improves memory locality.
/// Used before GCMC moves to enable rejection/restoration.
///
/// \param n_atoms        Number of atoms to backup
/// \param atom_indices   Device pointer to atom indices (size n_atoms)
/// \param xcrd           Device pointer to X coordinates (source)
/// \param ycrd           Device pointer to Y coordinates (source)
/// \param zcrd           Device pointer to Z coordinates (source)
/// \param xvel           Device pointer to X velocities (source)
/// \param yvel           Device pointer to Y velocities (source)
/// \param zvel           Device pointer to Z velocities (source)
/// \param saved_xcrd     Device pointer to backup X coordinates (destination, size n_atoms)
/// \param saved_ycrd     Device pointer to backup Y coordinates (destination, size n_atoms)
/// \param saved_zcrd     Device pointer to backup Z coordinates (destination, size n_atoms)
/// \param saved_xvel     Device pointer to backup X velocities (destination, size n_atoms)
/// \param saved_yvel     Device pointer to backup Y velocities (destination, size n_atoms)
/// \param saved_zvel     Device pointer to backup Z velocities (destination, size n_atoms)
void launchBackupCoordinatesAndVelocities(
    int n_atoms,
    const int* atom_indices,
    const double* xcrd, const double* ycrd, const double* zcrd,
    const double* xvel, const double* yvel, const double* zvel,
    double* saved_xcrd, double* saved_ycrd, double* saved_zcrd,
    double* saved_xvel, double* saved_yvel, double* saved_zvel);

/// \brief Launch GPU kernel to restore coordinates AND velocities (FUSED)
///
/// Combines restore of both coordinates and velocities into a single kernel launch.
/// This reduces kernel launch overhead and improves memory locality.
/// Used to reject GCMC moves by restoring previous state.
///
/// \param n_atoms        Number of atoms to restore
/// \param atom_indices   Device pointer to atom indices (size n_atoms)
/// \param saved_xcrd     Device pointer to backup X coordinates (source, size n_atoms)
/// \param saved_ycrd     Device pointer to backup Y coordinates (source, size n_atoms)
/// \param saved_zcrd     Device pointer to backup Z coordinates (source, size n_atoms)
/// \param saved_xvel     Device pointer to backup X velocities (source, size n_atoms)
/// \param saved_yvel     Device pointer to backup Y velocities (source, size n_atoms)
/// \param saved_zvel     Device pointer to backup Z velocities (source, size n_atoms)
/// \param xcrd           Device pointer to X coordinates (destination)
/// \param ycrd           Device pointer to Y coordinates (destination)
/// \param zcrd           Device pointer to Z coordinates (destination)
/// \param xvel           Device pointer to X velocities (destination)
/// \param yvel           Device pointer to Y velocities (destination)
/// \param zvel           Device pointer to Z velocities (destination)
void launchRestoreCoordinatesAndVelocities(
    int n_atoms,
    const int* atom_indices,
    const double* saved_xcrd, const double* saved_ycrd, const double* saved_zcrd,
    const double* saved_xvel, const double* saved_yvel, const double* saved_zvel,
    double* xcrd, double* ycrd, double* zcrd,
    double* xvel, double* yvel, double* zvel);

/// \brief Launch GPU kernel to calculate center of geometry (COG) of a molecule
///
/// Uses parallel reduction to compute the average position of all atoms in the molecule.
/// Result is written to device memory arrays cog_x, cog_y, cog_z (single element each).
///
/// NOTE: This function does NOT synchronize. Caller must ensure synchronization before
/// using COG results (typically via Hybrid::download() which implicitly syncs).
///
/// \param n_atoms        Number of atoms in molecule
/// \param atom_indices   Device pointer to atom indices (size n_atoms)
/// \param xcrd           Device pointer to X coordinates (read-only)
/// \param ycrd           Device pointer to Y coordinates (read-only)
/// \param zcrd           Device pointer to Z coordinates (read-only)
/// \param cog_x          Device pointer to output COG X coordinate (single element)
/// \param cog_y          Device pointer to output COG Y coordinate (single element)
/// \param cog_z          Device pointer to output COG Z coordinate (single element)
void launchCalculateCOG(
    int n_atoms,
    const int* atom_indices,
    const double* xcrd, const double* ycrd, const double* zcrd,
    double* cog_x, double* cog_y, double* cog_z);

/// \brief Launch GPU kernel to generate random rotation matrix (uniform random rotation)
///
/// Uses cuRAND to generate a uniform random rotation using the Shoemake method.
/// The rotation is represented as a 3x3 matrix in row-major order (9 elements).
/// Reference: K. Shoemake, "Uniform random rotations", Graphics Gems III, 1992
///
/// \param rot_matrix_gpu  Device pointer to 9-element rotation matrix (row-major, output)
/// \param curand_states   Device pointer to cuRAND states (used for RNG)
void launchGenerateRandomRotationMatrix(
    double* rot_matrix_gpu,
    void* curand_states);

/// \brief Launch GPU kernel to generate Maxwell-Boltzmann velocities for a set of atoms
///
/// Uses cuRAND to generate random velocities from a Maxwell-Boltzmann distribution
/// on the GPU. This eliminates CPU generation and the need to upload velocities.
///
/// The Maxwell-Boltzmann distribution for a single velocity component is:
///   v ~ N(0, sqrt(kB*T/m))
/// where kB = Boltzmann constant, T = temperature, m = atomic mass
///
/// NOTE: This function does NOT synchronize. Velocities are consumed by subsequent
/// energy evaluation which will ensure synchronization.
///
/// \param n_atoms        Number of atoms to generate velocities for
/// \param atom_indices   Device pointer to atom indices (size n_atoms)
/// \param masses         Device pointer to atomic masses in amu (size n_atoms)
/// \param temperature    Temperature in Kelvin
/// \param xvel           Device pointer to X velocities (modified in-place)
/// \param yvel           Device pointer to Y velocities (modified in-place)
/// \param zvel           Device pointer to Z velocities (modified in-place)
/// \param curand_states  Device pointer to cuRAND states (one per atom)
void launchGenerateMaxwellBoltzmannVelocities(
    int n_atoms,
    const int* atom_indices,
    const double* masses,
    double temperature,
    double* xvel, double* yvel, double* zvel,
    void* curand_states);

/// \brief Initialize cuRAND states for GPU random number generation
///
/// Allocates and initializes cuRAND states on the GPU with unique seeds.
/// Each thread gets its own RNG state for independent random streams.
///
/// \param n_states       Number of cuRAND states to allocate
/// \param base_seed      Base seed for RNG initialization
/// \return Device pointer to allocated cuRAND states (must be freed with cudaFree)
void* initializeCurandStates(int n_states, unsigned long long base_seed);

/// \brief Launch GPU kernel to apply periodic boundary conditions to wrap molecule
///
/// Computes PBC shift needed to bring molecule's center of geometry into primary box,
/// then applies the same shift to all atoms to preserve molecular geometry.
///
/// This GPU-accelerated version eliminates the ~3-4ms overhead of CPU download/upload
/// that occurs in the CPU version (GCMCSampler::applyPBC).
///
/// NOTE: Requires pre-computed COG from launchCalculateCOG().
/// NOTE: This function does NOT synchronize - PBC wrapping is followed by energy eval which syncs.
///
/// \param n_atoms        Number of atoms in molecule
/// \param atom_indices   Device pointer to atom indices (size n_atoms)
/// \param cog_x          Center of geometry X coordinate (pre-computed)
/// \param cog_y          Center of geometry Y coordinate (pre-computed)
/// \param cog_z          Center of geometry Z coordinate (pre-computed)
/// \param box_x          Box dimension X (Angstroms)
/// \param box_y          Box dimension Y (Angstroms)
/// \param box_z          Box dimension Z (Angstroms)
/// \param xcrd           Device pointer to X coordinates (modified in-place)
/// \param ycrd           Device pointer to Y coordinates (modified in-place)
/// \param zcrd           Device pointer to Z coordinates (modified in-place)
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
    double* zcrd);

/// \brief Launch GPU kernel to apply PBC with COG on GPU (fully GPU-resident)
///
/// Variant that takes COG as device pointers, eliminating the need to download COG to CPU.
/// Computes PBC shift entirely on GPU. This eliminates ~0.1-0.3ms of COG download overhead.
///
/// NOTE: This function does NOT synchronize - PBC wrapping is followed by energy eval which syncs.
///
/// \param n_atoms        Number of atoms in molecule
/// \param atom_indices   Device pointer to atom indices (size n_atoms)
/// \param cog_gpu        Device pointer to COG array [x, y, z] (3 doubles)
/// \param box_x          Box dimension X (Angstroms)
/// \param box_y          Box dimension Y (Angstroms)
/// \param box_z          Box dimension Z (Angstroms)
/// \param xcrd           Device pointer to X coordinates (modified in-place)
/// \param ycrd           Device pointer to Y coordinates (modified in-place)
/// \param zcrd           Device pointer to Z coordinates (modified in-place)
void launchApplyPBCWrapGPU(
    int n_atoms,
    const int* atom_indices,
    const double* cog_gpu,
    double box_x,
    double box_y,
    double box_z,
    double* xcrd,
    double* ycrd,
    double* zcrd);

#endif // STORMM_USE_HPC

} // namespace sampling
} // namespace stormm

#endif // STORMM_HPC_MC_MOVES_H
