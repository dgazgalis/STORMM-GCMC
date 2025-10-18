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

#endif // STORMM_USE_HPC

} // namespace sampling
} // namespace stormm

#endif // STORMM_HPC_MC_MOVES_H
