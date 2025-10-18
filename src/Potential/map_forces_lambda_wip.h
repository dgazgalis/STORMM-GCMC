// -*-c++-*-
#ifndef STORMM_MAP_FORCES_LAMBDA_H
#define STORMM_MAP_FORCES_LAMBDA_H

#include "copyright.h"
#include "MolecularMechanics/mm_controls.h"
#include "cellgrid.h"
#include "map_density.h"
#include "pmigrid.h"
#include "Synthesis/atomgraph_synthesis.h"
#include "Synthesis/phasespace_synthesis.h"

namespace stormm {
namespace energy {

using synthesis::AtomGraphSynthesis;
using synthesis::PhaseSpaceSynthesis;
using synthesis::PsSynthesisWriter;

/// \brief Interpolate forces from PME reciprocal space grid with lambda scaling for GCMC.
///
/// This function performs the inverse of mapDensityLambda() by extracting forces from the
/// reciprocal space potential grid after FFT and convolution. Forces are computed using
/// B-spline derivatives and scaled by electrostatic lambda values.
///
/// Key features:
/// - Uses B-spline derivatives to compute force components from the potential grid
/// - Scales interpolated forces by λ_ele[i] for proper lambda dynamics
/// - Atoms with λ < threshold contribute zero force (ghosts have no interaction)
/// - Forces are accumulated into existing force arrays (not overwritten)
///
/// Physical interpretation:
/// After the PME reciprocal space convolution, the grid contains the electrostatic potential
/// φ(r). Forces are computed as:
///   F_i = -q_i·λ_i·∇φ(r_i)
/// where the gradient is evaluated using B-spline derivative coefficients.
///
/// The chain rule for lambda-scaled forces gives:
///   F_reciprocal,i = -λ_i · q_i · Σ_grid [φ(grid) · ∇B_spline(r_i - grid)]
///
/// Usage in GCMC/Lambda dynamics workflow:
/// 1. Call mapDensityLambda() to spread lambda-scaled charges to grid
/// 2. Perform FFT forward transform
/// 3. Apply reciprocal space convolution (energy calculation)
/// 4. Perform FFT inverse transform
/// 5. Call mapForcesLambda() to extract lambda-scaled forces
///
/// \param pm                 The particle-mesh interaction grid containing potential after FFT
/// \param topology_synthesis Topology synthesis with atomic charges and system definitions
/// \param ps_synthesis       Phase space synthesis containing force arrays to accumulate into
/// \param lambda_ele         Per-atom electrostatic lambda values [0.0, 1.0]
/// \param lambda_threshold   Atoms with λ < threshold are skipped (default 0.01)
/// \{
void mapForcesLambda(PMIGrid *pm,
                     const AtomGraphSynthesis *topology_synthesis,
                     PhaseSpaceSynthesis *ps_synthesis,
                     const double* lambda_ele,
                     double lambda_threshold = 0.01);

void mapForcesLambda(PMIGrid &pm,
                     const AtomGraphSynthesis &topology_synthesis,
                     PhaseSpaceSynthesis &ps_synthesis,
                     const double* lambda_ele,
                     double lambda_threshold = 0.01);
/// \}

/// \brief Accumulate lambda-scaled reciprocal forces for atoms in a single spatial cell.
///
/// This is the core function that performs lambda-aware force interpolation. It iterates over
/// all atoms in the specified cell, computes B-spline derivative coefficients for their
/// positions, scales forces by lambda, and accumulates into the force arrays.
///
/// The force on atom i is computed as:
///   F_i = -λ_i · q_i · Σ_jkl [φ(j,k,l) · dB_a(i-j) · B_b(i-k) · B_c(i-l)]
/// where dB represents the B-spline derivative and φ is the potential grid.
///
/// \param pm_reader         Read-only abstract for PMI grid containing potential
/// \param ps_writer         Writeable abstract for phase space (force arrays)
/// \param sysid            Index of the system in the synthesis
/// \param cell_i           Cell index along system grid's A axis
/// \param cell_j           Cell index along system grid's B axis
/// \param cell_k           Cell index along system grid's C axis
/// \param cgr              Cell grid reader (contains coordinates and cell decomposition)
/// \param synbk            Non-bonded parameter tables (charges) for all systems
/// \param lambda_ele       Per-atom lambda values
/// \param lambda_threshold Threshold below which atoms are skipped
/// \{
template <typename T, typename Tacc, typename Tcalc, typename Tcalc2, typename T4>
void accumulateCellForcesLambda(const PMIGridReader &pm_reader,
                                PsSynthesisWriter &ps_writer,
                                int sysid,
                                int cell_i, int cell_j, int cell_k,
                                const CellGridReader<T, Tacc, Tcalc, T4> &cgr,
                                const SyNonbondedKit<Tcalc, Tcalc2> &synbk,
                                const double* lambda_ele,
                                double lambda_threshold = 0.01);
/// \}

/// \brief Template unrolling helpers for mapForcesLambda
///
/// These helper functions handle the template unrolling needed to select the correct precision
/// model based on the CellGrid and PMIGrid types.
///
/// \param pm                 The particle-mesh interaction grids
/// \param cg_tacc           Type index for force accumulation in cell grid
/// \param cg_tcalc          Type index for calculations in cell grid
/// \param topology_synthesis Topology synthesis
/// \param ps_synthesis      Phase space synthesis with force arrays
/// \param lambda_ele        Lambda values
/// \param lambda_threshold  Threshold
/// \{
template <typename T, typename T4>
void unrollMapForcesLambdaCall(PMIGrid *pm, size_t cg_tacc, size_t cg_tcalc,
                               const AtomGraphSynthesis *topology_synthesis,
                               PhaseSpaceSynthesis *ps_synthesis,
                               const double* lambda_ele,
                               double lambda_threshold = 0.01);

template <typename T, typename Tacc, typename T4>
void unrollMapForcesLambdaCall(PMIGrid *pm, size_t cg_tcalc,
                               const AtomGraphSynthesis *topology_synthesis,
                               PhaseSpaceSynthesis *ps_synthesis,
                               const double* lambda_ele,
                               double lambda_threshold = 0.01);
/// \}

#ifdef STORMM_USE_HPC
/// \brief GPU implementation of lambda-scaled force interpolation from PME grid.
///
/// This function launches GPU kernels to perform force interpolation in parallel.
/// Forces are computed using B-spline derivatives and scaled by per-atom lambda values.
///
/// \param pm                 PME grid containing reciprocal space potential
/// \param mm_ctrl           Molecular mechanics control object for GPU work units
/// \param cg                Cell grid with particle coordinates
/// \param topology_synthesis Topology with charges
/// \param ps_synthesis      Phase space with force arrays to accumulate into
/// \param lambda_ele        Per-atom electrostatic lambda values
/// \param lambda_threshold  Threshold below which atoms are treated as ghosts
/// \param launcher          Kernel launcher configuration
/// \param approach          Method for GPU kernel execution
template <typename T, typename Tacc, typename Tcalc, typename T4>
void mapForcesLambda(PMIGrid *pm,
                     MolecularMechanicsControls *mm_ctrl,
                     const CellGrid<T, Tacc, Tcalc, T4> *cg,
                     const AtomGraphSynthesis *topology_synthesis,
                     PhaseSpaceSynthesis *ps_synthesis,
                     const double* lambda_ele,
                     double lambda_threshold,
                     const CoreKlManager &launcher,
                     const QMapMethod approach = QMapMethod::AUTOMATIC);
#endif

} // namespace energy
} // namespace stormm

#include "map_forces_lambda_wip.tpp"

#endif