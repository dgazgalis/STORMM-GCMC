// -*-c++-*-
#ifndef STORMM_MAP_DENSITY_LAMBDA_H
#define STORMM_MAP_DENSITY_LAMBDA_H

#include "copyright.h"
#include "cellgrid.h"
#include "pmigrid.h"
#include "Synthesis/atomgraph_synthesis.h"

namespace stormm {
namespace energy {

using synthesis::AtomGraphSynthesis;

/// \brief Spread particle density to PME grid with lambda scaling for GCMC.
///
/// This function extends the standard mapDensity() to scale charges by electrostatic lambda
/// values (λ_ele). Atoms with λ below threshold contribute zero charge to the grid, ensuring
/// that ghost molecules in GCMC simulations don't affect reciprocal space electrostatics.
///
/// Key differences from standard mapDensity():
/// - Charges scaled by λ_ele[i] during B-spline spreading
/// - Atoms with λ < threshold are skipped entirely (no grid contribution)
/// - Coupled atoms (λ ≈ 1) contribute full charge
/// - Transitioning atoms (0 < λ < 1) contribute proportionally
///
/// Physical interpretation:
/// In GCMC, molecules transition between coupled (λ=1) and ghost (λ=0) states. The reciprocal
/// space electrostatic energy must reflect only coupled atoms to maintain detailed balance:
///   E_reciprocal(λ) = (1/2V) Σₖ |S_λ(k)|²/k²·exp(-k²/4α²)
///   where S_λ(k) = Σᵢ (qᵢ·λᵢ)·exp(ik·rᵢ)  <-- Lambda-scaled structure factor
///
/// Usage in GCMC workflow:
/// 1. During NCMC: Call each MD step as λ values gradually change (λ ramping)
/// 2. Energy evaluation: Call before acceptance/rejection decision
/// 3. PME grid rebuilt automatically with current lambda-scaled charges
///
/// \param pm                The particle-mesh interaction grid (must be NonbondedTheme::ELECTROSTATIC)
/// \param cg                Cell grid containing particle coordinates in local cell basis
/// \param poly_ag           Topology synthesis with atomic charges and system definitions
/// \param lambda_ele        Per-atom electrostatic lambda values [0.0, 1.0], length = total atoms
/// \param lambda_threshold  Atoms with λ < threshold treated as zero charge (default 0.01)
/// \{
template <typename T, typename Tacc, typename Tcalc, typename T4>
void mapDensityLambda(PMIGrid *pm, const CellGrid<T, Tacc, Tcalc, T4> *cg,
                      const AtomGraphSynthesis *poly_ag,
                      const double* lambda_ele,
                      double lambda_threshold = 0.01);

template <typename T, typename Tacc, typename Tcalc, typename T4>
void mapDensityLambda(PMIGrid *pm, const CellGrid<T, Tacc, Tcalc, T4> &cg,
                      const AtomGraphSynthesis &poly_ag,
                      const double* lambda_ele,
                      double lambda_threshold = 0.01);
/// \}

/// \brief Accumulate lambda-scaled density for atoms in a single spatial decomposition cell.
///
/// This is the core function that performs lambda-aware charge spreading. It iterates over all
/// atoms in the specified cell, computes B-spline coefficients for their positions, scales the
/// charge by lambda, and spreads to the PME grid.
///
/// Overloaded:
///   - Operate on real-valued particle-mesh interaction grids (PMIGridWriter)
///   - Operate on fixed-precision particle-mesh interaction grids (PMIGridAccumulator)
///
/// \param pm_wrt            Writeable abstract for real-valued PMI grids
/// \param pm_acc            Writeable abstract for fixed-precision PMI grids
/// \param sysid             Index of the system in the synthesis
/// \param cell_i            Cell index along system grid's A axis
/// \param cell_j            Cell index along system grid's B axis
/// \param cell_k            Cell index along system grid's C axis
/// \param cgr               Cell grid reader (contains coordinates and cell decomposition)
/// \param synbk             Non-bonded parameter tables (charges, LJ params) for all systems
/// \param lambda_ele        Per-atom lambda values
/// \param lambda_threshold  Threshold below which atoms are skipped
/// \{
template <typename T, typename Tacc, typename Tcalc, typename Tcalc2, typename T4>
void accumulateCellDensityLambda(PMIGridWriter *pm_wrt, int sysid,
                                 int cell_i, int cell_j, int cell_k,
                                 const CellGridReader<T, Tacc, Tcalc, T4> &cgr,
                                 const SyNonbondedKit<Tcalc, Tcalc2> &synbk,
                                 const double* lambda_ele,
                                 double lambda_threshold = 0.01);

template <typename T, typename Tacc, typename Tcalc, typename Tcalc2, typename T4>
void accumulateCellDensityLambda(PMIGridAccumulator *pm_acc, int sysid,
                                 int cell_i, int cell_j, int cell_k,
                                 const CellGridReader<T, Tacc, Tcalc, T4> &cgr,
                                 const SyNonbondedKit<Tcalc, Tcalc2> &synbk,
                                 const double* lambda_ele,
                                 double lambda_threshold = 0.01);
/// \}

/// \brief Non-template wrapper for mapDensityLambda that extracts the CellGrid from PMIGrid.
///
/// This function provides a simple interface for lambda-aware charge spreading when you only
/// have the PMIGrid and AtomGraphSynthesis. The CellGrid is retrieved internally from PMIGrid.
///
/// \param pm              PME grid (contains embedded CellGrid)
/// \param poly_ag         Topology synthesis
/// \param lambda_ele      Per-atom electrostatic lambda values
/// \param lambda_threshold Threshold below which atoms are considered ghosts (default 0.01)
/// \{
void mapDensityLambda(PMIGrid *pm, const AtomGraphSynthesis *poly_ag,
                      const double* lambda_ele, double lambda_threshold = 0.01);

void mapDensityLambda(PMIGrid *pm, const AtomGraphSynthesis &poly_ag,
                      const double* lambda_ele, double lambda_threshold = 0.01);
/// \}

/// \brief Unroll template calls for mapDensityLambda at the accumulator precision level.
///
/// These helper functions handle the template unrolling needed to select the correct precision
/// model for accumulation and calculation based on the CellGrid and PMIGrid types.
///
/// \param pm        The particle-mesh interaction grids
/// \param cg_tacc   Type index for force accumulation in cell grid
/// \param cg_tcalc  Type index for calculations in cell grid
/// \param poly_ag   Topology synthesis
/// \param lambda_ele      Lambda values
/// \param lambda_threshold Threshold
/// \{
template <typename T, typename T4>
void unrollMapDensityLambdaCall(PMIGrid *pm, size_t cg_tacc, size_t cg_tcalc,
                                const AtomGraphSynthesis *poly_ag,
                                const double* lambda_ele,
                                double lambda_threshold = 0.01);

template <typename T, typename Tacc, typename T4>
void unrollMapDensityLambdaCall(PMIGrid *pm, size_t cg_tcalc,
                                const AtomGraphSynthesis *poly_ag,
                                const double* lambda_ele,
                                double lambda_threshold = 0.01);
/// \}

} // namespace energy
} // namespace stormm

#include "map_density_lambda.tpp"

#endif
