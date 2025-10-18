// -*-c++-*-
#ifndef STORMM_LAMBDA_NONBONDED_H
#define STORMM_LAMBDA_NONBONDED_H

#include "copyright.h"
#include "Constants/behavior.h"
#include "DataTypes/stormm_vector_types.h"
#include "Math/vector_ops.h"
#include "Potential/energy_enumerators.h"
#include "Potential/scorecard.h"
#include "Potential/static_exclusionmask.h"
#include "Topology/atomgraph_abstracts.h"
#include "Topology/atomgraph_enumerators.h"

namespace stormm {
namespace energy {

using topology::LambdaNonbondedKit;
using topology::UnitCellType;

/// \brief Evaluate lambda-scaled nonbonded energy and forces for GCMC/NCMC simulations.
///        This function applies per-atom lambda scaling to both electrostatic and VDW
///        interactions, with softcore potentials to prevent singularities during
///        coupling/decoupling transitions. Ghost atoms (lambda = 0) are excluded from
///        force calculations entirely for optimal performance.
///
/// \param lambda_nbk          Lambda-aware nonbonded parameters with per-atom coupling factors
/// \param ser                  Static exclusion mask for 1-2, 1-3, 1-4 exclusions
/// \param xcrd                 Cartesian X coordinates of all atoms
/// \param ycrd                 Cartesian Y coordinates of all atoms
/// \param zcrd                 Cartesian Z coordinates of all atoms
/// \param umat                 Unit cell transformation matrix (or nullptr for vacuum)
/// \param invu                 Inverse unit cell transformation (or nullptr for vacuum)
/// \param unit_cell            Type of periodic boundary conditions
/// \param xfrc                 X-direction forces (accumulated)
/// \param yfrc                 Y-direction forces (accumulated)
/// \param zfrc                 Z-direction forces (accumulated)
/// \param sc                   ScoreCard for energy accumulation
/// \param eval_force           Whether to evaluate forces
/// \param eval_energy          Whether to evaluate energies
/// \param system_index         Index of the system in the ScoreCard
/// \param inv_gpos_factor      Inverse scaling factor for coordinates (for fixed precision)
/// \param force_factor         Scaling factor for forces (for fixed precision)
/// \param clash_distance       Minimum distance below which atoms are considered clashing (Angstroms)
/// \param clash_ratio          Minimum r/sigma ratio for clash detection
/// \param vdw_coupling_threshold  Lambda threshold for VDW->electrostatic transition (typically 0.75)
/// \param softcore_alpha       Softcore parameter (typically 0.5)
template <typename Tcoord, typename Tforce, typename Tcalc>
void evaluateLambdaNonbondedEnergy(const LambdaNonbondedKit<Tcalc> &lambda_nbk,
                                   const StaticExclusionMaskReader &ser,
                                   const Tcoord* xcrd, const Tcoord* ycrd, const Tcoord* zcrd,
                                   const double* umat, const double* invu,
                                   UnitCellType unit_cell, Tforce* xfrc, Tforce* yfrc,
                                   Tforce* zfrc, ScoreCard *sc, EvaluateForce eval_force,
                                   EvaluateEnergy eval_energy, int system_index,
                                   Tcalc inv_gpos_factor, Tcalc force_factor,
                                   Tcalc clash_distance, Tcalc clash_ratio,
                                   Tcalc vdw_coupling_threshold = 0.75,
                                   Tcalc softcore_alpha = 0.5);

} // namespace energy
} // namespace stormm

#include "lambda_nonbonded.tpp"

#endif
