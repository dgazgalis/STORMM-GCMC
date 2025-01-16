// -*-c++-*-
#ifndef STORMM_EVAL_SYNTHESIS_H
#define STORMM_EVAL_SYNTHESIS_H

#include "copyright.h"
#include "DataTypes/common_types.h"
#include "DataTypes/stormm_vector_types.h"
#include "Math/vector_ops.h"
#include "MolecularMechanics/mm_evaluation.h"
#include "Potential/scorecard.h"
#include "Potential/static_exclusionmask.h"
#include "Potential/valence_potential.h"
#include "Synthesis/atomgraph_synthesis.h"
#include "Synthesis/phasespace_synthesis.h"
#include "Synthesis/nonbonded_workunit.h"
#include "Synthesis/static_mask_synthesis.h"
#include "Synthesis/synthesis_abstracts.h"
#include "Synthesis/synthesis_enumerators.h"
#include "Synthesis/valence_workunit.h"

namespace stormm {
namespace energy {

using mm::commitVwuEnergies;
using mm::evalVwuInitEnergy;
using stmath::readBitFromMask;
using synthesis::AtomGraphSynthesis;
using synthesis::maximum_valence_work_unit_atoms;
using synthesis::small_block_max_imports;
using synthesis::supertile_wu_abstract_length;
using synthesis::tile_groups_wu_abstract_length;
using synthesis::PhaseSpaceSynthesis;
using synthesis::PsSynthesisWriter;
using synthesis::SeMaskSynthesisReader;
using synthesis::StaticExclusionMaskSynthesis;
using synthesis::SyAtomUpdateKit;
using synthesis::SyNonbondedKit;
using synthesis::SyValenceKit;
using synthesis::SyRestraintKit;
using synthesis::vwu_abstract_length;
using synthesis::VwuAbstractMap;
using synthesis::VwuGoal;
using synthesis::VwuTask;

/// \brief Evaluate the Generalized Born radii for a synthesis of systems following the protocols
///        of GPU kernels regarding preservation of precision and conversions to and from the
///        floating point calculation type.
///
/// \param synbk  
/// \param syse
/// \param psyw

/// \brief Evaluate the non-bonded energies (and possibly forces) of a synthesis of systems in
///        isolated boundary conditions using non-bonded work units composed of tile groups.  These
///        smaller forms of the all-to-all non-bonded work units enumerate all of the tiles they
///        process.
///
/// \param synbk                   Non-bonded parameters for all atoms in the compilation of
///                                systems
/// \param syse                    Abstract of the exclusion masks for all systems
/// \param psyr                    Abstract for the coordinate synthesis
/// \param ecardw                  Energy tracker object writer
/// \param eval_elec_force         Flag to also carry out force evaluation of electrostatic
///                                interactions (energy is always evaluated in CPU functions)
/// \param eval_vdw_force          Flag to also carry out force evaluation of van-der Waals
///                                interactions
/// \param clash_minimum_distance  The minimum distance at which two particles will interact via
///                                the standard, and not a softened, potential
/// \param clash_ratio             The minimum ratio of interparticle distance to Lennard-Jones
///                                pairwise parameter at which two particles will interact via the
///                                standard, and not a softened, potential
template <typename Tcalc, typename Tcalc2>
void evalSyNonbondedTileGroups(const SyNonbondedKit<Tcalc, Tcalc2> synbk,
                               const SeMaskSynthesisReader syse, PsSynthesisWriter *psyw,
                               ScoreCard *ecard, NonbondedTask task, EvaluateForce eval_elec_force,
                               EvaluateForce eval_vdw_force, Tcalc clash_minimum_distance = 0.0,
                               Tcalc clash_ratio = 0.0);

/// \brief Evaluate the non-bonded energy with a particular precision level.  This will invoke
///        the proper C++ function.  Descriptions of input parameters follow from
///        evalSynNonbondedTileGroups(), above, in addition to:
///
/// \param poly_ag          Hold non-bonded parameters for all systems in the synthesis
/// \param poly_se          Holds exclusions mapped for all systems in the synthesis
/// \param poly_ps          Holds coordinates and force accumulators for the synthesis of systems
/// \param ecard            Tracking object to collect electrostatic and van-der Waals energies
/// \param eval_elec_force  Indicate whether to evaluate electrostatic forces
/// \param evel_vdw_force   Indicate whether to evaluate van-der Waals (Lennard-Jones) forces
void evalSyNonbondedEnergy(const AtomGraphSynthesis &poly_ag,
                           const StaticExclusionMaskSynthesis &poly_se,
                           PhaseSpaceSynthesis *poly_ps, ScoreCard *ecard,
                           NonbondedTask task,
                           PrecisionModel prec = PrecisionModel::DOUBLE,
                           EvaluateForce eval_elec_force = EvaluateForce::YES,
                           EvaluateForce eval_vdw_force = EvaluateForce::YES,
                           double clash_minimum_distance = 0.0, double clash_ratio = 0.0);
  
} // namespace energy
} // namespace stormm

#include "eval_synthesis.tpp"

#endif
