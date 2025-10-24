// -*-c++-*-
#ifndef STORMM_HPC_LAMBDA_DYNAMICS_WRAPPER_H
#define STORMM_HPC_LAMBDA_DYNAMICS_WRAPPER_H

#include <string>
#include "copyright.h"
#include "Accelerator/core_kernel_manager.h"
#include "Accelerator/gpu_details.h"
#include "Constants/behavior.h"
#include "Constants/fixed_precision.h"
#include "MolecularMechanics/mm_controls.h"
#include "Namelists/nml_dynamics.h"
#include "Numerics/numeric_enumerators.h"
#include "Potential/cacheresource.h"
#include "Potential/energy_enumerators.h"
#include "Potential/scorecard.h"
#include "Synthesis/atomgraph_synthesis.h"
#include "Synthesis/implicit_solvent_workspace.h"
#include "Synthesis/nonbonded_workunit.h"
#include "Synthesis/synthesis_enumerators.h"
#include "Synthesis/synthesis_cache_map.h"
#include "Synthesis/systemcache.h"
#include "Synthesis/valence_workunit.h"
#include "Trajectory/motion_sweeper.h"
#include "Trajectory/thermostat.h"
#include "UnitTesting/stopwatch.h"

namespace stormm {
namespace mm {

using card::CoreKlManager;
using card::GpuDetails;
using constants::PrecisionModel;
using energy::CacheResource;
using energy::ScoreCard;
using namelist::DynamicsControls;
using numerics::AccumulationMethod;
using numerics::default_energy_scale_bits;
using synthesis::AtomGraphSynthesis;
using synthesis::ImplicitSolventWorkspace;
using synthesis::maximum_valence_work_unit_atoms;
using synthesis::NbwuKind;
using synthesis::PhaseSpaceSynthesis;
using synthesis::small_block_max_atoms;
using synthesis::StaticExclusionMaskSynthesis;
using synthesis::SynthesisCacheMap;
using synthesis::SystemCache;
using testing::StopWatch;
using trajectory::MotionSweeper;
using trajectory::Thermostat;

/// \brief Launch lambda-aware molecular dynamics for parity testing
///
/// This function runs lambda-scaled MD with all lambda values set to 1.0 (fully coupled).
/// The purpose is to verify that the lambda-aware dynamics kernel produces identical
/// results to standard MD when all particles are fully coupled.
///
/// Key features:
/// 1. All lambda values set to 1.0 (fully coupled - no scaling)
/// 2. Uses launchGpuLambdaDynamicsStep for integration
/// 3. All atoms included in coupled list (lambda > 0.02)
/// 4. Identical behavior to standard dynamics expected
/// 5. Validates correctness of lambda-aware force calculations
///
/// This serves as a critical validation step before using the lambda dynamics
/// kernel for actual GCMC sampling where lambda values vary between 0 and 1.
///
/// For actual lambda-scaled simulations (variable lambda values), use the
/// GCMC sampler applications which manage lambda values dynamically.
///
/// \param poly_ag        The topology synthesis spanning all systems
/// \param poly_se        Static exclusion masks spanning all systems
/// \param tst            The thermostat controlling integration and temperature regulation
/// \param poly_ps        Coordinates (positions, velocities, and forces) for all systems
/// \param dyncon         User-provided dynamics control input
/// \param sysc           The original cache of systems provided by user input
/// \param syscmap        A map linking the systems in the cache to synthesis instances
/// \param gpu            Details of the GPU that will run calculations
/// \param valence_prec   The precision model for valence calculations (default SINGLE)
/// \param nonbond_prec   Arithmetic precision for non-bonded calculations (default SINGLE)
/// \param energy_bits    The number of bits with which energy will be tracked
/// \param timer          Wall time tracker (optional)
/// \param task_name      Developer-specified string for error tracing (optional)
/// \return               ScoreCard with energy tracking data from the dynamics run
ScoreCard launchLambdaDynamics(const AtomGraphSynthesis &poly_ag,
                               const StaticExclusionMaskSynthesis &poly_se,
                               Thermostat *tst,
                               PhaseSpaceSynthesis *poly_ps,
                               const DynamicsControls &dyncon,
                               const SystemCache &sysc,
                               const SynthesisCacheMap &syscmap,
                               const GpuDetails &gpu,
                               PrecisionModel valence_prec = PrecisionModel::SINGLE,
                               PrecisionModel nonbond_prec = PrecisionModel::SINGLE,
                               int energy_bits = default_energy_scale_bits,
                               StopWatch *timer = nullptr,
                               const std::string &task_name = std::string(""));

} // namespace mm
} // namespace stormm

#endif // STORMM_HPC_LAMBDA_DYNAMICS_WRAPPER_H
