#include <vector>
#include "copyright.h"
#include "../../../src/Accelerator/gpu_details.h"
#ifdef STORMM_USE_HPC
#  include "../../../src/Accelerator/core_kernel_manager.h"
#  include "../../../src/Accelerator/hpc_config.h"
#  include "../../../src/MolecularMechanics/hpc_lambda_dynamics.h"
#  include "../../../src/MolecularMechanics/hpc_lambda_dynamics_wrapper.h"
#  include "../../../src/MolecularMechanics/hpc_minimization.h"
#else
#  include "../../../src/MolecularMechanics/dynamics.h"
#  include "../../../src/MolecularMechanics/minimization.h"
#endif
#include "../../../src/Accelerator/hybrid.h"
#include "../../../src/Chemistry/chemistry_enumerators.h"
#include "../../../src/DataTypes/common_types.h"
#include "../../../src/DataTypes/stormm_vector_types.h"
#include "../../../src/Math/vector_ops.h"
#include "../../../src/MolecularMechanics/minimization.h"
#include "../../../src/MolecularMechanics/mm_controls.h"
#include "../../../src/Namelists/input_transcript.h"
#include "../../../src/Namelists/nml_dynamics.h"
#include "../../../src/Namelists/nml_minimize.h"
#include "../../../src/Namelists/nml_pppm.h"
#include "../../../src/Namelists/nml_precision.h"
#include "../../../src/Namelists/nml_random.h"
#include "../../../src/Namelists/nml_remd.h"
#include "../../../src/Namelists/user_settings.h"
#include "../../../src/Potential/energy_enumerators.h"
#include "../../../src/Potential/local_exclusionmask.h"
#include "../../../src/Potential/static_exclusionmask.h"
#include "../../../src/Reporting/error_format.h"
#include "../../../src/Reporting/help_messages.h"
#include "../../../src/Reporting/present_energy.h"
#include "../../../src/Reporting/progress_bar.h"
#include "../../../src/Sampling/exchange_nexus.h"
#include "../../../src/Synthesis/atomgraph_synthesis.h"
#include "../../../src/Synthesis/phasespace_synthesis.h"
#include "../../../src/Synthesis/static_mask_synthesis.h"
#include "../../../src/Synthesis/systemcache.h"
#include "../../../src/Topology/atomgraph_enumerators.h"
#include "../../../src/Trajectory/phasespace.h"
#include "../../../src/Trajectory/thermostat.h"
#include "../../../src/UnitTesting/stopwatch.h"
#include "../../../src/UnitTesting/unit_test.h"
#include "setup.h"

using namespace stormm::card;
using namespace stormm::chemistry;
using namespace stormm::data_types;
using namespace stormm::display;
using namespace stormm::energy;
using namespace stormm::mm;
using namespace stormm::namelist;
using namespace stormm::random;
using namespace stormm::reporting;
using namespace stormm::restraints;
using namespace stormm::review;
using namespace stormm::sampling;
using namespace stormm::synthesis;
using namespace stormm::testing;
using namespace stormm::topology;
using namespace stormm::trajectory;
using namespace lambda_dyna_app::setup;

//-------------------------------------------------------------------------------------------------
// main
//-------------------------------------------------------------------------------------------------
int main(int argc, const char* argv[]) {

  // Check for a help message

  // Wall time tracking
  StopWatch timer("Timings for lambda_dynamics.stormm");
  const int file_parse_tm = timer.addCategory("File Parsing");
  const int gen_setup_tm  = timer.addCategory("Setup, General");
  const int min_setup_tm  = timer.addCategory("Setup, Minimization");
  const int dyn_setup_tm  = timer.addCategory("Setup, Dynamics");
  const int min_run_tm    = timer.addCategory("Run, Minimization");
  const int dyn_run_tm    = timer.addCategory("Run, Dynamics");
  const int download_tm   = timer.addCategory("GPU Data Download");
  const int output_tm     = timer.addCategory("Trajectory Output");

  // Engage the GPU
#ifdef STORMM_USE_HPC
  const HpcConfig gpu_config(ExceptionResponse::WARN);
  const std::vector<int> my_gpus = gpu_config.getGpuDevice(1);
  const GpuDetails gpu = gpu_config.getGpuInfo(my_gpus[0]);
  const Hybrid<int> array_to_trigger_gpu_mapping(1);

  // Notify user about lambda dynamics parity testing
  std::cout << "\n===================================================================\n";
  std::cout << "  LAMBDA DYNAMICS - PARITY TESTING MODE\n";
  std::cout << "===================================================================\n";
  std::cout << "This application tests the lambda-aware dynamics kernel.\n";
  std::cout << "All lambda values are set to 1.0 (fully coupled).\n";
  std::cout << "Results should be IDENTICAL to standard dynamics.stormm.cuda.\n";
  std::cout << "\n";
  std::cout << "Purpose: Verify correctness of launchGpuLambdaDynamicsStep()\n";
  std::cout << "         before using with variable lambda in GCMC sampling.\n";
  std::cout << "===================================================================\n\n";
#else
  const GpuDetails gpu = null_gpu;
#endif
  timer.assignTime(gen_setup_tm);

  // Parse the command line
  CommandLineParser clip("lambda_dynamics.stormm", "Lambda-aware molecular dynamics engine for GCMC parity testing.");
  clip.addStandardApplicationInputs();
  const std::vector<std::string> my_namelists = { "&files", "&minimize", "&dynamics", "&remd",
                                                  "&restraint", "&solvent", "&random", "&report",
                                                  "&precision" };
  clip.addControlBlocks(my_namelists);
  if (displayNamelistHelp(argc, argv, my_namelists) && clip.doesProgramExitOnHelp()) {
    return 0;
  }
  clip.parseUserInput(argc, argv);

  // Read information from the command line and initialize the UserSettings object
  UserSettings ui(clip, { "-pe", "-ce" });

  // Read topologies and coordinate files.  Assemble critical details about each system.
  SystemCache sc(ui.getFilesNamelistInfo(), ui.getExceptionBehavior(), MapRotatableGroups::NO,
                 ui.getPrintingPolicy());
  timer.assignTime(file_parse_tm);

  // Prepare a synthesis of systems from the user input.
  const std::vector<AtomGraph*> agv = sc.getTopologyPointer();

  // Preview the implicit solvent model.
  NeckGeneralizedBornTable ngb_tab;
  if (ui.getSolventPresence()) {
    const SolventControls& isvcon = ui.getSolventNamelistInfo();
    for (int i = 0; i < sc.getTopologyCount(); i++) {
      AtomGraph *ag = sc.getTopologyPointer(i);
      ag->setImplicitSolventModel(isvcon.getImplicitSolventModel());
    }
  }

  // Create the synthesis of systems, including exclusion masks and non-bonded work units as
  // necessary.
  int system_count = sc.getSystemCount();
  const PrecisionControls& preccon = ui.getPrecisionNamelistInfo();
  const DynamicsControls& dyncon = ui.getDynamicsNamelistInfo();
  const PPPMControls& pmecon = ui.getPPPMNamelistInfo();
  const ThermostatKind tstat_choice = dyncon.getThermostatKind();
  PhaseSpaceSynthesis poly_ps = sc.exportCoordinateSynthesis(preccon.getGlobalPosScalingBits(),
                                                             preccon.getVelocityScalingBits(),
                                                             preccon.getForceScalingBits());
  AtomGraphSynthesis poly_ag = sc.exportTopologySynthesis(gpu, ui.getExceptionBehavior());
  if (ui.getSolventPresence()) {
    const SolventControls& isvcon = ui.getSolventNamelistInfo();
    poly_ag.setImplicitSolventModel(isvcon.getImplicitSolventModel(), ngb_tab,
                                    isvcon.getPBRadiiSet(), isvcon.getExternalDielectric(),
                                    isvcon.getSaltConcentration(), ui.getExceptionBehavior());
  }
  StaticExclusionMaskSynthesis poly_se = createMaskSynthesis(sc, poly_ag);
  LocalExclusionMask lem(poly_ag);
  SynthesisCacheMap scmap(incrementingSeries(0, poly_ag.getSystemCount()), &sc, &poly_ag,
                          &poly_ps);

  // Initialization of trajectories for a given REMD process.
  if (ui.getRemdPresence()) {
    RemdControls remdcon = ui.getRemdNamelistInfo();
    int total_swap_count = remdcon.getTotalSwapCount();
    std::string remd_type = remdcon.getRemdType();
    int frequency_swaps_count = remdcon.getFrequencyOfSwaps();
    std::string swap_storage = remdcon.getSwapStore();
    std::string temp_distribution = remdcon.getTemperatureDistributionMethod();
    double exchange_probability = remdcon.getExchangeProbability();
    double tolerance = remdcon.getTolerance();
    int max_replicas = remdcon.getMaxReplicas();
    double low_temperature = remdcon.getLowTemperature();
    double high_temperature = remdcon.getHighTemperature();
    ExchangeNexus remd_a( system_count, total_swap_count, remd_type, frequency_swaps_count,
                          swap_storage, temp_distribution, exchange_probability, tolerance,
                          max_replicas, low_temperature, high_temperature);

    remd_a.setAtomGraphSynthesis(&poly_ag);
    std::vector<double> t_replicas = remd_a.getTempDistribution();
    std::vector<int> remd_top(t_replicas.size());
    for(int i = 0; i < remd_top.size(); ++i){
      remd_top[i] = 0;
    }
    scmap = SynthesisCacheMap(remd_top, &sc, &poly_ag, &poly_ps);
  }

  // Set the progress bar to system_count
  ProgressBar progress_bar;
  progress_bar.initialize(system_count);
  timer.assignTime(gen_setup_tm);

  // Perform minimizations as requested.
  if (ui.getMinimizePresence()) {
    const MinimizeControls mincon = ui.getMinimizeNamelistInfo();
#ifdef STORMM_USE_HPC
    switch (poly_ag.getUnitCellType()) {
    case UnitCellType::NONE:
      {
        // Isolated boundary conditions involve all-to-all interactions and open the door to
        // implicit solvent models.  However, the non-bonded work units for minimizations in such
        // a case are not the same as those for dynamics.
        InitializationTask ism_prep;
        const SolventControls& isvcon = ui.getSolventNamelistInfo();
        switch (isvcon.getImplicitSolventModel()) {
        case ImplicitSolventModel::NONE:
          ism_prep = InitializationTask::GENERAL_MINIMIZATION;
          break;
        case ImplicitSolventModel::HCT_GB:
        case ImplicitSolventModel::OBC_GB:
        case ImplicitSolventModel::OBC_GB_II:
        case ImplicitSolventModel::NECK_GB:
        case ImplicitSolventModel::NECK_GB_II:
          ism_prep = InitializationTask::GB_MINIMIZATION;
          break;
        }
        poly_ag.loadNonbondedWorkUnits(poly_se, ism_prep, 0, gpu);
      }
      break;
    case UnitCellType::ORTHORHOMBIC:
    case UnitCellType::TRICLINIC:
      rtErr("Minimization is not yet operational for periodic boundary conditions.", "main");
    }

    // Upload data to prepare for energy minimizations
    poly_ps.upload();
    poly_ag.upload();
    poly_se.upload();
    timer.assignTime(min_setup_tm);

    // Perform energy minimization
    ScoreCard emin = launchMinimization(poly_ag, poly_se, &poly_ps, mincon, gpu,
                                        preccon.getValenceMethod(),
                                        preccon.getEnergyScalingBits());
    emin.computeTotalEnergy(HybridTargetLevel::DEVICE, gpu);
    cudaDeviceSynchronize();
    timer.assignTime(min_run_tm);

    // Download the energies and also coordinates, to prepare for a CPU-based velocity seeding.
    emin.download();
    poly_ps.download();
    timer.assignTime(download_tm);
#else
    std::vector<ScoreCard> all_mme;
    all_mme.reserve(system_count);

    // Print out the stage for progress bar, reset bar
    std::cout << "Minimization" << std::endl;
    progress_bar.reset();

    switch (poly_ps.getUnitCellType()) {
    case UnitCellType::NONE:

      // Loop over all systems
      for (int i = 0; i < system_count; i++) {
        progress_bar.update();
        const int icache_top = scmap.getTopologyCacheIndex(i);
        PhaseSpace ps = poly_ps.exportSystem(i);
        AtomGraph *ag = sc.getTopologyPointer(icache_top);
        const RestraintApparatus& ra = sc.getRestraints(i);
        all_mme.emplace_back(minimize(&ps, *ag, ra, sc.getSystemStaticMask(i), mincon));
        poly_ps.importSystem(ps, i);
      }
      timer.assignTime(min_run_tm);
      break;
    case UnitCellType::ORTHORHOMBIC:
    case UnitCellType::TRICLINIC:
      rtErr("Minimization is not yet operational for periodic boundary conditions.", "main");
    }

    // Terminate the progress bar's line for the rest of the program to print correctly
    std::cout << std::endl;
#endif
    // Print restart files from energy minimization
    if (mincon.getCheckpointProduction()) {
      for (int i = 0; i < system_count; i++) {
        const PhaseSpace ps = poly_ps.exportSystem(i);
        ps.exportToFile(sc.getCheckpointName(i), 0.0, TrajectoryKind::POSITIONS,
                        CoordinateFileKind::AMBER_ASCII_RST, ui.getPrintingPolicy());
      }
      timer.assignTime(output_tm);
    }
  }

  // Kick-start dynamics if necessary.  A CPU-based routine is used for this, as it will involve a
  // a great deal of code to get it working on the GPU.  This will modify the thermostat's random
  // state.  Create a dummy thermostat to ensure that the same random numbers are not used to seed
  // velocities and then perform a first stochastic velocity modification.
  DynamicsControls mod_dyncon = dyncon;
  mod_dyncon.setThermostatSeed(dyncon.getThermostatSeed() + 715829320);
  mod_dyncon.setThermostatKind("langevin");
  Thermostat tst(poly_ag, mod_dyncon, sc, incrementingSeries(0, sc.getSystemCount()), gpu);
  velocityKickStart(&poly_ps, poly_ag, &tst, mod_dyncon, preccon.getValenceMethod(),
                    EnforceExactTemperature::YES);
#ifndef STORMM_USE_HPC
  // In order to perform CPU-based dynamics on systems in implicit solvent, thermostats
  // must be created for each system and persist throughout the entire simulation.  If
  // thermostats are created anew for each epoch of the replica exchange, they would need
  // to be initiated with different random seeds each time.  The most efficient way is to
  // create these thermostats once and let them keep charging forward.
  std::vector<Thermostat> tst_vec;
  tst_vec.reserve(system_count);
  for (int i = 0; i < system_count; i++) {
    const AtomGraph* ag = poly_ps.getSystemTopologyPointer(i);
    tst_vec.emplace_back(ag->getAtomCount(), dyncon.getThermostatKind(), 298.15, 298.15,
                         dyncon.getThermostatEvolutionStart(),
                         dyncon.getThermostatEvolutionEnd(), PrecisionModel::SINGLE,
                         dyncon.getThermostatSeed() + i);
    tst_vec.back().setGeometryConstraints(dyncon.constrainGeometry());
    tst_vec.back().setRattleTolerance(dyncon.getRattleTolerance());
    tst_vec.back().setRattleIterations(dyncon.getRattleIterations());
  }
#endif

  // Run dynamics
  if (ui.getDynamicsPresence()) {
    const DynamicsControls dyncon = ui.getDynamicsNamelistInfo();
#ifdef STORMM_USE_HPC
    switch (poly_ag.getUnitCellType()) {
    case UnitCellType::NONE:
      {
        // Isolated boundary conditions involve all-to-all interactions and open the door to
        // implicit solvent models.  However, the non-bonded work units for minimizations in such
        // a case are not the same as those for dynamics.
        InitializationTask ism_prep;
        const SolventControls& isvcon = ui.getSolventNamelistInfo();
        switch (isvcon.getImplicitSolventModel()) {
        case ImplicitSolventModel::NONE:
          switch (dyncon.getThermostatKind()) {
          case ThermostatKind::NONE:
          case ThermostatKind::BERENDSEN:
            ism_prep = InitializationTask::GENERAL_DYNAMICS;
            break;
          case ThermostatKind::ANDERSEN:
          case ThermostatKind::LANGEVIN:
            ism_prep = InitializationTask::LANGEVIN_DYNAMICS;
            break;
          }
          break;
        case ImplicitSolventModel::HCT_GB:
        case ImplicitSolventModel::OBC_GB:
        case ImplicitSolventModel::OBC_GB_II:
        case ImplicitSolventModel::NECK_GB:
        case ImplicitSolventModel::NECK_GB_II:
          switch (dyncon.getThermostatKind()) {
          case ThermostatKind::NONE:
          case ThermostatKind::BERENDSEN:
            ism_prep = InitializationTask::GB_DYNAMICS;
            break;
          case ThermostatKind::ANDERSEN:
          case ThermostatKind::LANGEVIN:
            ism_prep = InitializationTask::GB_LANGEVIN_DYNAMICS;
            break;
          }
          break;
        }

        // Build the static exclusion mask synthesis, if it has not been built already.  Otherwise,
        // load non-bonded work units appropriate for dynamics as opposed to energy minimization.
        // The difference lies in array initialization instructions assigned to each work unit.
        poly_ag.loadNonbondedWorkUnits(poly_se, ism_prep, dyncon.getThermostatCacheDepth(), gpu);
      }
      break;
    case UnitCellType::ORTHORHOMBIC:
    case UnitCellType::TRICLINIC:
      rtErr("Minimization and dynamics are not yet operational for periodic boundary conditions.",
            "main");
    }

    // Upload data to prepare for dynamics.  If energy minimizations were performed on the GPU, the
    // coordinates were downloaded afterward in order to do the velocity kick-start on the host.
    // The most current coordinates must therefore be uploaded to the GPU.
    tst = Thermostat(poly_ag, dyncon, sc, incrementingSeries(0, sc.getSystemCount()), gpu);
    tst.uploadPartitions();
    poly_ps.upload();
    poly_ag.upload();
    poly_se.upload();
    timer.assignTime(dyn_setup_tm);

    // ========================================================================================
    // LAMBDA DYNAMICS: Call lambda-aware dynamics function instead of standard launchDynamics
    // ========================================================================================
    // NOTE: This is the KEY difference from dynamics.stormm.cuda
    // launchLambdaDynamics (PARITY TESTING MODE):
    // 1. Sets all lambda values to 1.0 (fully coupled, no force scaling)
    // 2. Uses launchGpuLambdaDynamicsStep for integration
    // 3. Results should be IDENTICAL to standard launchDynamics
    // 4. Validates correctness of lambda-aware kernel before GCMC use
    // 5. Otherwise maintains identical behavior to standard dynamics

    ScoreCard edyn = launchLambdaDynamics(poly_ag, poly_se, &tst, &poly_ps, dyncon, sc, scmap, gpu,
                                          preccon.getValenceMethod(), preccon.getNonbondedMethod(),
                                          preccon.getEnergyScalingBits(), &timer, "lambda_dynamics");

    cudaDeviceSynchronize();
    timer.assignTime(dyn_run_tm);
    edyn.download();
    timer.assignTime(download_tm);
#else
    const int nstep = dyncon.getStepCount();
    const int ntpr  = dyncon.getDiagnosticPrintFrequency();
    ScoreCard edyn(system_count, ((nstep + ntpr - 1) / ntpr) + 1, preccon.getEnergyScalingBits());

    // CPU path not implemented for lambda dynamics
    rtErr("Lambda dynamics is only supported with CUDA/GPU acceleration. "
          "Please rebuild STORMM with CUDA enabled (cmake -DSTORMM_ENABLE_CUDA=ON ...).", "main");
#endif

    // Turn the energy tracking data into an output report
    createDiagnosticReport(edyn, scmap, ui);

    // Print restart files from dynamics
#ifdef STORMM_USE_HPC
    poly_ps.download();
#endif
    // Progress Bar reset before loop, and purpose
    std::cout << "Exporting data" << std::endl;
    progress_bar.reset();
    for (int i = 0; i < system_count; i++) {
      progress_bar.update(); // Update progress bar at the beginning of the loop
      const PhaseSpace ps = poly_ps.exportSystem(i);
      ps.exportToFile(sc.getCheckpointName(i), 0.0, TrajectoryKind::POSITIONS,
                      CoordinateFileKind::AMBER_ASCII_RST, ui.getPrintingPolicy());
    }
    // At the end of the progress bar, endl for the rest of the program
    std::cout << std::endl;
    timer.assignTime(output_tm);
  }

  // Summarize the results
  timer.assignTime(output_tm);
  timer.printResults();

  return 0;
}
