// Minimal NCMC GCMC test application with lambda dynamics kernel integration
// Uses NCMC (Nonequilibrium Candidate Monte Carlo) protocols for improved acceptance
// System-wide NCMC GCMC only (no sphere mode)

#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include "copyright.h"

// Command line parsing
#include "Namelists/command_line_parser.h"
#include "Namelists/namelist_emulator.h"

// Core includes
#ifdef STORMM_USE_HPC
#  include "Accelerator/gpu_details.h"
#  include "Accelerator/hpc_config.h"
#  include "Accelerator/hybrid.h"
#endif

// Topology and trajectory
#include "Topology/atomgraph.h"
#include "Topology/atomgraph_enumerators.h"
#include "Trajectory/coordinateframe.h"
#include "Trajectory/coordinate_series.h"
#include "Trajectory/phasespace.h"
#include "Trajectory/thermostat.h"
#include "Trajectory/trajectory_enumerators.h"

// Potential
#include "Potential/static_exclusionmask.h"

// GCMC Sampling
#include "Sampling/gcmc_sampler.h"

// Reporting
#include "Reporting/error_format.h"

using namespace stormm;
using namespace stormm::card;
using namespace stormm::energy;
using namespace stormm::errors;
using namespace stormm::namelist;
using namespace stormm::sampling;
using namespace stormm::topology;
using namespace stormm::trajectory;

//-------------------------------------------------------------------------------------------------
// Main function
//-------------------------------------------------------------------------------------------------
int main(int argc, const char* argv[]) {

  // GPU setup
#ifdef STORMM_USE_HPC
  const HpcConfig gpu_config(ExceptionResponse::WARN);
  const std::vector<int> my_gpus = gpu_config.getGpuDevice(1);
  const GpuDetails gpu = (my_gpus.empty()) ? null_gpu : gpu_config.getGpuInfo(my_gpus[0]);
  if (!my_gpus.empty()) {
    const Hybrid<int> array_to_trigger_gpu_mapping(1);
  }
#else
  const GpuDetails gpu = null_gpu;
#endif

  // Parse command line
  CommandLineParser clip("gcmc_ncmc_test.stormm",
                        "NCMC GCMC test with lambda dynamics kernel (system-wide NCMC GCMC only)");

  // Add standard application inputs
  clip.addStandardApplicationInputs({ "-p", "-c", "-o" });

  // Get the namelist emulator to add custom keywords
  NamelistEmulator *t_nml = clip.getNamelistPointer();

  // Fragment files
  t_nml->addKeyword("--fragment-prmtop", NamelistType::STRING, "");
  t_nml->addHelp("--fragment-prmtop", "Fragment topology file (required)");

  t_nml->addKeyword("--fragment-inpcrd", NamelistType::STRING, "");
  t_nml->addHelp("--fragment-inpcrd", "Fragment coordinate file (required)");

  // GCMC parameters
  t_nml->addKeyword("-n", NamelistType::INTEGER, "1000");
  t_nml->addHelp("-n", "Number of GCMC cycles (default: 1000)");

  t_nml->addKeyword("--moves", NamelistType::INTEGER, "1000");
  t_nml->addHelp("--moves", "Number of GCMC cycles (alias for -n)");

  t_nml->addKeyword("--temp", NamelistType::REAL, "300.0");
  t_nml->addHelp("--temp", "Temperature in Kelvin (default: 300.0)");

  // NOTE: Recommended ghost limit is 20-50 molecules due to memory constraints from
  // temporary GPU allocations in lambda dynamics kernels. Large values (100+) may
  // cause cudaFreeHost failures from pinned memory fragmentation during long runs.
  t_nml->addKeyword("--nghost", NamelistType::INTEGER, "50");
  t_nml->addHelp("--nghost", "Number of ghost molecules (default: 50, recommended: 20-50)");

  t_nml->addKeyword("-b", NamelistType::REAL, "5.0");
  t_nml->addHelp("-b", "Adams B parameter (default: 5.0)");

  t_nml->addKeyword("--bvalue", NamelistType::REAL, "5.0");
  t_nml->addHelp("--bvalue", "Adams B parameter (alias for -b)");

  t_nml->addKeyword("--mu-ex", NamelistType::REAL, "0.0");
  t_nml->addHelp("--mu-ex", "Excess chemical potential in kcal/mol (default: 0.0, only used if B not set)");

  t_nml->addKeyword("--standard-volume", NamelistType::REAL, "30.0");
  t_nml->addHelp("--standard-volume", "Standard volume in Angstrom^3 (default: 30.0)");

  // NCMC-specific parameters
  t_nml->addKeyword("--npert", NamelistType::INTEGER, "50");
  t_nml->addHelp("--npert", "Number of lambda perturbation steps for NCMC (default: 50)");

  t_nml->addKeyword("--nprop", NamelistType::INTEGER, "2");
  t_nml->addHelp("--nprop", "MD propagation steps per perturbation (default: 2)");

  t_nml->addKeyword("--timestep", NamelistType::REAL, "2.0");
  t_nml->addHelp("--timestep", "MD timestep in fs (default: 2.0)");

  // Parse command line
  clip.parseUserInput(argc, argv);

  // Get values
  std::string base_prmtop = t_nml->getStringValue("-p");
  std::string base_inpcrd = t_nml->getStringValue("-c");
  std::string output_prefix = t_nml->getStringValue("-o");

  std::string fragment_prmtop = t_nml->getStringValue("--fragment-prmtop");
  std::string fragment_inpcrd = t_nml->getStringValue("--fragment-inpcrd");

  int n_moves = std::max(t_nml->getIntValue("-n"), t_nml->getIntValue("--moves"));
  double temperature = t_nml->getRealValue("--temp");
  int n_ghosts = t_nml->getIntValue("--nghost");
  double b_value = std::max(t_nml->getRealValue("-b"), t_nml->getRealValue("--bvalue"));
  double mu_ex = t_nml->getRealValue("--mu-ex");
  double standard_volume = t_nml->getRealValue("--standard-volume");

  // NCMC-specific parameters
  int n_pert_steps = t_nml->getIntValue("--npert");
  int n_prop_steps = t_nml->getIntValue("--nprop");
  double timestep = t_nml->getRealValue("--timestep");

  // Validation
  if (fragment_prmtop.empty()) {
    rtErr("--fragment-prmtop is required", "gcmc_ncmc_test_runner");
  }
  if (fragment_inpcrd.empty()) {
    rtErr("--fragment-inpcrd is required", "gcmc_ncmc_test_runner");
  }
  if (output_prefix.empty()) {
    output_prefix = "gcmc_ncmc_test_output";
  }

  // Print configuration
  std::cout << "\n========================================\n";
  std::cout << "NCMC GCMC Test Runner (Lambda Dynamics)\n";
  std::cout << "========================================\n";
  std::cout << "Mode: System-wide NCMC GCMC\n";
  std::cout << "\nInput Files:\n";
  if (!base_prmtop.empty()) {
    std::cout << "  Base topology:     " << base_prmtop << "\n";
    std::cout << "  Base coordinates:  " << base_inpcrd << "\n";
  } else {
    std::cout << "  Fragment-only mode (no base system)\n";
  }
  std::cout << "  Fragment topology: " << fragment_prmtop << "\n";
  std::cout << "  Fragment coords:   " << fragment_inpcrd << "\n";
  std::cout << "\nGCMC Parameters:\n";
  std::cout << "  GCMC cycles:       " << n_moves << "\n";
  std::cout << "  Temperature:       " << temperature << " K\n";
  std::cout << "  Ghost molecules:   " << n_ghosts << "\n";
  std::cout << "  Adams B:           " << b_value << "\n";
  std::cout << "  Mu excess:         " << mu_ex << " kcal/mol\n";
  std::cout << "  Standard volume:   " << standard_volume << " A^3\n";
  std::cout << "\nNCMC Protocol:\n";
  std::cout << "  Perturbation steps: " << n_pert_steps << "\n";
  std::cout << "  Prop steps/pert:    " << n_prop_steps << "\n";
  std::cout << "  Total MD steps:     " << (n_pert_steps * n_prop_steps) << " per move\n";
  std::cout << "  Timestep:           " << timestep << " fs\n";
  std::cout << "\nOutput:\n";
  std::cout << "  Prefix:            " << output_prefix << "\n";
  std::cout << "========================================\n\n";

  try {
    // Load topologies
    std::cout << "Loading topologies...\n";
    AtomGraph fragment(fragment_prmtop, ExceptionResponse::WARN);

    AtomGraph combined_topology;
    int base_molecule_count = 0;

    if (!base_prmtop.empty()) {
      // System with base + ghosts
      AtomGraph base_topology(base_prmtop, ExceptionResponse::WARN);
      base_molecule_count = base_topology.getMoleculeCount();

      std::cout << "  Base system:    " << base_topology.getAtomCount() << " atoms, "
                << base_molecule_count << " molecules\n";
      std::cout << "  Fragment:       " << fragment.getAtomCount() << " atoms\n";
      std::cout << "  Building combined topology with " << n_ghosts << " ghosts...\n";

      combined_topology = buildTopologyWithGhosts(base_topology, fragment, n_ghosts);
    } else {
      // Fragment-only mode
      std::cout << "  Fragment:       " << fragment.getAtomCount() << " atoms\n";
      std::cout << "  Building fragment-only topology with " << n_ghosts << " copies...\n";

      // Create empty base topology
      AtomGraph empty_base;
      combined_topology = buildTopologyWithGhosts(empty_base, fragment, n_ghosts);
    }

    std::cout << "  Combined:       " << combined_topology.getAtomCount() << " atoms, "
              << combined_topology.getMoleculeCount() << " molecules\n";

    // Identify ghost molecules
    std::cout << "\nIdentifying ghost molecules...\n";
    GhostMoleculeMetadata ghost_metadata = identifyGhostMolecules(
        combined_topology, base_molecule_count, n_ghosts);

    std::cout << "  Base molecules:  " << ghost_metadata.base_molecule_count << "\n";
    std::cout << "  Ghost molecules: " << ghost_metadata.n_ghost_molecules << "\n";
    std::cout << "  Base atoms:      " << ghost_metadata.base_atom_count << "\n";

    // Create phase space
    std::cout << "\nInitializing phase space...\n";
    PhaseSpace ps;

    // Load fragment coordinates
    PhaseSpace fragment_ps(fragment_inpcrd, CoordinateFileKind::AMBER_INPCRD);
    const int fragment_natoms = fragment.getAtomCount();

    if (fragment_ps.getAtomCount() != fragment_natoms) {
      rtErr("Fragment coordinate count mismatch", "gcmc_ncmc_test_runner");
    }

    const int total_atoms = combined_topology.getAtomCount();
    std::vector<double> combined_xcrd(total_atoms, 0.0);
    std::vector<double> combined_ycrd(total_atoms, 0.0);
    std::vector<double> combined_zcrd(total_atoms, 0.0);
    std::vector<double> box_dims(6);

    if (!base_inpcrd.empty()) {
      // Load base coordinates
      PhaseSpace base_ps(base_inpcrd, CoordinateFileKind::AMBER_INPCRD);

      if (base_ps.getAtomCount() != ghost_metadata.base_atom_count) {
        rtErr("Base coordinate count mismatch", "gcmc_ncmc_test_runner");
      }

      // Copy base coordinates
      const PhaseSpaceReader base_psr = base_ps.data();
      for (int i = 0; i < ghost_metadata.base_atom_count; i++) {
        combined_xcrd[i] = base_psr.xcrd[i];
        combined_ycrd[i] = base_psr.ycrd[i];
        combined_zcrd[i] = base_psr.zcrd[i];
      }

      // Copy box dimensions from base (or use default if not present)
      const double box_threshold = 1.0;  // If box < 1 Å, it's not set
      if (base_psr.boxdim[0] > box_threshold &&
          base_psr.boxdim[1] > box_threshold &&
          base_psr.boxdim[2] > box_threshold) {
        // Use box from base coordinate file
        for (int i = 0; i < 6; i++) {
          box_dims[i] = base_psr.boxdim[i];
        }
      } else {
        // Base has no box - use default 25 Å cubic box for testing
        const double default_box_size = 25.0;
        box_dims[0] = default_box_size;
        box_dims[1] = default_box_size;
        box_dims[2] = default_box_size;
        box_dims[3] = 90.0;
        box_dims[4] = 90.0;
        box_dims[5] = 90.0;
        std::cout << "  Note: Base system has no box dimensions, using default "
                  << default_box_size << " Å cubic box\n";
      }

      // Place ghost molecules from fragment coordinates
      const PhaseSpaceReader frag_psr = fragment_ps.data();
      for (int ghost_idx = 0; ghost_idx < ghost_metadata.n_ghost_molecules; ghost_idx++) {
        const int atom_offset = ghost_metadata.base_atom_count + (ghost_idx * fragment_natoms);
        for (int i = 0; i < fragment_natoms; i++) {
          combined_xcrd[atom_offset + i] = frag_psr.xcrd[i];
          combined_ycrd[atom_offset + i] = frag_psr.ycrd[i];
          combined_zcrd[atom_offset + i] = frag_psr.zcrd[i];
        }
      }
    } else {
      // Fragment-only mode: place all copies from fragment coordinates
      const double box_size = 25.0;  // Default 25 Å cubic box for testing
      box_dims[0] = box_size;
      box_dims[1] = box_size;
      box_dims[2] = box_size;
      box_dims[3] = 90.0;
      box_dims[4] = 90.0;
      box_dims[5] = 90.0;
      std::cout << "  Fragment-only mode: using default " << box_size << " Å cubic box\n";

      const PhaseSpaceReader frag_psr = fragment_ps.data();
      for (int ghost_idx = 0; ghost_idx < ghost_metadata.n_ghost_molecules; ghost_idx++) {
        const int atom_offset = ghost_idx * fragment_natoms;
        for (int i = 0; i < fragment_natoms; i++) {
          combined_xcrd[atom_offset + i] = frag_psr.xcrd[i];
          combined_ycrd[atom_offset + i] = frag_psr.ycrd[i];
          combined_zcrd[atom_offset + i] = frag_psr.zcrd[i];
        }
      }
    }

    // Create PhaseSpace and fill with coordinates
    ps = PhaseSpace(total_atoms);
    ps.fill(combined_xcrd, combined_ycrd, combined_zcrd,
            TrajectoryKind::POSITIONS, CoordinateCycle::WHITE, 0, box_dims);

    std::cout << "  Phase space initialized with " << ps.getAtomCount() << " atoms\n";

    // Create exclusion mask
    std::cout << "\nBuilding exclusion mask...\n";
    StaticExclusionMask exclusions(&combined_topology);
    std::cout << "  Exclusion mask built\n";

    // Create thermostat
    std::cout << "\nInitializing thermostat...\n";
    Thermostat thermostat(combined_topology, ThermostatKind::LANGEVIN, temperature);
    thermostat.setTimeStep(timestep / 1000.0);  // Convert fs to ps
    thermostat.upload();  // Upload all thermostat arrays to GPU (including random cache)
    std::cout << "  Langevin thermostat at " << temperature << " K\n";

    // Create NCMC GCMC sampler
    std::cout << "\nCreating NCMC GCMC sampler...\n";
    std::string ghost_file = output_prefix + "_ghosts.txt";
    std::string log_file = output_prefix + "_gcmc.log";

    NCMCSystemSampler sampler(
        &combined_topology,
        &ps,
        &exclusions,
        &thermostat,
        temperature,
        ghost_metadata,
        n_pert_steps,      // Number of lambda perturbation steps
        n_prop_steps,      // MD steps per perturbation
        timestep,          // Integration timestep (fs)
        {},                // Empty lambda schedule (use default)
        false,             // Don't record trajectory
        mu_ex,
        standard_volume,
        b_value,           // adams parameter
        0.0,               // adams_shift
        ImplicitSolventModel::NONE,
        "HOH",             // resname
        ghost_file,
        log_file);

    std::cout << "  NCMCSystemSampler created (lambda dynamics kernel)\n";
    std::cout << "  NCMC protocol: " << n_pert_steps << " perturbations x "
              << n_prop_steps << " MD steps = "
              << (n_pert_steps * n_prop_steps) << " total MD steps per move\n";

    // Open occupancy output file
    std::string occupancy_file = output_prefix + "_occupancy.dat";
    std::ofstream occ_out(occupancy_file);
    if (!occ_out.is_open()) {
      rtErr("Cannot open occupancy file: " + occupancy_file, "gcmc_ncmc_test_runner");
    }
    occ_out << "# Cycle  Active_Molecules\n";

    // Main NCMC GCMC loop
    std::cout << "\n========================================\n";
    std::cout << "Starting NCMC GCMC simulation\n";
    std::cout << "========================================\n";

    for (int cycle = 0; cycle < n_moves; cycle++) {
      // Attempt NCMC GCMC move (insertion or deletion)
      // NCMC protocol includes MD propagation during lambda changes
      bool accepted = sampler.runGCMCCycle();

      // Write occupancy every cycle
      int n_active = sampler.getActiveCount();
      occ_out << cycle << "  " << n_active << "\n";

      // Progress output
      if (cycle % 100 == 0) {
        std::cout << "Cycle " << cycle << "/" << n_moves
                  << "  Active: " << n_active
                  << "/" << n_ghosts << "\r" << std::flush;
      }
    }

    std::cout << "\n\n========================================\n";
    std::cout << "NCMC GCMC simulation complete\n";
    std::cout << "========================================\n";

    // Final statistics
    const GCMCStatistics& stats = sampler.getStatistics();
    int final_active = sampler.getActiveCount();

    std::cout << "\nFinal Statistics:\n";
    std::cout << "  Total moves:       " << stats.n_moves << "\n";
    std::cout << "  Insertions:        " << stats.n_inserts << "\n";
    std::cout << "  Deletions:         " << stats.n_deletes << "\n";
    std::cout << "  Accepted:          " << stats.n_accepted << "\n";
    std::cout << "  Acceptance rate:   " << std::fixed << std::setprecision(2)
              << (100.0 * stats.n_accepted / stats.n_moves) << "%\n";
    std::cout << "  Final occupancy:   " << final_active << "/" << n_ghosts << "\n";
    std::cout << "  Final fraction:    " << std::fixed << std::setprecision(3)
              << (static_cast<double>(final_active) / n_ghosts) << "\n";

    // Close output file
    occ_out.close();

    std::cout << "\nOutput files:\n";
    std::cout << "  Occupancy:  " << occupancy_file << "\n";
    std::cout << "  Ghosts:     " << ghost_file << "\n";
    std::cout << "  Log:        " << log_file << "\n";
    std::cout << "\n========================================\n";

  } catch (const std::exception& e) {
    std::cerr << "\nERROR: " << e.what() << "\n";
    return 1;
  }

  return 0;
}
