// Minimal GCMC test application with lambda dynamics kernel integration
// Supports system-wide instant GCMC only (no sphere mode, no NCMC)

#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <random>
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
  CommandLineParser clip("gcmc_test.stormm",
                        "Minimal GCMC test with lambda dynamics kernel (system-wide instant GCMC only)");

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

  // MD parameters
  t_nml->addKeyword("--timestep", NamelistType::REAL, "2.0");
  t_nml->addHelp("--timestep", "MD timestep in fs (default: 2.0)");

  t_nml->addKeyword("--md-steps", NamelistType::INTEGER, "100");
  t_nml->addHelp("--md-steps", "MD steps between GCMC moves (default: 100)");

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
  double timestep = t_nml->getRealValue("--timestep");
  int md_steps = t_nml->getIntValue("--md-steps");

  // Validation
  if (fragment_prmtop.empty()) {
    rtErr("--fragment-prmtop is required", "gcmc_test_runner");
  }
  if (fragment_inpcrd.empty()) {
    rtErr("--fragment-inpcrd is required", "gcmc_test_runner");
  }
  if (output_prefix.empty()) {
    output_prefix = "gcmc_test_output";
  }

  // Print configuration
  std::cout << "\n========================================\n";
  std::cout << "GCMC Test Runner (Lambda Dynamics)\n";
  std::cout << "========================================\n";
  std::cout << "Mode: System-wide instant GCMC\n";
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
  std::cout << "\nMD Parameters:\n";
  std::cout << "  Timestep:          " << timestep << " fs\n";
  std::cout << "  MD steps/cycle:    " << md_steps << "\n";
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
      rtErr("Fragment coordinate count mismatch", "gcmc_test_runner");
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
        rtErr("Base coordinate count mismatch", "gcmc_test_runner");
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
      // FIX: Randomly distribute ghosts throughout periodic box to avoid stacking
      const PhaseSpaceReader frag_psr = fragment_ps.data();

      // Calculate center of mass of fragment for translation
      double frag_com_x = 0.0, frag_com_y = 0.0, frag_com_z = 0.0;
      for (int i = 0; i < fragment_natoms; i++) {
        frag_com_x += frag_psr.xcrd[i];
        frag_com_y += frag_psr.ycrd[i];
        frag_com_z += frag_psr.zcrd[i];
      }
      frag_com_x /= fragment_natoms;
      frag_com_y /= fragment_natoms;
      frag_com_z /= fragment_natoms;

      // Random number generator for ghost placement
      std::random_device rd;
      std::mt19937 gen(rd());
      std::uniform_real_distribution<> dis(0.0, 1.0);

      for (int ghost_idx = 0; ghost_idx < ghost_metadata.n_ghost_molecules; ghost_idx++) {
        // Random position in periodic box
        const double rand_x = dis(gen) * box_dims[0];
        const double rand_y = dis(gen) * box_dims[1];
        const double rand_z = dis(gen) * box_dims[2];

        const int atom_offset = ghost_metadata.base_atom_count + (ghost_idx * fragment_natoms);
        for (int i = 0; i < fragment_natoms; i++) {
          // Translate fragment to random position (preserving internal geometry)
          combined_xcrd[atom_offset + i] = frag_psr.xcrd[i] - frag_com_x + rand_x;
          combined_ycrd[atom_offset + i] = frag_psr.ycrd[i] - frag_com_y + rand_y;
          combined_zcrd[atom_offset + i] = frag_psr.zcrd[i] - frag_com_z + rand_z;
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

      // FIX: Randomly distribute ghosts throughout periodic box to avoid stacking
      const PhaseSpaceReader frag_psr = fragment_ps.data();

      // Calculate center of mass of fragment
      double frag_com_x = 0.0, frag_com_y = 0.0, frag_com_z = 0.0;
      for (int i = 0; i < fragment_natoms; i++) {
        frag_com_x += frag_psr.xcrd[i];
        frag_com_y += frag_psr.ycrd[i];
        frag_com_z += frag_psr.zcrd[i];
      }
      frag_com_x /= fragment_natoms;
      frag_com_y /= fragment_natoms;
      frag_com_z /= fragment_natoms;

      // Random number generator
      std::random_device rd;
      std::mt19937 gen(rd());
      std::uniform_real_distribution<> dis(0.0, 1.0);

      for (int ghost_idx = 0; ghost_idx < ghost_metadata.n_ghost_molecules; ghost_idx++) {
        // Random position in periodic box
        const double rand_x = dis(gen) * box_dims[0];
        const double rand_y = dis(gen) * box_dims[1];
        const double rand_z = dis(gen) * box_dims[2];

        const int atom_offset = ghost_idx * fragment_natoms;
        for (int i = 0; i < fragment_natoms; i++) {
          // Translate fragment to random position (preserving internal geometry)
          combined_xcrd[atom_offset + i] = frag_psr.xcrd[i] - frag_com_x + rand_x;
          combined_ycrd[atom_offset + i] = frag_psr.ycrd[i] - frag_com_y + rand_y;
          combined_zcrd[atom_offset + i] = frag_psr.zcrd[i] - frag_com_z + rand_z;
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

    // Create GCMC sampler
    std::cout << "\nCreating GCMC sampler...\n";
    std::string ghost_file = output_prefix + "_ghosts.txt";
    std::string log_file = output_prefix + "_gcmc.log";

    GCMCSystemSampler sampler(
        &combined_topology,
        &ps,
        &exclusions,
        &thermostat,
        temperature,
        ghost_metadata,
        mu_ex,
        standard_volume,
        b_value,           // adams parameter
        0.0,               // adams_shift
        ImplicitSolventModel::NONE,
        "HOH",             // resname
        ghost_file,
        log_file);

    std::cout << "  GCMCSystemSampler created (lambda dynamics kernel)\n";

    // Open occupancy output file
    std::string occupancy_file = output_prefix + "_occupancy.dat";
    std::ofstream occ_out(occupancy_file);
    if (!occ_out.is_open()) {
      rtErr("Cannot open occupancy file: " + occupancy_file, "gcmc_test_runner");
    }
    occ_out << "# Cycle  Active_Molecules\n";

    // Open GPU and system memory tracking file
    std::string memory_file = output_prefix + "_gpu_memory.dat";
    std::ofstream mem_out(memory_file);
    if (!mem_out.is_open()) {
      rtErr("Cannot open memory file: " + memory_file, "gcmc_test_runner");
    }
    mem_out << "# Cycle  GPU_Free_MB  GPU_Used_MB  GPU_Total_MB  RSS_MB  VMS_MB\n";

    // Main GCMC loop
    std::cout << "\n========================================\n";
    std::cout << "Starting GCMC simulation\n";
    std::cout << "========================================\n";

    for (int cycle = 0; cycle < n_moves; cycle++) {
      // Query GPU memory BEFORE move
#ifdef STORMM_USE_HPC
      size_t free_bytes_before = 0;
      size_t total_bytes = 0;
      cudaError_t cuda_err = cudaMemGetInfo(&free_bytes_before, &total_bytes);
      if (cuda_err != cudaSuccess) {
        std::cerr << "Warning: cudaMemGetInfo failed at cycle " << cycle << "\n";
      }
#endif

      // Attempt GCMC move (insertion or deletion)
      bool accepted = sampler.runGCMCCycle();

      // Run MD propagation
      if (md_steps > 0) {
        sampler.propagateSystem(md_steps);
      }

      // Query GPU memory AFTER move and MD
#ifdef STORMM_USE_HPC
      size_t free_bytes_after = 0;
      cuda_err = cudaMemGetInfo(&free_bytes_after, &total_bytes);

      // Read system memory usage (includes CUDA pinned host memory)
      double rss_mb = 0.0;
      double vms_mb = 0.0;
      std::ifstream status_file("/proc/self/status");
      if (status_file.is_open()) {
        std::string line;
        while (std::getline(status_file, line)) {
          if (line.substr(0, 6) == "VmRSS:") {
            // Extract RSS in kB and convert to MB
            size_t pos = line.find_first_of("0123456789");
            if (pos != std::string::npos) {
              rss_mb = std::stod(line.substr(pos)) / 1024.0;
            }
          } else if (line.substr(0, 6) == "VmSize") {
            // Extract VMS in kB and convert to MB
            size_t pos = line.find_first_of("0123456789");
            if (pos != std::string::npos) {
              vms_mb = std::stod(line.substr(pos)) / 1024.0;
            }
          }
        }
        status_file.close();
      }

      if (cuda_err == cudaSuccess) {
        double free_mb = static_cast<double>(free_bytes_after) / (1024.0 * 1024.0);
        double total_mb = static_cast<double>(total_bytes) / (1024.0 * 1024.0);
        double used_mb = total_mb - free_mb;
        mem_out << cycle << "  " << std::fixed << std::setprecision(2)
                << free_mb << "  " << used_mb << "  " << total_mb << "  "
                << rss_mb << "  " << vms_mb << "\n";

        // Print memory warning if free GPU memory drops below 100 MB or RSS grows > 2GB
        if (free_mb < 100.0) {
          std::cout << "\nWARNING: Low GPU memory at cycle " << cycle
                    << " - Free: " << free_mb << " MB\n";
        }
        if (rss_mb > 2048.0) {
          std::cout << "\nWARNING: High RSS at cycle " << cycle
                    << " - RSS: " << rss_mb << " MB\n";
        }
      }
#endif

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

    mem_out.close();

    std::cout << "\n\n========================================\n";
    std::cout << "GCMC simulation complete\n";
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
    std::cout << "  GPU Memory: " << memory_file << "\n";
    std::cout << "  Ghosts:     " << ghost_file << "\n";
    std::cout << "  Log:        " << log_file << "\n";
    std::cout << "\n========================================\n";

  } catch (const std::exception& e) {
    std::cerr << "\nERROR: " << e.what() << "\n";
    return 1;
  }

  return 0;
}
