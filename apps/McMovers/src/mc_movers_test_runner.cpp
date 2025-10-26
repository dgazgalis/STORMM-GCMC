// Test runner for Monte Carlo movers (Translation, Rotation, Torsion)
// Validates that MC moves work correctly with lambda dynamics integration

#include <iostream>
#include <iomanip>
#include <fstream>
#include <memory>
#include <string>
#include <vector>
#include "copyright.h"

// Core includes
#include "Accelerator/gpu_details.h"
#include "Accelerator/core_kernel_manager.h"
#include "DataTypes/common_types.h"
#include "FileManagement/file_listing.h"
#include "Parsing/parse.h"
#include "Random/random.h"
#include "Reporting/error_format.h"

// Topology and trajectory
#include "Topology/atomgraph.h"
#include "Trajectory/phasespace.h"
#include "Trajectory/thermostat.h"

// Synthesis
#include "Synthesis/atomgraph_synthesis.h"
#include "Synthesis/phasespace_synthesis.h"
#include "Synthesis/static_mask_synthesis.h"

// Potential
#include "Potential/static_exclusionmask.h"

// Molecular mechanics
#include "MolecularMechanics/mm_controls.h"

// Sampling
#include "Sampling/gcmc_sampler.h"
#include "Sampling/mc_mover.h"

using namespace stormm;
using namespace stormm::card;
using namespace stormm::data_types;
using namespace stormm::diskutil;
using namespace stormm::energy;
using namespace stormm::errors;
using namespace stormm::mm;
using namespace stormm::parse;
using namespace stormm::random;
using namespace stormm::review;
using namespace stormm::sampling;
using namespace stormm::synthesis;
using namespace stormm::topology;
using namespace stormm::trajectory;

//-------------------------------------------------------------------------------------------------
// Print usage information
//-------------------------------------------------------------------------------------------------
void printUsage(const char* program_name) {
  std::cout << "MC Movers Test Application\n";
  std::cout << "Tests translation, rotation, and torsion Monte Carlo moves\n\n";
  std::cout << "Usage: " << program_name << " [options]\n\n";
  std::cout << "Required arguments:\n";
  std::cout << "  --topology <file>       Topology file (Amber prmtop format)\n";
  std::cout << "  --coords <file>         Coordinate file (Amber inpcrd format)\n\n";
  std::cout << "Optional arguments:\n";
  std::cout << "  --temp <K>              Temperature in Kelvin (default: 300.0)\n";
  std::cout << "  --n-moves <int>         Number of MC moves per mover type (default: 100)\n";
  std::cout << "  --max-disp <A>          Maximum translation displacement in Angstroms (default: 0.5)\n";
  std::cout << "  --max-rot <deg>         Maximum rotation angle in degrees (default: 30.0)\n";
  std::cout << "  --max-torsion <deg>     Maximum torsion angle change in degrees (default: 30.0)\n";
  std::cout << "  --output <file>         Output file for statistics (default: mc_movers_test.out)\n";
  std::cout << "  --trajectory <file>     Trajectory file for visualizing moves (optional)\n\n";
  std::cout << "Example:\n";
  std::cout << "  " << program_name << " --topology benzene.prmtop --coords benzene.inpcrd \\\n";
  std::cout << "    --temp 300 --n-moves 500 --max-disp 1.0 --max-rot 45 --output test.out\n";
}

//-------------------------------------------------------------------------------------------------
// Simple command-line argument parser
//-------------------------------------------------------------------------------------------------
std::string getArgValue(int argc, char* argv[], const std::string& flag, const std::string& default_val = "") {
  for (int i = 1; i < argc - 1; i++) {
    if (std::string(argv[i]) == flag) {
      return std::string(argv[i + 1]);
    }
  }
  return default_val;
}

double getArgDouble(int argc, char* argv[], const std::string& flag, double default_val) {
  std::string val = getArgValue(argc, argv, flag);
  return val.empty() ? default_val : std::stod(val);
}

int getArgInt(int argc, char* argv[], const std::string& flag, int default_val) {
  std::string val = getArgValue(argc, argv, flag);
  return val.empty() ? default_val : std::stoi(val);
}

bool hasArg(int argc, char* argv[], const std::string& flag) {
  for (int i = 1; i < argc; i++) {
    if (std::string(argv[i]) == flag) {
      return true;
    }
  }
  return false;
}

//-------------------------------------------------------------------------------------------------
// Test a specific MC mover
//-------------------------------------------------------------------------------------------------
void testMover(MCMover* mover,
               GCMCMolecule& test_molecule,
               int n_moves,
               const std::string& mover_name,
               std::ofstream& output_file) {

  std::cout << "\n========================================\n";
  std::cout << "Testing " << mover_name << " Mover\n";
  std::cout << "========================================\n";

  // Note: Statistics are automatically tracked, no need to reset
  // (each mover tracks its own statistics starting from zero)

  // Perform MC moves
  int accepted = 0;
  int rejected = 0;

  for (int i = 0; i < n_moves; i++) {
    bool accepted_move = mover->attemptMove(test_molecule);
    if (accepted_move) {
      accepted++;
    } else {
      rejected++;
    }

    // Print progress every 10 moves
    if ((i + 1) % 10 == 0) {
      double acceptance_rate = 100.0 * accepted / (i + 1);
      std::cout << "  Progress: " << (i + 1) << "/" << n_moves
                << " moves, acceptance rate: " << std::fixed << std::setprecision(1)
                << acceptance_rate << "%\r" << std::flush;
    }
  }

  std::cout << std::endl;

  // Get final statistics
  const MCMoveStatistics& stats = mover->getStatistics();

  // Print results to console
  std::cout << "\nResults:\n";
  std::cout << "  Total attempted: " << stats.n_attempted << "\n";
  std::cout << "  Accepted: " << stats.n_accepted << "\n";
  std::cout << "  Rejected: " << stats.n_rejected << "\n";
  std::cout << "  Acceptance rate: " << std::fixed << std::setprecision(2)
            << (stats.getAcceptanceRate() * 100.0) << "%\n";
  std::cout << "  Total energy change: " << std::fixed << std::setprecision(4)
            << stats.total_energy_change << " kcal/mol\n";

  // Write to output file
  output_file << "\n========================================\n";
  output_file << mover_name << " Mover Results\n";
  output_file << "========================================\n";
  output_file << "Total attempted:    " << stats.n_attempted << "\n";
  output_file << "Accepted:           " << stats.n_accepted << "\n";
  output_file << "Rejected:           " << stats.n_rejected << "\n";
  output_file << "Acceptance rate:    " << std::fixed << std::setprecision(2)
              << (stats.getAcceptanceRate() * 100.0) << "%\n";
  output_file << "Total energy change: " << std::fixed << std::setprecision(4)
              << stats.total_energy_change << " kcal/mol\n";
  output_file << "Avg energy change:   " << std::fixed << std::setprecision(4)
              << (stats.n_accepted > 0 ? stats.total_energy_change / stats.n_accepted : 0.0)
              << " kcal/mol per accepted move\n";
}

//-------------------------------------------------------------------------------------------------
// Main function
//-------------------------------------------------------------------------------------------------
int main(int argc, char* argv[]) {

  // Check for help flag
  if (argc < 2 || hasArg(argc, argv, "--help") || hasArg(argc, argv, "-h")) {
    printUsage(argv[0]);
    return 0;
  }

  // Parse command-line arguments
  const std::string topology_file = getArgValue(argc, argv, "--topology");
  const std::string coords_file = getArgValue(argc, argv, "--coords");
  const double temperature = getArgDouble(argc, argv, "--temp", 300.0);
  const int n_moves = getArgInt(argc, argv, "--n-moves", 100);
  const double max_displacement = getArgDouble(argc, argv, "--max-disp", 0.5);
  const double max_rotation = getArgDouble(argc, argv, "--max-rot", 30.0);
  const double max_torsion = getArgDouble(argc, argv, "--max-torsion", 30.0);
  const std::string output_file = getArgValue(argc, argv, "--output", "mc_movers_test.out");
  const std::string trajectory_file = getArgValue(argc, argv, "--trajectory");

  // Validate required arguments
  if (topology_file.empty() || coords_file.empty()) {
    std::cerr << "Error: --topology and --coords are required\n\n";
    printUsage(argv[0]);
    return 1;
  }

  // Print configuration
  std::cout << "MC Movers Test Application\n";
  std::cout << "==========================\n";
  std::cout << "Topology:             " << topology_file << "\n";
  std::cout << "Coordinates:          " << coords_file << "\n";
  std::cout << "Temperature:          " << temperature << " K\n";
  std::cout << "Moves per mover:      " << n_moves << "\n";
  std::cout << "Max displacement:     " << max_displacement << " A\n";
  std::cout << "Max rotation:         " << max_rotation << " deg\n";
  std::cout << "Max torsion:          " << max_torsion << " deg\n";
  std::cout << "Output file:          " << output_file << "\n";
  if (!trajectory_file.empty()) {
    std::cout << "Trajectory file:      " << trajectory_file << "\n";
  }
  std::cout << std::endl;

#ifndef STORMM_USE_CUDA
  std::cerr << "Error: This test requires CUDA support\n";
  return 1;
#endif

  try {
    // Load topology and coordinates
    std::cout << "Loading system...\n";
    AtomGraph topology(topology_file, ExceptionResponse::WARN);
    const int n_atoms = topology.getAtomCount();
    std::cout << "  Loaded " << n_atoms << " atoms\n";

    PhaseSpace ps(coords_file, topology);
    StaticExclusionMask exclusions(&topology);

    // Create GPU infrastructure for GCMC sampler
    std::vector<AtomGraph*> ag_list = {&topology};
    std::vector<PhaseSpace> ps_vec = {ps};
    AtomGraphSynthesis ag_synthesis(ag_list);
    PhaseSpaceSynthesis ps_synthesis(ps_vec, ag_list);

    std::vector<StaticExclusionMask*> mask_list = {&exclusions};
    std::vector<int> topology_indices = {0};
    StaticExclusionMaskSynthesis poly_se(mask_list, topology_indices);

    ag_synthesis.loadNonbondedWorkUnits(poly_se);

    const GpuDetails gpu;
    const CoreKlManager launcher(gpu, ag_synthesis);

    Thermostat thermostat(topology, ThermostatKind::NONE, temperature);

    // Create GCMC sampler (provides energy evaluation framework)
    // Use dummy GCMC parameters since we only need the energy evaluation
    const double dummy_mu = 0.0;
    const double dummy_vol = 1661.0;  // Standard state volume
    const double dummy_b = 0.0;
    GhostMoleculeMetadata dummy_metadata;
    dummy_metadata.base_atom_count = n_atoms;
    dummy_metadata.base_molecule_count = 1;
    dummy_metadata.base_residue_count = topology.getResidueCount();
    dummy_metadata.n_ghost_molecules = 0;

    GCMCSystemSampler sampler(
        &topology,
        &ps,
        &exclusions,
        &thermostat,
        temperature,
        dummy_metadata,
        dummy_mu,
        dummy_vol,
        dummy_b,
        0.0,
        ImplicitSolventModel::NONE,
        "",
        "",
        "");

    // Create a test molecule containing all atoms
    GCMCMolecule test_molecule;
    test_molecule.resid = 0;
    test_molecule.lambda_vdw = 1.0;
    test_molecule.lambda_ele = 1.0;
    test_molecule.atom_indices.reserve(n_atoms);
    for (int i = 0; i < n_atoms; i++) {
      test_molecule.atom_indices.push_back(i);
    }

    // Initialize random number generator
    const int random_seed = 42;
    Xoshiro256ppGenerator rng(random_seed);

    // Calculate beta = 1/(kB * T)
    const double kB = 0.001987204;  // kcal/(mol*K)
    const double beta = 1.0 / (kB * temperature);

    // Open output file
    std::ofstream out_file(output_file);
    out_file << "MC Movers Test Results\n";
    out_file << "======================\n";
    out_file << "Topology:        " << topology_file << "\n";
    out_file << "Coordinates:     " << coords_file << "\n";
    out_file << "Temperature:     " << temperature << " K\n";
    out_file << "Moves per mover: " << n_moves << "\n";
    out_file << "Random seed:     " << random_seed << "\n";
    out_file << "Number of atoms: " << n_atoms << "\n";

    // Test Translation Mover
    std::cout << "\nCreating Translation Mover...\n";
    TranslationMover translation_mover(&sampler, beta, &rng, max_displacement);
    testMover(&translation_mover, test_molecule, n_moves, "Translation", out_file);

    // Test Rotation Mover
    std::cout << "\nCreating Rotation Mover...\n";
    const double max_rot_rad = max_rotation * 3.14159265359 / 180.0;
    RotationMover rotation_mover(&sampler, beta, &rng, max_rot_rad);
    testMover(&rotation_mover, test_molecule, n_moves, "Rotation", out_file);

    // Test Torsion Mover
    std::cout << "\nCreating Torsion Mover...\n";
    const double max_torsion_rad = max_torsion * 3.14159265359 / 180.0;
    TorsionMover torsion_mover(&sampler, beta, &rng, &topology, max_torsion_rad);
    testMover(&torsion_mover, test_molecule, n_moves, "Torsion", out_file);

    // Final summary
    out_file << "\n========================================\n";
    out_file << "Test completed successfully\n";
    out_file << "========================================\n";
    out_file.close();

    std::cout << "\n========================================\n";
    std::cout << "All tests completed successfully!\n";
    std::cout << "Results written to: " << output_file << "\n";
    std::cout << "========================================\n";

  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
