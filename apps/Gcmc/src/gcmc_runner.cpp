#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>
#include "../../../src/copyright.h"
#ifdef STORMM_USE_HPC
#  include "../../../src/Accelerator/core_kernel_manager.h"
#  include "../../../src/Accelerator/gpu_details.h"
#  include "../../../src/Accelerator/hpc_config.h"
#endif
#include "../../../src/Chemistry/chemistry_enumerators.h"
#include "../../../src/Constants/behavior.h"
#include "../../../src/DataTypes/common_types.h"
#include "../../../src/DataTypes/stormm_vector_types.h"
#include "../../../src/FileManagement/file_listing.h"
#include "../../../src/Math/vector_ops.h"
#include "../../../src/MoleculeFormat/pdb.h"
#include "../../../src/Namelists/command_line_parser.h"
#include "../../../src/Namelists/input_transcript.h"
#include "../../../src/Namelists/nml_dynamics.h"
#include "../../../src/Namelists/nml_files.h"
#include "../../../src/Namelists/nml_minimize.h"
#include "../../../src/Namelists/nml_precision.h"
#include "../../../src/Namelists/nml_random.h"
#include "../../../src/Namelists/nml_solvent.h"
#include "../../../src/Namelists/user_settings.h"
#include "../../../src/Parsing/parse.h"
#include "../../../src/Potential/energy_enumerators.h"
#include "../../../src/Potential/static_exclusionmask.h"
#include "../../../src/Reporting/error_format.h"
#include "../../../src/Reporting/help_messages.h"
#include "../../../src/Reporting/present_energy.h"
#include "../../../src/Reporting/progress_bar.h"
#include "../../../src/Sampling/gcmc_sampler.h"
#include "../../../src/Structure/structure_enumerators.h"
#include "../../../src/Synthesis/systemcache.h"
#include "../../../src/Topology/atomgraph.h"
#include "../../../src/Topology/atomgraph_enumerators.h"
#include "../../../src/Trajectory/coordinateframe.h"
#include "../../../src/Trajectory/phasespace.h"
#include "../../../src/Trajectory/thermostat.h"
#include "../../../src/Trajectory/trajectory_enumerators.h"
#include "../../../src/UnitTesting/stopwatch.h"

using namespace stormm::card;
using namespace stormm::chemistry;
using namespace stormm::constants;
using namespace stormm::data_types;
using namespace stormm::diskutil;
using namespace stormm::display;
using namespace stormm::energy;
using namespace stormm::errors;
using namespace stormm::namelist;
using namespace stormm::parse;
using namespace stormm::reporting;
using namespace stormm::review;
using namespace stormm::sampling;
using namespace stormm::stmath;
using namespace stormm::structure;
using namespace stormm::synthesis;
using namespace stormm::testing;
using namespace stormm::topology;
using namespace stormm::trajectory;

//-------------------------------------------------------------------------------------------------
// Helper function to write configuration file
//-------------------------------------------------------------------------------------------------
void writeConfigFile(const std::string& filename,
                     const std::string& system_prmtop,
                     const std::string& system_inpcrd,
                     const std::string& fragment_prmtop,
                     const std::string& fragment_inpcrd,
                     const std::string& output_prefix,
                     int n_moves, double bvalue, double sphere_radius,
                     const std::string& ref_atoms_str, double temperature,
                     double timestep, int npert, int nprop, int nghost,
                     double mu_ex, double standard_volume, bool overwrite,
                     const std::string& mode, double box_size, bool freeze_fragments,
                     bool save_trajectory, int snapshot_interval, const std::string& trajectory_prefix,
                     double mc_translation, double mc_rotation, double mc_torsion, int mc_frequency) {

  std::ofstream config(filename);
  if (!config.is_open()) {
    rtErr("Could not open configuration file for writing: " + filename, "writeConfigFile");
  }

  config << "# STORMM GCMC Configuration File\n";
  config << "# Generated: " << __DATE__ << " " << __TIME__ << "\n";
  config << "#\n";
  config << "# Usage: gcmc.stormm --config " << filename << "\n";
  config << "#\n\n";

  config << "# Input files\n";
  if (!system_prmtop.empty()) {
    config << "-p " << system_prmtop << "\n";
  }
  if (!system_inpcrd.empty()) {
    config << "-c " << system_inpcrd << "\n";
  }
  if (!fragment_prmtop.empty() && fragment_prmtop != "none") {
    config << "--fragment-prmtop " << fragment_prmtop << "\n";
  }
  if (!fragment_inpcrd.empty() && fragment_inpcrd != "none") {
    config << "--fragment-inpcrd " << fragment_inpcrd << "\n";
  }
  if (!output_prefix.empty()) {
    config << "-o " << output_prefix << "\n";
  }
  if (overwrite) {
    config << "-O\n";
  }
  config << "\n";

  config << "# GCMC parameters\n";
  config << "-n " << n_moves << "\n";
  config << "-b " << bvalue << "  # Adams B parameter (if set, overrides mu-ex)\n";
  config << "--temp " << temperature << "\n";
  config << "--nghost " << nghost << "\n";
  config << "--mu-ex " << mu_ex << "  # Only used if B is not set (NaN)\n";
  config << "--standard-volume " << standard_volume << "  # Only used if B is not set (NaN)\n";
  config << "--mode " << mode << "\n";
  config << "\n";

  config << "# NCMC parameters\n";
  config << "--npert " << npert << "\n";
  config << "--nprop " << nprop << "\n";
  config << "--timestep " << timestep << "\n";
  config << "\n";

  config << "# Sphere mode parameters (only used if mode=sphere)\n";
  config << "--sphere-radius " << sphere_radius << "\n";
  if (!ref_atoms_str.empty() && ref_atoms_str != "none") {
    config << "--ref-atoms " << ref_atoms_str << "\n";
  }
  config << "\n";

  config << "# Fragment-only mode parameters\n";
  config << "--box-size " << box_size << "\n";
  if (freeze_fragments) {
    config << "--freeze-fragments\n";
  }
  config << "\n";

  config << "# Monte Carlo move parameters\n";
  if (mc_translation > 0.0) {
    config << "--mc-translation " << mc_translation << "\n";
  }
  if (mc_rotation > 0.0) {
    config << "--mc-rotation " << mc_rotation << "\n";
  }
  if (mc_torsion > 0.0) {
    config << "--mc-torsion " << mc_torsion << "\n";
  }
  if (mc_frequency > 0) {
    config << "--mc-frequency " << mc_frequency << "\n";
  }
  config << "\n";

  config << "# Trajectory output\n";
  if (save_trajectory) {
    config << "--save-trajectory\n";
    config << "--snapshot-interval " << snapshot_interval << "\n";
    config << "--trajectory-prefix " << trajectory_prefix << "\n";
  }
  config << "\n";

  config.close();
  std::cout << "Configuration written to: " << filename << "\n";
}

//-------------------------------------------------------------------------------------------------
// Helper function to read configuration file
//-------------------------------------------------------------------------------------------------
void readConfigFile(const std::string& filename, std::vector<std::string>& args) {
  std::ifstream config(filename);
  if (!config.is_open()) {
    rtErr("Could not open configuration file: " + filename, "readConfigFile");
  }

  std::string line;
  while (std::getline(config, line)) {
    // Remove leading/trailing whitespace
    line.erase(0, line.find_first_not_of(" \t"));
    line.erase(line.find_last_not_of(" \t") + 1);

    // Skip empty lines and comments
    if (line.empty() || line[0] == '#') {
      continue;
    }

    // Parse the line
    std::istringstream iss(line);
    std::string token;
    while (iss >> token) {
      args.push_back(token);
    }
  }

  config.close();
  std::cout << "Read configuration from: " << filename << "\n";
}

//-------------------------------------------------------------------------------------------------
// Helper function to freeze fragment internal degrees of freedom
//-------------------------------------------------------------------------------------------------
void freezeFragmentGeometry(AtomGraph& topology, const GhostMoleculeMetadata& ghost_metadata) {
  // TODO: Implement fragment geometry freezing
  // Current approach requires access to private AtomGraph members.
  // Possible solutions:
  // 1. Use public setter methods (setBondParameters, setAngleParameters, setDihedralParameters)
  // 2. Implement this as a friend function or member of AtomGraph
  // 3. Add public methods to AtomGraph for bulk parameter modification

  std::cout << "\nWARNING: --freeze-fragments is not yet fully implemented\n";
  std::cout << "  Fragment molecules will retain internal flexibility during sampling\n";

  // For now, this is a no-op - fragments remain flexible
  (void)topology;  // Suppress unused parameter warning
  (void)ghost_metadata;
}

//-------------------------------------------------------------------------------------------------
// Helper function to find reference atoms for the sphere center
//-------------------------------------------------------------------------------------------------
std::vector<int> findReferenceAtoms(const AtomGraph& ag, const std::string& ref_atoms_str) {
  std::vector<int> ref_atoms;

  if (!ref_atoms_str.empty() && ref_atoms_str != "none") {
    // Parse comma-separated list of atom indices
    std::istringstream iss(ref_atoms_str);
    std::string token;
    while (std::getline(iss, token, ',')) {
      try {
        int atom_idx = std::stoi(token) - 1; // Convert to 0-based indexing
        if (atom_idx >= 0 && atom_idx < ag.getAtomCount()) {
          ref_atoms.push_back(atom_idx);
        } else {
          rtWarn("Atom index " + std::to_string(atom_idx + 1) + " is out of range. Skipping.",
                 "findReferenceAtoms");
        }
      } catch (const std::exception& e) {
        rtWarn("Invalid atom index: " + token + ". Skipping.", "findReferenceAtoms");
      }
    }
  }

  // If no reference atoms specified, try to find protein CA atoms
  if (ref_atoms.empty()) {
    const ChemicalDetailsKit cdk = ag.getChemicalDetailsKit();
    for (int i = 0; i < ag.getAtomCount(); i++) {
      // Convert char4 to string for comparison
      const char4 atom_name_char4 = cdk.atom_names[i];
      char atom_name_buf[5] = {atom_name_char4.x, atom_name_char4.y,
                               atom_name_char4.z, atom_name_char4.w, '\0'};
      std::string atom_name_str(atom_name_buf);

      if (atom_name_str == " CA " || atom_name_str == "CA  ") {
        // Check if it's in a protein residue (simplified check)
        const int res_idx = ag.getResidueIndex(i);
        const char4 res_name_char4 = cdk.res_names[res_idx];
        char res_name_buf[5] = {res_name_char4.x, res_name_char4.y,
                                res_name_char4.z, res_name_char4.w, '\0'};
        std::string res_name(res_name_buf);
        // Trim whitespace
        res_name.erase(res_name.find_last_not_of(" ") + 1);

        // Common amino acid residues
        const std::vector<std::string> amino_acids = {
          "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
          "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"
        };

        if (std::find(amino_acids.begin(), amino_acids.end(), res_name) != amino_acids.end()) {
          ref_atoms.push_back(i);
        }
      }
    }

    if (!ref_atoms.empty()) {
      std::cout << "Found " << ref_atoms.size() << " protein CA atoms for sphere center\n";
    }
  }

  // If still no reference atoms, use geometric center of the system
  if (ref_atoms.empty()) {
    std::cout << "No reference atoms specified or found. Using all heavy atoms for sphere center.\n";
    const ChemicalDetailsKit cdk = ag.getChemicalDetailsKit();
    for (int i = 0; i < ag.getAtomCount(); i++) {
      // Include all non-hydrogen atoms
      if (cdk.z_numbers[i] > 1) {
        ref_atoms.push_back(i);
      }
    }
  }

  if (ref_atoms.empty()) {
    rtErr("No reference atoms could be determined for sphere center.", "findReferenceAtoms");
  }

  return ref_atoms;
}

//-------------------------------------------------------------------------------------------------
// Helper function to identify ghost molecules in the topology
//-------------------------------------------------------------------------------------------------
bool identifyGhostMolecules(const AtomGraph& ag,
                           const std::string& fragment_resname,
                           int expected_ghosts,
                           GhostMoleculeMetadata& ghost_metadata) {

  const ChemicalDetailsKit cdk = ag.getChemicalDetailsKit();
  std::vector<int> matching_molecules;

  // Find all molecules that match the fragment residue name
  for (int mol_idx = 0; mol_idx < ag.getMoleculeCount(); mol_idx++) {
    const int2 mol_limits = ag.getMoleculeLimits(mol_idx);

    // Check the first atom of the molecule for residue name
    if (mol_limits.x < ag.getAtomCount()) {
      const int res_idx = ag.getResidueIndex(mol_limits.x);
      const char4 res_name_char4 = cdk.res_names[res_idx];
      char res_name_buf[5] = {res_name_char4.x, res_name_char4.y,
                              res_name_char4.z, res_name_char4.w, '\0'};
      std::string res_name_str(res_name_buf);
      res_name_str.erase(res_name_str.find_last_not_of(" ") + 1);

      if (res_name_str == fragment_resname) {
        matching_molecules.push_back(mol_idx);
      }
    }
  }

  if (matching_molecules.empty()) {
    std::cerr << "\nERROR: No molecules with residue name '" << fragment_resname
              << "' found in the topology.\n";
    std::cerr << "The system topology must contain pre-allocated ghost molecules.\n";
    return false;
  }

  std::cout << "Found " << matching_molecules.size() << " molecules with residue name '"
            << fragment_resname << "'\n";

  // Determine which are ghost molecules (typically the last N molecules)
  // Assume the last expected_ghosts matching molecules are ghosts
  int n_to_use = std::min(expected_ghosts, static_cast<int>(matching_molecules.size()));
  int start_idx = matching_molecules.size() - n_to_use;

  ghost_metadata.n_ghost_molecules = n_to_use;
  ghost_metadata.base_molecule_count = ag.getMoleculeCount() - n_to_use;

  // Get base atom and residue counts (before first ghost)
  if (start_idx > 0) {
    ghost_metadata.base_atom_count = ag.getMoleculeLimits(matching_molecules[start_idx] - 1).y;
    ghost_metadata.base_residue_count = ag.getResidueIndex(ghost_metadata.base_atom_count - 1) + 1;
  } else {
    // All molecules are ghosts
    ghost_metadata.base_atom_count = 0;
    ghost_metadata.base_residue_count = 0;
  }

  // Fill in ghost molecule information
  for (int i = start_idx; i < static_cast<int>(matching_molecules.size()); i++) {
    int mol_idx = matching_molecules[i];
    int2 mol_limits = ag.getMoleculeLimits(mol_idx);
    int res_idx = ag.getResidueIndex(mol_limits.x);

    ghost_metadata.ghost_molecule_indices.push_back(mol_idx);
    ghost_metadata.ghost_residue_indices.push_back(res_idx);
    ghost_metadata.ghost_atom_ranges.push_back(mol_limits);
  }

  std::cout << "Identified " << n_to_use << " ghost molecules (molecules "
            << matching_molecules[start_idx] << " to "
            << matching_molecules.back() << ")\n";
  std::cout << "Base system: " << ghost_metadata.base_molecule_count << " molecules, "
            << ghost_metadata.base_atom_count << " atoms\n";

  return true;
}

//-------------------------------------------------------------------------------------------------
// Main function
//-------------------------------------------------------------------------------------------------
int main(int argc, const char* argv[]) {

  // Wall time tracking
  StopWatch timer("Timings for gcmc.stormm");
  const int file_parse_tm = timer.addCategory("File Parsing");
  const int gen_setup_tm  = timer.addCategory("Setup, General");
  const int gcmc_setup_tm = timer.addCategory("Setup, GCMC");
  const int gcmc_run_tm   = timer.addCategory("Run, GCMC");
  const int output_tm     = timer.addCategory("Output");

  // Engage the GPU if available
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
  timer.assignTime(gen_setup_tm);

  // Parse the command line
  CommandLineParser clip("gcmc.stormm", "Grand Canonical Monte Carlo sampling with NCMC moves");

  // Add standard inputs for topology and coordinates
  clip.addStandardApplicationInputs({ "-p", "-c", "-O", "-o", "-except" });

  // Get the namelist emulator to add custom keywords
  NamelistEmulator *t_nml = clip.getNamelistPointer();

  // System files
  t_nml->addKeyword("--fragment-prmtop", NamelistType::STRING, "none");
  t_nml->addHelp("--fragment-prmtop", "Fragment topology file (prmtop) to insert/delete");

  t_nml->addKeyword("--fragment-inpcrd", NamelistType::STRING, "none");
  t_nml->addHelp("--fragment-inpcrd", "Fragment coordinate file (inpcrd)");

  // Add GCMC-specific command line options
  t_nml->addKeyword("-n", NamelistType::INTEGER, "1000");
  t_nml->addHelp("-n", "Number of GCMC moves to perform (default: 1000)");

  t_nml->addKeyword("--moves", NamelistType::INTEGER, "1000");
  t_nml->addHelp("--moves", "Number of GCMC moves to perform (default: 1000)");

  t_nml->addKeyword("-b", NamelistType::REAL, "5.0");
  t_nml->addHelp("-b", "Adams B parameter for chemical potential (default: 5.0, overrides --mu-ex)");

  t_nml->addKeyword("--bvalue", NamelistType::REAL, "5.0");
  t_nml->addHelp("--bvalue", "Adams B parameter for chemical potential (default: 5.0, overrides --mu-ex)");

  t_nml->addKeyword("--sphere-radius", NamelistType::REAL, "10.0");
  t_nml->addHelp("--sphere-radius", "Sphere radius in Angstroms (default: 10.0, sphere mode only)");

  t_nml->addKeyword("--ref-atoms", NamelistType::STRING, "none");
  t_nml->addHelp("--ref-atoms", "Comma-separated atom indices for sphere center (sphere mode only)");

  t_nml->addKeyword("--temp", NamelistType::REAL, "300.0");
  t_nml->addHelp("--temp", "Temperature in Kelvin (default: 300.0)");

  t_nml->addKeyword("--timestep", NamelistType::REAL, "2.0");
  t_nml->addHelp("--timestep", "MD timestep in fs (default: 2.0)");

  t_nml->addKeyword("--npert", NamelistType::INTEGER, "50");
  t_nml->addHelp("--npert", "Number of NCMC perturbation steps (default: 50, balanced for speed)");

  t_nml->addKeyword("--nprop", NamelistType::INTEGER, "2");
  t_nml->addHelp("--nprop", "Propagation steps per perturbation (default: 2, balanced for speed)");

  t_nml->addKeyword("--md-steps", NamelistType::INTEGER, "0");
  t_nml->addHelp("--md-steps", "MD steps to run between GCMC moves (default: 0, disabled)");

  t_nml->addKeyword("--nghost", NamelistType::INTEGER, "50");
  t_nml->addHelp("--nghost", "Number of ghost molecules to use from topology (default: 50)");

  t_nml->addKeyword("--mu-ex", NamelistType::REAL, "0.0");
  t_nml->addHelp("--mu-ex", "Excess chemical potential in kcal/mol (default: 0.0)");

  t_nml->addKeyword("--standard-volume", NamelistType::REAL, "30.0");
  t_nml->addHelp("--standard-volume", "Standard volume in Angstrom^3 (default: 30.0)");

  t_nml->addKeyword("--mode", NamelistType::STRING, "system");
  t_nml->addHelp("--mode", "GCMC sampling mode: 'sphere' (local) or 'system' (box-wide) (default: system)");

  t_nml->addKeyword("--box-size", NamelistType::REAL, "50.0");
  t_nml->addHelp("--box-size", "Box size in Angstroms for fragment-only mode (default: 50.0)");

  t_nml->addKeyword("--freeze-fragments", NamelistType::BOOLEAN, "");
  t_nml->addHelp("--freeze-fragments", "Freeze fragment internal geometry (disable dihedrals/angles)");

  t_nml->addKeyword("--config", NamelistType::STRING, "none");
  t_nml->addHelp("--config", "Read parameters from configuration file");

  t_nml->addKeyword("--write-config", NamelistType::STRING, "none");
  t_nml->addHelp("--write-config", "Write current parameters to configuration file and exit");

  t_nml->addKeyword("--save-trajectory", NamelistType::BOOLEAN, "");
  t_nml->addHelp("--save-trajectory", "Save trajectory snapshots as PDB files");

  t_nml->addKeyword("--snapshot-interval", NamelistType::INTEGER, "100");
  t_nml->addHelp("--snapshot-interval", "Save snapshot every N moves (default: 100)");

  t_nml->addKeyword("--trajectory-prefix", NamelistType::STRING, "gcmc_snapshot");
  t_nml->addHelp("--trajectory-prefix", "Prefix for trajectory files (default: gcmc_snapshot)");

  // Add Generalized Born implicit solvent model parameter
  t_nml->addKeyword("--gb-model", NamelistType::STRING, "none");
  t_nml->addHelp("--gb-model", "Generalized Born model: none, hct, obc, obc2, neck, neck2 (default: none)");

  // Add adaptive B protocol parameters
  t_nml->addKeyword("--adaptive-b", NamelistType::BOOLEAN, "");
  t_nml->addHelp("--adaptive-b", "Enable three-stage adaptive B protocol");

  t_nml->addKeyword("--stage1-moves", NamelistType::INTEGER, "5000000");
  t_nml->addHelp("--stage1-moves", "Number of moves for discovery stage (default: 5M)");

  t_nml->addKeyword("--stage2-moves", NamelistType::INTEGER, "1000000");
  t_nml->addHelp("--stage2-moves", "Number of moves for coarse equilibration (default: 1M)");

  t_nml->addKeyword("--stage3-moves", NamelistType::INTEGER, "2000000");
  t_nml->addHelp("--stage3-moves", "Number of moves for fine annealing (default: 2M)");

  t_nml->addKeyword("--b-discovery", NamelistType::REAL, "10.0");
  t_nml->addHelp("--b-discovery", "High B value for discovery stage (default: 10.0)");

  t_nml->addKeyword("--target-occupancy", NamelistType::REAL, "0.5");
  t_nml->addHelp("--target-occupancy", "Target fraction of N_max for stage 2 (default: 0.5)");

  t_nml->addKeyword("--coarse-rate", NamelistType::REAL, "0.1");
  t_nml->addHelp("--coarse-rate", "Learning rate for coarse equilibration (default: 0.1)");

  t_nml->addKeyword("--fine-rate", NamelistType::REAL, "0.01");
  t_nml->addHelp("--fine-rate", "Learning rate for fine annealing (default: 0.01)");

  t_nml->addKeyword("--b-min", NamelistType::REAL, "0.1");
  t_nml->addHelp("--b-min", "Minimum B clamp value (default: 0.1)");

  t_nml->addKeyword("--b-max", NamelistType::REAL, "15.0");
  t_nml->addHelp("--b-max", "Maximum B clamp value (default: 15.0)");

  // Monte Carlo move parameters
  t_nml->addKeyword("--mc-translation", NamelistType::REAL, "0.0");
  t_nml->addHelp("--mc-translation", "Enable translation moves with max displacement (Angstroms, 0=disabled)");

  t_nml->addKeyword("--mc-rotation", NamelistType::REAL, "0.0");
  t_nml->addHelp("--mc-rotation", "Enable rotation moves with max angle (degrees, 0=disabled)");

  t_nml->addKeyword("--mc-torsion", NamelistType::REAL, "0.0");
  t_nml->addHelp("--mc-torsion", "Enable torsion moves with max angle (degrees, 0=disabled)");

  t_nml->addKeyword("--mc-frequency", NamelistType::INTEGER, "0");
  t_nml->addHelp("--mc-frequency", "MC moves between GCMC moves (0=disabled)");

  // Hybrid MD/MC simulation parameters
  t_nml->addKeyword("--hybrid-mode", NamelistType::BOOLEAN, "");
  t_nml->addHelp("--hybrid-mode", "Run hybrid MD/MC simulation instead of pure GCMC");

  t_nml->addKeyword("--hybrid-md-steps", NamelistType::INTEGER, "1000000");
  t_nml->addHelp("--hybrid-md-steps", "Total MD steps for hybrid simulation");

  t_nml->addKeyword("--move-frequency", NamelistType::INTEGER, "100");
  t_nml->addHelp("--move-frequency", "Attempt move every N MD steps");

  t_nml->addKeyword("--p-gcmc", NamelistType::REAL, "0.5");
  t_nml->addHelp("--p-gcmc", "Probability of GCMC vs MC move (0-1)");

  // Check if we need to read from a config file first
  std::vector<std::string> config_args;
  bool has_config = false;
  for (int i = 1; i < argc; i++) {
    if (std::string(argv[i]) == "--config" && i + 1 < argc) {
      readConfigFile(argv[i + 1], config_args);
      has_config = true;
      break;
    }
  }

  // If config file was read, parse it first, then overlay command line args
  if (has_config) {
    // Build new argv from config file + original command line
    std::vector<const char*> new_argv;
    new_argv.push_back(argv[0]);  // Program name

    // Add args from config file
    std::vector<std::string> config_strings;
    for (const auto& arg : config_args) {
      config_strings.push_back(arg);
    }
    for (const auto& arg : config_strings) {
      new_argv.push_back(arg.c_str());
    }

    // Add original command line args (will override config file)
    for (int i = 1; i < argc; i++) {
      // Skip --config and its argument
      if (std::string(argv[i]) == "--config") {
        i++;  // Skip the filename too
        continue;
      }
      new_argv.push_back(argv[i]);
    }

    clip.parseUserInput(new_argv.size(), new_argv.data());
  } else {
    // Normal parsing
    clip.parseUserInput(argc, argv);
  }

  // Extract command line values
  const std::string system_prmtop = t_nml->getStringValue("-p");
  const std::string system_inpcrd = t_nml->getStringValue("-c");
  const std::string fragment_prmtop = t_nml->getStringValue("--fragment-prmtop");
  const std::string fragment_inpcrd = t_nml->getStringValue("--fragment-inpcrd");
  const std::string output_prefix = t_nml->getStringValue("-o");

  // GCMC parameters
  const int n_moves = std::max(t_nml->getIntValue("-n"), t_nml->getIntValue("--moves"));
  const double bvalue = std::max(t_nml->getRealValue("-b"), t_nml->getRealValue("--bvalue"));
  const double sphere_radius = t_nml->getRealValue("--sphere-radius");
  const std::string ref_atoms_str = t_nml->getStringValue("--ref-atoms");
  const double temperature = t_nml->getRealValue("--temp");
  const double timestep = t_nml->getRealValue("--timestep");
  const int npert = t_nml->getIntValue("--npert");
  const int nprop = t_nml->getIntValue("--nprop");
  const int md_steps = t_nml->getIntValue("--md-steps");
  const int nghost = t_nml->getIntValue("--nghost");
  const double mu_ex = t_nml->getRealValue("--mu-ex");
  const double standard_volume = t_nml->getRealValue("--standard-volume");
  const bool overwrite = t_nml->getBoolValue("-O");
  const std::string mode = t_nml->getStringValue("--mode");
  const double box_size = t_nml->getRealValue("--box-size");
  const bool freeze_fragments = t_nml->getBoolValue("--freeze-fragments");
  const std::string write_config_file = t_nml->getStringValue("--write-config");
  const bool save_trajectory = t_nml->getBoolValue("--save-trajectory");
  const int snapshot_interval = t_nml->getIntValue("--snapshot-interval");
  const std::string trajectory_prefix = t_nml->getStringValue("--trajectory-prefix");

  // Parse adaptive B protocol parameters
  const bool adaptive_b = t_nml->getBoolValue("--adaptive-b");
  const int stage1_moves = t_nml->getIntValue("--stage1-moves");
  const int stage2_moves = t_nml->getIntValue("--stage2-moves");
  const int stage3_moves = t_nml->getIntValue("--stage3-moves");
  const double b_discovery = t_nml->getRealValue("--b-discovery");
  const double target_occupancy = t_nml->getRealValue("--target-occupancy");
  const double coarse_rate = t_nml->getRealValue("--coarse-rate");
  const double fine_rate = t_nml->getRealValue("--fine-rate");
  const double b_min = t_nml->getRealValue("--b-min");
  const double b_max = t_nml->getRealValue("--b-max");

  // Parse Monte Carlo move parameters
  const double mc_translation = t_nml->getRealValue("--mc-translation");
  const double mc_rotation = t_nml->getRealValue("--mc-rotation");
  const double mc_torsion = t_nml->getRealValue("--mc-torsion");
  const int mc_frequency = t_nml->getIntValue("--mc-frequency");

  // Parse hybrid MD/MC simulation parameters
  const bool hybrid_mode = t_nml->getBoolValue("--hybrid-mode");
  const int hybrid_md_steps = t_nml->getIntValue("--hybrid-md-steps");
  const int move_frequency = t_nml->getIntValue("--move-frequency");
  const double p_gcmc = t_nml->getRealValue("--p-gcmc");

  // Parse Generalized Born model parameter
  const std::string gb_model_str = t_nml->getStringValue("--gb-model");
  ImplicitSolventModel gb_model = ImplicitSolventModel::NONE;
  if (gb_model_str == "none") {
    gb_model = ImplicitSolventModel::NONE;
  } else if (gb_model_str == "hct") {
    gb_model = ImplicitSolventModel::HCT_GB;
  } else if (gb_model_str == "obc") {
    gb_model = ImplicitSolventModel::OBC_GB;
  } else if (gb_model_str == "obc2") {
    gb_model = ImplicitSolventModel::OBC_GB_II;
  } else if (gb_model_str == "neck") {
    gb_model = ImplicitSolventModel::NECK_GB;
  } else if (gb_model_str == "neck2") {
    gb_model = ImplicitSolventModel::NECK_GB_II;
  } else if (gb_model_str != "none") {
    rtErr("Unknown GB model: " + gb_model_str + ". Valid options: none, hct, obc, obc2, neck, neck2", "main");
  }

  // Check if user just wants to write config and exit
  if (!write_config_file.empty() && write_config_file != "none") {
    writeConfigFile(write_config_file, system_prmtop, system_inpcrd,
                   fragment_prmtop, fragment_inpcrd, output_prefix,
                   n_moves, bvalue, sphere_radius, ref_atoms_str, temperature,
                   timestep, npert, nprop, nghost, mu_ex, standard_volume,
                   overwrite, mode, box_size, freeze_fragments,
                   save_trajectory, snapshot_interval, trajectory_prefix,
                   mc_translation, mc_rotation, mc_torsion, mc_frequency);
    std::cout << "\nConfiguration file written successfully.\n";
    std::cout << "To use it, run: gcmc.stormm --config " << write_config_file << "\n";
    return 0;
  }

  // Validate input files
  // Allow fragment-only mode: if no system provided, use fragment as the system
  // Treat default values "prmtop" and "inpcrd" as empty (they're just placeholders)
  const bool has_system = (!system_prmtop.empty() && system_prmtop != "prmtop");
  const bool has_fragment = (!fragment_prmtop.empty() && fragment_prmtop != "none");
  const bool fragment_only_mode = (!has_system && has_fragment);

  if (!has_system && !has_fragment) {
    std::cerr << "Error: At least one topology file is required.\n";
    std::cerr << "\nUsage:\n";
    std::cerr << "  # With base system + fragment:\n";
    std::cerr << "  gcmc.stormm -p system.prmtop -c system.inpcrd \\\n";
    std::cerr << "              --fragment-prmtop fragment.prmtop [--fragment-inpcrd fragment.inpcrd] \\\n";
    std::cerr << "              [options]\n\n";
    std::cerr << "  # Fragment-only mode (pure fragment GCMC):\n";
    std::cerr << "  gcmc.stormm --fragment-prmtop fragment.prmtop [--fragment-inpcrd fragment.inpcrd] \\\n";
    std::cerr << "              [options]\n\n";
    return 1;
  }

  // Validate mode
  if (mode != "sphere" && mode != "system") {
    rtErr("Invalid mode: " + mode + ". Use 'sphere' or 'system'.", "gcmc_runner");
  }

  timer.assignTime(file_parse_tm);

  // Print startup information
  std::cout << "\n";
  std::cout << "========================================\n";
  std::cout << "       GCMC Simulation with STORMM      \n";
  std::cout << "========================================\n";

  if (fragment_only_mode) {
    std::cout << "\n** FRAGMENT-ONLY MODE **\n";
    std::cout << "Running pure fragment GCMC (no base system)\n";
  }

  std::cout << "\nInput Files:\n";
  if (has_system) {
    std::cout << "  System topology:     " << system_prmtop << "\n";
    std::cout << "  System coordinates:  " << system_inpcrd << "\n";
  }
  if (has_fragment) {
    std::cout << "  Fragment topology:   " << fragment_prmtop << "\n";
  }
  if (!fragment_inpcrd.empty() && fragment_inpcrd != "none") {
    std::cout << "  Fragment coords:     " << fragment_inpcrd << "\n";
  }
  std::cout << "\nGCMC Parameters:\n";
  std::cout << "  Moves:        " << n_moves << "\n";
  std::cout << "  B value:      " << bvalue << "\n";
  std::cout << "  Temperature:  " << temperature << " K\n";
  std::cout << "  Mode:         " << mode << "\n";
  if (mode == "sphere") {
    std::cout << "  Sphere radius: " << sphere_radius << " Angstroms\n";
  }
  std::cout << "  Ghost molecules: " << nghost << "\n";
  if (freeze_fragments) {
    std::cout << "  Fragment geometry: FROZEN (rigid)\n";
  } else {
    std::cout << "  Fragment geometry: FLEXIBLE\n";
  }
  if (save_trajectory) {
    std::cout << "  Trajectory output: ENABLED\n";
    std::cout << "  Snapshot interval: every " << snapshot_interval << " moves\n";
    std::cout << "  Trajectory prefix:  " << trajectory_prefix << "\n";
  }
  std::cout << "\nNCMC Parameters:\n";
  std::cout << "  Perturbation steps: " << npert << "\n";
  std::cout << "  Propagation steps:  " << nprop << "\n";
  std::cout << "  Timestep:           " << timestep << " fs\n";
  if (md_steps > 0) {
    std::cout << "  MD steps between GCMC moves: " << md_steps << "\n";
  }
  if (mc_translation > 0.0 || mc_rotation > 0.0 || mc_torsion > 0.0) {
    std::cout << "\nMonte Carlo Move Parameters:\n";
    if (mc_translation > 0.0) {
      std::cout << "  Translation:    " << mc_translation << " Angstroms max displacement\n";
    }
    if (mc_rotation > 0.0) {
      std::cout << "  Rotation:       " << mc_rotation << " degrees max angle\n";
    }
    if (mc_torsion > 0.0) {
      std::cout << "  Torsion:        " << mc_torsion << " degrees max angle\n";
    }
    if (mc_frequency > 0) {
      std::cout << "  MC frequency:   Every " << mc_frequency << " GCMC moves\n";
    }
  }
  std::cout << "\n";

  try {
    AtomGraph system_ag;
    PhaseSpace system_ps;
    GhostMoleculeMetadata ghost_metadata;
    std::string fragment_resname;

    // Check if we need to build a combined topology automatically
    if (has_fragment) {
      // AUTOMATIC MODE: Build combined topology from separate files

      // Load base topology (or create empty one for fragment-only mode)
      AtomGraph base_topology;
      if (fragment_only_mode) {
        std::cout << "Fragment-only mode: No base system.\n";
        // Create an empty topology - just a placeholder
        // The combined topology will only contain ghost molecules
        base_topology = AtomGraph();
      } else {
        std::cout << "Loading base system topology from " << system_prmtop << "...\n";
        base_topology = AtomGraph(system_prmtop, ExceptionResponse::SILENT);
      }

      std::cout << "Loading fragment template from " << fragment_prmtop << "...\n";
      AtomGraph fragment_template(fragment_prmtop, ExceptionResponse::SILENT);

      // Get fragment residue name
      const ChemicalDetailsKit frag_cdk = fragment_template.getChemicalDetailsKit();
      if (fragment_template.getResidueCount() > 0) {
        const char4 res_name_char4 = frag_cdk.res_names[0];
        char res_name_buf[5] = {res_name_char4.x, res_name_char4.y,
                                res_name_char4.z, res_name_char4.w, '\0'};
        fragment_resname = std::string(res_name_buf);
        fragment_resname.erase(fragment_resname.find_last_not_of(" ") + 1);

        std::cout << "Fragment residue name: " << fragment_resname << "\n";
        std::cout << "Fragment has " << fragment_template.getAtomCount() << " atoms\n";
      } else {
        rtErr("Fragment topology has no residues", "main");
      }

      std::cout << "\nBuilding combined topology with " << nghost << " ghost molecules...\n";

      // Build combined topology automatically
      std::vector<int> ghost_residue_indices;
      std::vector<int2> ghost_atom_ranges;
      system_ag = buildTopologyWithGhosts(base_topology, fragment_template, nghost,
                                         &ghost_residue_indices, &ghost_atom_ranges);

      std::cout << "Combined topology created: " << system_ag.getAtomCount() << " atoms, "
                << system_ag.getMoleculeCount() << " molecules\n";

      // Build ghost metadata
      const int base_molecule_count = fragment_only_mode ? 0 : base_topology.getMoleculeCount();
      ghost_metadata = identifyGhostMolecules(system_ag, base_molecule_count, nghost);

      // Load base coordinates (if not fragment-only mode)
      PhaseSpace base_ps;
      if (!fragment_only_mode) {
        std::cout << "Loading base coordinates from " << system_inpcrd << "...\n";
        base_ps = PhaseSpace(system_inpcrd, CoordinateFileKind::AMBER_INPCRD);

        // Validate base coordinates
        if (base_ps.getAtomCount() != base_topology.getAtomCount()) {
          rtErr("Base coordinate count mismatch: base topology has " +
                std::to_string(base_topology.getAtomCount()) + " atoms, coordinates have " +
                std::to_string(base_ps.getAtomCount()) + " atoms.", "main");
        }
      } else {
        // Fragment-only: no base coordinates
        base_ps = PhaseSpace(0);
      }

      // Load fragment coordinates if provided
      PhaseSpace fragment_ps;
      if (!fragment_inpcrd.empty() && fragment_inpcrd != "none") {
        std::cout << "Loading fragment coordinates from " << fragment_inpcrd << "...\n";
        fragment_ps = PhaseSpace(fragment_inpcrd, CoordinateFileKind::AMBER_INPCRD);

        if (fragment_ps.getAtomCount() != fragment_template.getAtomCount()) {
          rtErr("Fragment coordinate count mismatch", "main");
        }
      } else {
        // Use fragment topology's default coordinates
        std::cout << "No fragment coordinates provided, using template geometry\n";
        fragment_ps = PhaseSpace(fragment_template.getAtomCount());
      }

      // Build combined coordinates: base + ghosts placed at origin
      std::cout << "Building combined coordinate set...\n";

      // Build coordinate vectors for combined system
      const int total_atoms = system_ag.getAtomCount();
      const int base_atom_count = fragment_only_mode ? 0 : base_topology.getAtomCount();
      std::vector<double> combined_xcrd(total_atoms, 0.0);
      std::vector<double> combined_ycrd(total_atoms, 0.0);
      std::vector<double> combined_zcrd(total_atoms, 0.0);

      // Copy base coordinates (if not fragment-only)
      if (!fragment_only_mode) {
        const PhaseSpaceReader base_psr = base_ps.data();
        for (int i = 0; i < base_topology.getAtomCount(); i++) {
          combined_xcrd[i] = base_psr.xcrd[i];
          combined_ycrd[i] = base_psr.ycrd[i];
          combined_zcrd[i] = base_psr.zcrd[i];
        }
      }

      // Place ghost molecules using fragment coordinates
      const PhaseSpaceReader frag_psr = fragment_ps.data();
      for (int ghost_idx = 0; ghost_idx < nghost; ghost_idx++) {
        const int atom_offset = base_atom_count + (ghost_idx * fragment_template.getAtomCount());
        for (int i = 0; i < fragment_template.getAtomCount(); i++) {
          combined_xcrd[atom_offset + i] = frag_psr.xcrd[i];
          combined_ycrd[atom_offset + i] = frag_psr.ycrd[i];
          combined_zcrd[atom_offset + i] = frag_psr.zcrd[i];
        }
      }

      // Create PhaseSpace and fill with coordinates using the proper API
      system_ps = PhaseSpace(system_ag.getAtomCount());

      // Prepare box dimensions
      std::vector<double> box_dims(6);
      if (!fragment_only_mode) {
        // Copy box dimensions from base
        const PhaseSpaceReader base_psr = base_ps.data();
        for (int i = 0; i < 6; i++) {
          box_dims[i] = base_psr.boxdim[i];
        }
      } else {
        // Fragment-only: set cubic box
        std::cout << "Setting simulation box: " << box_size << " x " << box_size
                  << " x " << box_size << " Angstroms\n";
        box_dims[0] = box_size;
        box_dims[1] = box_size;
        box_dims[2] = box_size;
        box_dims[3] = 90.0;
        box_dims[4] = 90.0;
        box_dims[5] = 90.0;
      }

      // Fill PhaseSpace with coordinates and box dimensions
      system_ps.fill(combined_xcrd, combined_ycrd, combined_zcrd,
                     TrajectoryKind::POSITIONS, CoordinateCycle::WHITE, 0, box_dims);

      // Upload coordinates and box dimensions to GPU
      #ifdef STORMM_USE_HPC

      system_ps.upload();

      #endif

      // Download and verify coordinates were initialized correctly
      #ifdef STORMM_USE_HPC

      system_ps.download();

      #endif
      const PhaseSpaceReader verify_psr = system_ps.data();
      std::cout << "Verification - Box dimensions: "
                << verify_psr.boxdim[0] << " x "
                << verify_psr.boxdim[1] << " x "
                << verify_psr.boxdim[2] << " Angstroms\n";
      std::cout << "DEBUG: First atom coordinates: "
                << verify_psr.xcrd[0] << ", "
                << verify_psr.ycrd[0] << ", "
                << verify_psr.zcrd[0] << "\n";

    } else {
      // MANUAL MODE: User provided pre-built combined topology
      std::cout << "Loading pre-built combined topology from " << system_prmtop << "...\n";
      system_ag = AtomGraph(system_prmtop, ExceptionResponse::SILENT);

      std::cout << "Loading coordinates from " << system_inpcrd << "...\n";
      system_ps = PhaseSpace(system_inpcrd, CoordinateFileKind::AMBER_INPCRD);

      // Validate that coordinates match topology
      if (system_ps.getAtomCount() != system_ag.getAtomCount()) {
        rtErr("Atom count mismatch: topology has " + std::to_string(system_ag.getAtomCount()) +
              " atoms, coordinates have " + std::to_string(system_ps.getAtomCount()) + " atoms.",
              "main");
      }

      std::cout << "System contains " << system_ag.getAtomCount() << " atoms in "
                << system_ag.getMoleculeCount() << " molecules.\n";

      // Auto-detect fragment residue name
      std::cout << "\nNo fragment topology specified. Searching for ghost molecules...\n";
      const ChemicalDetailsKit sys_cdk = system_ag.getChemicalDetailsKit();
      std::vector<std::string> found_residues;

      for (int res_idx = 0; res_idx < system_ag.getResidueCount(); res_idx++) {
        const char4 res_name_char4 = sys_cdk.res_names[res_idx];
        char res_name_buf[5] = {res_name_char4.x, res_name_char4.y,
                                res_name_char4.z, res_name_char4.w, '\0'};
        std::string res_name_str(res_name_buf);
        res_name_str.erase(res_name_str.find_last_not_of(" ") + 1);

        if (std::find(found_residues.begin(), found_residues.end(), res_name_str) ==
            found_residues.end()) {
          found_residues.push_back(res_name_str);
        }
      }

      std::cout << "Found residue types: ";
      for (const auto& name : found_residues) {
        std::cout << name << " ";
      }
      std::cout << "\n";

      // Default to HOH/WAT
      if (std::find(found_residues.begin(), found_residues.end(), "HOH") != found_residues.end()) {
        fragment_resname = "HOH";
      } else if (std::find(found_residues.begin(), found_residues.end(), "WAT") != found_residues.end()) {
        fragment_resname = "WAT";
      } else if (!found_residues.empty()) {
        fragment_resname = found_residues.back();
      } else {
        rtErr("No residue types found in system topology", "main");
      }

      std::cout << "Using residue name: " << fragment_resname << "\n";

      // Identify ghost molecules
      if (!identifyGhostMolecules(system_ag, fragment_resname, nghost, ghost_metadata)) {
        std::cerr << "\nError: Unable to identify ghost molecules.\n";
        std::cerr << "Please provide --fragment-prmtop to build topology automatically.\n";
        return 1;
      }
    }

    timer.assignTime(gcmc_setup_tm);

    std::cout << "\nGhost molecule metadata:\n";
    std::cout << "  Number of ghosts: " << ghost_metadata.n_ghost_molecules << "\n";
    std::cout << "  Base molecules:   " << ghost_metadata.base_molecule_count << "\n";
    std::cout << "  Base atoms:       " << ghost_metadata.base_atom_count << "\n";

    // Freeze fragment geometry if requested
    if (freeze_fragments) {
      freezeFragmentGeometry(system_ag, ghost_metadata);
    }

    // Build exclusion mask
    std::cout << "\nBuilding exclusion mask...\n";
    StaticExclusionMask exclusions(&system_ag);

    // Create thermostat
    std::cout << "Creating thermostat at " << temperature << " K...\n";
    const std::vector<double> bath_temps = {temperature};
    const std::vector<double> tau_values = {1.0};  // Relaxation time in ps
    const std::vector<int2> atom_ranges = {{0, system_ag.getAtomCount()}};
    Thermostat thermostat(&system_ag, ThermostatKind::ANDERSEN,
                         bath_temps, tau_values, atom_ranges,
                         1,  // random_seed
                         100, // cache_depth
                         PrecisionModel::DOUBLE,
                         32,  // stride
                         gpu);

    // Enable SHAKE/RATTLE constraints for rigid bonds (e.g., C-H bonds)
    // Constraints are applied to ALL atoms (both coupled and ghost molecules)
    // to maintain correct molecular geometry during MD propagation
    thermostat.setGeometryConstraints(ApplyConstraints::YES);
    std::cout << "Constraints enabled: SHAKE/RATTLE will be applied to all atoms\n";

    // Prepare output file names
    const std::string ghost_file = output_prefix.empty() ?
      "gcmc-ghosts.txt" : output_prefix + "-ghosts.txt";
    const std::string log_file = output_prefix.empty() ?
      "gcmc.log" : output_prefix + ".log";
    const std::string final_crd = output_prefix.empty() ?
      "gcmc-final.inpcrd" : output_prefix + "-final.inpcrd";

    // Check if output files exist and handle overwrite
    if (!overwrite) {
      std::ifstream test_ghost(ghost_file);
      std::ifstream test_log(log_file);
      std::ifstream test_final(final_crd);
      if (test_ghost.good() || test_log.good() || test_final.good()) {
        std::cerr << "Error: Output files already exist. Use -O to overwrite.\n";
        test_ghost.close();
        test_log.close();
        test_final.close();
        return 1;
      }
    }

    // Create appropriate sampler based on mode and npert
    const bool use_ncmc = (npert > 0);
    if (use_ncmc) {
      std::cout << "\nCreating NCMC " << mode << " sampler with B = " << bvalue << "...\n";
    } else {
      std::cout << "\nCreating basic GCMC " << mode << " sampler with B = " << bvalue << "...\n";
    }

    // Use unique_ptr to manage sampler lifetime
    std::unique_ptr<GCMCSampler> sampler;

    if (mode == "sphere") {
      // Find reference atoms for sphere center
      std::vector<int> ref_atoms = findReferenceAtoms(system_ag, ref_atoms_str);
      std::cout << "Using " << ref_atoms.size() << " atoms for sphere center calculation.\n";

      if (use_ncmc) {
        // Create NCMC sphere sampler
        sampler = std::make_unique<NCMCSampler>(
          &system_ag, &system_ps, &exclusions, &thermostat, temperature,
          ghost_metadata, npert, nprop, timestep,
          std::vector<double>(),  // Use default linear schedule
          false,  // Don't record trajectory
          ref_atoms, sphere_radius,
          mu_ex,
          standard_volume,
          bvalue,  // Adams B parameter
          0.0,     // adams_shift
          nghost,  // max_N
          gb_model,  // GB implicit solvent model
          fragment_resname,
          ghost_file,
          log_file);
      } else {
        // Create basic GCMC sphere sampler (instant GCMC)
        sampler = std::make_unique<GCMCSphereSampler>(
          &system_ag, &system_ps, &exclusions, &thermostat, temperature,
          ghost_metadata,
          ref_atoms, sphere_radius,
          mu_ex,
          standard_volume,
          bvalue,  // Adams B parameter
          0.0,     // adams_shift
          nghost,  // max_N
          gb_model,  // GB implicit solvent model
          fragment_resname,
          ghost_file,
          log_file);
      }
    } else {
      if (use_ncmc) {
        // Create NCMC system-wide sampler
        sampler = std::make_unique<NCMCSystemSampler>(
          &system_ag, &system_ps, &exclusions, &thermostat, temperature,
          ghost_metadata, npert, nprop, timestep,
          std::vector<double>(),  // Use default linear schedule
          false,  // Don't record trajectory
          mu_ex,
          standard_volume,
          bvalue,  // Adams B parameter
          0.0,     // adams_shift
          gb_model,  // GB implicit solvent model
          fragment_resname,
          ghost_file,
          log_file);
      } else {
        // Create basic GCMC system-wide sampler (instant GCMC)
        sampler = std::make_unique<GCMCSystemSampler>(
          &system_ag, &system_ps, &exclusions, &thermostat, temperature,
          ghost_metadata,
          mu_ex,
          standard_volume,
          bvalue,  // Adams B parameter
          0.0,     // adams_shift
          gb_model,  // GB implicit solvent model
          fragment_resname,
          ghost_file,
          log_file);
      }
    }

    // Register Monte Carlo movers if requested
    if (mc_translation > 0.0) {
      sampler->enableTranslationMoves(mc_translation);
      std::cout << "Enabled translation moves with max displacement: " << mc_translation << " Angstroms\n";
    }
    if (mc_rotation > 0.0) {
      sampler->enableRotationMoves(mc_rotation);
      std::cout << "Enabled rotation moves with max angle: " << mc_rotation << " degrees\n";
    }
    if (mc_torsion > 0.0) {
      sampler->enableTorsionMoves(mc_torsion);
      std::cout << "Enabled torsion moves with max angle: " << mc_torsion << " degrees\n";
    }

    // Enable adaptive B protocol if requested
    if (adaptive_b) {
      // Adjust n_moves if using adaptive B to match total of all stages
      int total_adaptive_moves = stage1_moves + stage2_moves + stage3_moves;

      if (mode == "system") {
        if (use_ncmc) {
          NCMCSystemSampler* system_sampler = static_cast<NCMCSystemSampler*>(sampler.get());
          system_sampler->enableAdaptiveB(stage1_moves, stage2_moves, stage3_moves,
                                          b_discovery, target_occupancy, coarse_rate,
                                          fine_rate, b_min, b_max);
        } else {
          GCMCSystemSampler* system_sampler = static_cast<GCMCSystemSampler*>(sampler.get());
          system_sampler->enableAdaptiveB(stage1_moves, stage2_moves, stage3_moves,
                                          b_discovery, target_occupancy, coarse_rate,
                                          fine_rate, b_min, b_max);
        }

        std::cout << "\nAdaptive B Protocol enabled:\n";
        std::cout << "  Stage 1 (Discovery): " << stage1_moves << " moves at B=" << b_discovery << "\n";
        std::cout << "  Stage 2 (Coarse Equilibration): " << stage2_moves << " moves, target "
                  << (target_occupancy * 100) << "% occupancy\n";
        std::cout << "  Stage 3 (Fine Annealing): " << stage3_moves << " moves, annealing to 0\n";
        std::cout << "  Total moves: " << total_adaptive_moves << "\n\n";
      } else {
        std::cerr << "Warning: Adaptive B protocol is currently only supported in system mode.\n";
        std::cerr << "         Proceeding with fixed B value.\n\n";
      }
    }

    // Check for hybrid mode
    if (hybrid_mode) {
      // ============================================================
      // Hybrid MD/MC Simulation Mode
      // ============================================================

      std::cout << "\n========================================\n";
      std::cout << "   Hybrid MD/MC Simulation Mode       \n";
      std::cout << "========================================\n\n";
      std::cout << "Configuration:\n";
      std::cout << "  Total MD steps:    " << hybrid_md_steps << "\n";
      std::cout << "  Move frequency:    every " << move_frequency << " steps\n";
      std::cout << "  GCMC probability:  " << (p_gcmc * 100.0) << "%\n";
      std::cout << "  MC probability:    " << ((1.0 - p_gcmc) * 100.0) << "%\n";
      std::cout << "\n";

      timer.assignTime(gcmc_run_tm);

      // Run hybrid simulation (only works with system-wide samplers)
      if (mode == "system") {
        if (use_ncmc) {
          NCMCSystemSampler* system_sampler = static_cast<NCMCSystemSampler*>(sampler.get());
          system_sampler->runHybridSimulation(hybrid_md_steps, move_frequency, p_gcmc);
        } else {
          GCMCSystemSampler* system_sampler = static_cast<GCMCSystemSampler*>(sampler.get());
          system_sampler->runHybridSimulation(hybrid_md_steps, move_frequency, p_gcmc);
        }
      } else {
        rtErr("Hybrid mode is currently only supported for system-wide sampling (--mode system)", "main");
      }

    } else {
      // ============================================================
      // Traditional GCMC Mode (existing code)
      // ============================================================

      std::cout << "\nStarting GCMC simulation...\n";
      std::cout << "========================================\n\n";

      timer.assignTime(gcmc_run_tm);

      // Determine the actual number of moves to run
      int actual_moves = n_moves;
      if (adaptive_b && mode == "system") {
        actual_moves = stage1_moves + stage2_moves + stage3_moves;
      }

      // Run GCMC cycles
      for (int move = 0; move < actual_moves; move++) {
      // Update adaptive B value if enabled
      if (adaptive_b && mode == "system") {
        if (use_ncmc) {
          NCMCSystemSampler* system_sampler = static_cast<NCMCSystemSampler*>(sampler.get());
          system_sampler->computeAdaptiveB(move);
        } else {
          GCMCSystemSampler* system_sampler = static_cast<GCMCSystemSampler*>(sampler.get());
          system_sampler->computeAdaptiveB(move);
        }
      }
      // Only update sphere center and classify molecules for sphere mode
      if (mode == "sphere") {
        if (use_ncmc) {
          NCMCSampler* sphere_sampler = static_cast<NCMCSampler*>(sampler.get());
          sphere_sampler->updateSphereCenter();
          sphere_sampler->classifyMolecules();
        } else {
          GCMCSphereSampler* sphere_sampler = static_cast<GCMCSphereSampler*>(sampler.get());
          sphere_sampler->updateSphereCenter();
          sphere_sampler->classifyMolecules();
        }
      }

      // Apply MC moves between GCMC moves if requested
      if (mc_frequency > 0 && move > 0 && move % mc_frequency == 0) {
        int n_mc_accepted = sampler->attemptMCMovesOnAllMolecules();
        if (n_mc_accepted > 0 && (move + 1) % 100 == 0) {
          std::cout << "  MC moves: " << n_mc_accepted << " accepted\n";
        }
      }

      // Run one GCMC cycle (insertion or deletion)
      bool accepted = false;
      if (mode == "sphere") {
        if (use_ncmc) {
          NCMCSampler* sphere_sampler = static_cast<NCMCSampler*>(sampler.get());
          accepted = sphere_sampler->runGCMCCycle();

          // Run MD steps between GCMC moves if requested
          if (md_steps > 0) {
            sphere_sampler->propagateSystem(md_steps);
          }
        } else {
          GCMCSphereSampler* sphere_sampler = static_cast<GCMCSphereSampler*>(sampler.get());
          accepted = sphere_sampler->runGCMCCycle();

          // Run MD steps between GCMC moves if requested
          if (md_steps > 0) {
            sphere_sampler->propagateSystem(md_steps);
          }
        }
      } else {
        if (use_ncmc) {
          NCMCSystemSampler* system_sampler = static_cast<NCMCSystemSampler*>(sampler.get());
          accepted = system_sampler->runGCMCCycle();

          // Run MD steps between GCMC moves if requested
          if (md_steps > 0) {
            system_sampler->propagateSystem(md_steps);
          }
        } else {
          GCMCSystemSampler* system_sampler = static_cast<GCMCSystemSampler*>(sampler.get());
          accepted = system_sampler->runGCMCCycle();

          // Run MD steps between GCMC moves if requested
          if (md_steps > 0) {
            system_sampler->propagateSystem(md_steps);
          }
        }
      }

      // Save trajectory snapshot if requested
      if (save_trajectory && (move + 1) % snapshot_interval == 0) {
        std::ostringstream snapshot_filename;
        snapshot_filename << trajectory_prefix << "_" << std::setfill('0') << std::setw(6) << (move + 1) << ".pdb";

        std::cout << "Saving snapshot: " << snapshot_filename.str() << " (N=" << sampler->getActiveCount() << ")\n";

        // Get active atom indices and write filtered PDB
        const std::vector<int> active_atoms = sampler->getActiveAtomIndices();

        // Download coordinates from GPU to CPU before writing PDB
        #ifdef STORMM_USE_HPC

        system_ps.download();

        #endif

        if (active_atoms.empty()) {
          // No active molecules - write empty PDB
          std::ofstream pdb_file(snapshot_filename.str());
          pdb_file << "REMARK   No active molecules\n";
          pdb_file << "END\n";
          pdb_file.close();
        } else {
          // Write PDB with only active atoms
          std::ofstream pdb_file(snapshot_filename.str());
          const PhaseSpaceReader psr = system_ps.data();
          const ChemicalDetailsKit cdk = system_ag.getChemicalDetailsKit();

          // Write box dimensions
          pdb_file << std::fixed << std::setprecision(3);
          pdb_file << "CRYST1"
                   << std::setw(9) << psr.boxdim[0]
                   << std::setw(9) << psr.boxdim[1]
                   << std::setw(9) << psr.boxdim[2]
                   << std::setw(7) << psr.boxdim[3]
                   << std::setw(7) << psr.boxdim[4]
                   << std::setw(7) << psr.boxdim[5]
                   << " P 1           1\n";

          // Write active atoms
          int pdb_serial = 1;
          for (int atom_idx : active_atoms) {
            // Use getResidueIndex() API instead of res_limits array to work around
            // bug in AtomGraph combination constructor
            const int res_idx = system_ag.getResidueIndex(atom_idx);
            const char* atom_name_ptr = reinterpret_cast<const char*>(cdk.atom_names + atom_idx);
            const char* res_name_ptr = reinterpret_cast<const char*>(cdk.res_names + res_idx);

            pdb_file << "ATOM  "
                     << std::setw(5) << pdb_serial
                     << " "
                     << std::left << std::setw(4) << std::string(atom_name_ptr, 4) << std::right
                     << " "
                     << std::left << std::setw(3) << std::string(res_name_ptr, 3) << std::right
                     << " A"
                     << std::setw(4) << (res_idx + 1)
                     << "    "
                     << std::setw(8) << psr.xcrd[atom_idx]
                     << std::setw(8) << psr.ycrd[atom_idx]
                     << std::setw(8) << psr.zcrd[atom_idx]
                     << "  1.00  0.00           "
                     << std::string(atom_name_ptr, 1) << " \n";
            pdb_serial++;
          }
          pdb_file << "END\n";
          pdb_file.close();
        }
      }

      // Print progress every 100 moves or at specific milestones
      if ((move + 1) % 100 == 0 || move == 0 || move == actual_moves - 1) {
        const auto& stats = sampler->getStatistics();
        std::cout << "Move " << std::setw(6) << (move + 1) << " / " << actual_moves << ": ";
        std::cout << "N = " << std::setw(3) << sampler->getActiveCount() << " | ";
        std::cout << "Ghosts = " << std::setw(3) << sampler->getGhostCount() << " | ";
        std::cout << "Accept = " << std::fixed << std::setprecision(1)
                  << std::setw(5) << stats.getAcceptanceRate() << "% | ";
        std::cout << "Ins = " << std::setw(5) << stats.getInsertionAcceptanceRate() << "% | ";
        std::cout << "Del = " << std::setw(5) << stats.getDeletionAcceptanceRate() << "%";

        // Add current B value if adaptive B is enabled
        if (adaptive_b && mode == "system") {
          double current_b;
          if (use_ncmc) {
            NCMCSystemSampler* system_sampler = static_cast<NCMCSystemSampler*>(sampler.get());
            current_b = system_sampler->getCurrentB();
          } else {
            GCMCSystemSampler* system_sampler = static_cast<GCMCSystemSampler*>(sampler.get());
            current_b = system_sampler->getCurrentB();
          }
          std::cout << " | B = " << std::fixed << std::setprecision(2) << current_b;
        }
        std::cout << "\n";
        // CRITICAL: Reset precision to prevent corruption of LJ parameters
        // Some code path serializes/deserializes topology data through text I/O
        std::cout << std::defaultfloat << std::setprecision(6);
      }

      // Occasionally flush output files
      if ((move + 1) % 1000 == 0) {
        sampler->writeGhostSnapshot();
      }
    }

    }  // End of traditional GCMC mode

    timer.assignTime(output_tm);

    // Write final output
    std::cout << "\n========================================\n";
    std::cout << "         GCMC Simulation Complete        \n";
    std::cout << "========================================\n\n";

    // Print final statistics
    const auto& stats = sampler->getStatistics();
    std::cout << "Final Statistics:\n";
    std::cout << "  Total moves:           " << stats.n_moves << "\n";
    std::cout << "  Accepted moves:        " << stats.n_accepted << "\n";
    std::cout << "  Overall acceptance:    " << std::fixed << std::setprecision(2)
              << stats.getAcceptanceRate() << "%\n";
    std::cout << "  Insertion attempts:    " << stats.n_inserts << "\n";
    std::cout << "  Insertion accepts:     " << stats.n_accepted_inserts << "\n";
    std::cout << "  Insertion acceptance:  " << stats.getInsertionAcceptanceRate() << "%\n";
    std::cout << "  Deletion attempts:     " << stats.n_deletes << "\n";
    std::cout << "  Deletion accepts:      " << stats.n_accepted_deletes << "\n";
    std::cout << "  Deletion acceptance:   " << stats.getDeletionAcceptanceRate() << "%\n";
    std::cout << "\nFinal State:\n";
    std::cout << "  Active molecules:      " << sampler->getActiveCount() << "\n";
    std::cout << "  Ghost molecules:       " << sampler->getGhostCount() << "\n";

    // Print MC move statistics if MC moves were enabled
    if (mc_translation > 0.0 || mc_rotation > 0.0 || mc_torsion > 0.0) {
      std::cout << "\nMonte Carlo Move Statistics:\n";
      auto mc_stats = sampler->getMCStatistics();
      for (const auto& [move_type, stats] : mc_stats) {
        if (stats.n_attempted > 0) {
          std::cout << "  " << move_type << ":\n";
          std::cout << "    Attempted: " << stats.n_attempted << "\n";
          std::cout << "    Accepted:  " << stats.n_accepted << "\n";
          std::cout << "    Acceptance: " << std::fixed << std::setprecision(2)
                    << stats.getAcceptanceRate() * 100.0 << "%\n";
        }
      }
    }

    // Print adaptive B results if enabled
    if (adaptive_b && mode == "system") {
      std::cout << "\nAdaptive B Protocol Results:\n";

      if (use_ncmc) {
        NCMCSystemSampler* system_sampler = static_cast<NCMCSystemSampler*>(sampler.get());
        std::cout << "  Maximum capacity (N_max): " << system_sampler->getMaxFragments() << " fragments\n";
        std::cout << "  Final B value:            " << std::fixed << std::setprecision(3)
                  << system_sampler->getCurrentB() << "\n";
        std::cout << "  Final stage:              ";
        switch (system_sampler->getCurrentStage()) {
          case AnnealingStage::DISCOVERY:
            std::cout << "Discovery\n";
            break;
          case AnnealingStage::COARSE:
            std::cout << "Coarse Equilibration\n";
            break;
          case AnnealingStage::FINE:
            std::cout << "Fine Annealing\n";
            break;
          case AnnealingStage::PRODUCTION:
            std::cout << "Production (complete)\n";
            break;
        }
      } else {
        GCMCSystemSampler* system_sampler = static_cast<GCMCSystemSampler*>(sampler.get());
        std::cout << "  Maximum capacity (N_max): " << system_sampler->getMaxFragments() << " fragments\n";
        std::cout << "  Final B value:            " << std::fixed << std::setprecision(3)
                  << system_sampler->getCurrentB() << "\n";
        std::cout << "  Final stage:              ";
        switch (system_sampler->getCurrentStage()) {
          case AnnealingStage::DISCOVERY:
            std::cout << "Discovery\n";
            break;
          case AnnealingStage::COARSE:
            std::cout << "Coarse Equilibration\n";
            break;
          case AnnealingStage::FINE:
            std::cout << "Fine Annealing\n";
            break;
          case AnnealingStage::PRODUCTION:
            std::cout << "Production (complete)\n";
            break;
        }
      }

      std::cout << "  Strongest binders:        " << sampler->getActiveCount() << " fragments remain\n";
      std::cout << "\nThe " << sampler->getActiveCount()
                << " remaining fragments represent the strongest binding sites.\n";
    }

    // Write final coordinates
    std::cout << "\nWriting final coordinates to " << final_crd << "...\n";

    // Download coordinates from GPU to CPU before writing
    #ifdef STORMM_USE_HPC

    system_ps.download();

    #endif

    system_ps.exportToFile(final_crd, 0.0,  // current_time
                   TrajectoryKind::POSITIONS,
                   CoordinateFileKind::AMBER_INPCRD,
                   overwrite ? PrintSituation::OVERWRITE : PrintSituation::OPEN_NEW);

    // Write final ghost snapshot
    sampler->writeGhostSnapshot();

    std::cout << "\nOutput files:\n";
    std::cout << "  Ghost IDs:     " << ghost_file << "\n";
    std::cout << "  Log file:      " << log_file << "\n";
    std::cout << "  Final coords:  " << final_crd << "\n";

    // Print timing information
    timer.assignTime(output_tm);
    std::cout << "\nTiming Information:\n";
    timer.printResults();

    std::cout << "\nGCMC simulation completed successfully.\n";

  } catch (const std::exception& e) {
    std::cerr << "\nError during GCMC simulation: " << e.what() << "\n";
    return 1;
  }

  return 0;
}
