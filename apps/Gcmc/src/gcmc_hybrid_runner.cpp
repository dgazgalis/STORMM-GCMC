#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>
#include "copyright.h"

#ifdef STORMM_USE_HPC
#  include "Accelerator/gpu_details.h"
#  include "Accelerator/hpc_config.h"
#  include "Accelerator/hybrid.h"
#endif

#include "Namelists/command_line_parser.h"
#include "Namelists/namelist_emulator.h"

#include "Potential/static_exclusionmask.h"
#include "Reporting/error_format.h"
#include "Sampling/gcmc_sampler.h"
#include "Structure/structure_enumerators.h"
#include "Topology/atomgraph.h"
#include "Topology/atomgraph_enumerators.h"
#include "Trajectory/phasespace.h"
#include "Trajectory/thermostat.h"
#include "Trajectory/trajectory_enumerators.h"

using namespace stormm;
using namespace stormm::card;
using namespace stormm::energy;
using namespace stormm::errors;
using namespace stormm::namelist;
using namespace stormm::sampling;
using namespace stormm::structure;
using namespace stormm::topology;
using namespace stormm::trajectory;

namespace {

GhostMoleculeMetadata buildSystemWithGhosts(const std::string &protein_prmtop,
                                            const std::string &protein_inpcrd,
                                            const std::string &fragment_prmtop,
                                            const std::string &fragment_inpcrd,
                                            const int n_ghosts,
                                            AtomGraph *combined_topology,
                                            PhaseSpace *phase_space,
                                            std::vector<double> *box_dims)
{
  AtomGraph fragment(fragment_prmtop, ExceptionResponse::WARN);

  AtomGraph base_topology;
  int base_molecule_count = 0;
  int base_atom_count = 0;

  if (!protein_prmtop.empty()) {
    base_topology = AtomGraph(protein_prmtop, ExceptionResponse::WARN);
    base_molecule_count = base_topology.getMoleculeCount();
    base_atom_count = base_topology.getAtomCount();
  }

  *combined_topology = buildTopologyWithGhosts(base_topology, fragment, n_ghosts);

  GhostMoleculeMetadata ghost_metadata =
      identifyGhostMolecules(*combined_topology, base_molecule_count, n_ghosts);

  const int total_atoms = combined_topology->getAtomCount();
  std::vector<double> xcrd(total_atoms, 0.0);
  std::vector<double> ycrd(total_atoms, 0.0);
  std::vector<double> zcrd(total_atoms, 0.0);

  PhaseSpace fragment_ps(fragment_inpcrd, CoordinateFileKind::AMBER_INPCRD);
  const PhaseSpaceReader frag_psr = fragment_ps.data();
  const int fragment_natoms = fragment.getAtomCount();

  if (fragment_ps.getAtomCount() != fragment_natoms) {
    rtErr("Fragment coordinate count mismatch", "gcmc_hybrid_runner");
  }

  std::random_device rd;
  std::mt19937 gen(rd());

  if (!protein_inpcrd.empty()) {
    PhaseSpace protein_ps(protein_inpcrd, CoordinateFileKind::AMBER_INPCRD);
    const PhaseSpaceReader protein_psr = protein_ps.data();

    if (protein_ps.getAtomCount() != ghost_metadata.base_atom_count) {
      rtErr("Protein coordinate count mismatch", "gcmc_hybrid_runner");
    }

    for (int i = 0; i < ghost_metadata.base_atom_count; i++) {
      xcrd[i] = protein_psr.xcrd[i];
      ycrd[i] = protein_psr.ycrd[i];
      zcrd[i] = protein_psr.zcrd[i];
    }

    const double box_threshold = 1.0;
    if (protein_psr.boxdim[0] > box_threshold &&
        protein_psr.boxdim[1] > box_threshold &&
        protein_psr.boxdim[2] > box_threshold)
    {
      for (int i = 0; i < 6; i++) {
        (*box_dims)[i] = protein_psr.boxdim[i];
      }
    }
    else {
      const double default_box_size = 25.0;
      (*box_dims)[0] = default_box_size;
      (*box_dims)[1] = default_box_size;
      (*box_dims)[2] = default_box_size;
      (*box_dims)[3] = 90.0;
      (*box_dims)[4] = 90.0;
      (*box_dims)[5] = 90.0;
      std::cout << "  Note: protein input lacks box dimensions, using "
                << default_box_size << " Å cubic box\n";
    }
  }
  else {
    const double default_box_size = 25.0;
    (*box_dims)[0] = default_box_size;
    (*box_dims)[1] = default_box_size;
    (*box_dims)[2] = default_box_size;
    (*box_dims)[3] = 90.0;
    (*box_dims)[4] = 90.0;
    (*box_dims)[5] = 90.0;
    std::cout << "  No protein coordinates supplied; using "
              << default_box_size << " Å cubic box\n";
  }

  std::uniform_real_distribution<double> dis_x(0.0, (*box_dims)[0]);
  std::uniform_real_distribution<double> dis_y(0.0, (*box_dims)[1]);
  std::uniform_real_distribution<double> dis_z(0.0, (*box_dims)[2]);

  double frag_com_x = 0.0, frag_com_y = 0.0, frag_com_z = 0.0;
  for (int i = 0; i < fragment_natoms; i++) {
    frag_com_x += frag_psr.xcrd[i];
    frag_com_y += frag_psr.ycrd[i];
    frag_com_z += frag_psr.zcrd[i];
  }
  frag_com_x /= fragment_natoms;
  frag_com_y /= fragment_natoms;
  frag_com_z /= fragment_natoms;

  for (int ghost_idx = 0; ghost_idx < ghost_metadata.n_ghost_molecules; ghost_idx++) {
    const double rand_x = dis_x(gen);
    const double rand_y = dis_y(gen);
    const double rand_z = dis_z(gen);
    const int atom_offset = ghost_metadata.base_atom_count + (ghost_idx * fragment_natoms);
    for (int i = 0; i < fragment_natoms; i++) {
      xcrd[atom_offset + i] = frag_psr.xcrd[i] - frag_com_x + rand_x;
      ycrd[atom_offset + i] = frag_psr.ycrd[i] - frag_com_y + rand_y;
      zcrd[atom_offset + i] = frag_psr.zcrd[i] - frag_com_z + rand_z;
    }
  }

  *phase_space = PhaseSpace(total_atoms);
  phase_space->fill(xcrd, ycrd, zcrd,
                    TrajectoryKind::POSITIONS, CoordinateCycle::WHITE, 0, *box_dims);

  return ghost_metadata;
}

void zeroProteinMasses(AtomGraph *combined_topology, int protein_atom_count) {
  if (protein_atom_count <= 0) {
    return;
  }
  ChemicalDetailsKit cdk = combined_topology->getChemicalDetailsKit();
  double* masses = const_cast<double*>(cdk.masses);
  float* sp_masses = const_cast<float*>(cdk.sp_masses);
  double* inv_masses = const_cast<double*>(cdk.inv_masses);
  float* sp_inv_masses = const_cast<float*>(cdk.sp_inv_masses);

  for (int i = 0; i < protein_atom_count; i++) {
    masses[i] = 0.0;
    sp_masses[i] = 0.0f;
    inv_masses[i] = 0.0;
    sp_inv_masses[i] = 0.0f;
  }

  combined_topology->upload();
}

void zeroProteinVelocities(PhaseSpace *phase_space, int protein_atom_count) {
  if (protein_atom_count <= 0) {
    return;
  }
  PhaseSpaceWriter psw = phase_space->data();
  for (int i = 0; i < protein_atom_count; i++) {
    psw.xvel[i] = 0.0;
    psw.yvel[i] = 0.0;
    psw.zvel[i] = 0.0;
  }
}

} // namespace

//-------------------------------------------------------------------------------------------------
int main(int argc, const char* argv[]) {
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

  CommandLineParser clip("gcmc_hybrid.stormm",
                        "Hybrid GPU GCMC + MC sampler for protein / fragment systems");
  clip.addStandardApplicationInputs({ "-p", "-c", "-o" });

  NamelistEmulator* t_nml = clip.getNamelistPointer();

  t_nml->addKeyword("--fragment-prmtop", NamelistType::STRING, "");
  t_nml->addKeyword("--fragment-inpcrd", NamelistType::STRING, "");
  t_nml->addHelp("--fragment-prmtop", "Fragment topology file (required)");
  t_nml->addHelp("--fragment-inpcrd", "Fragment coordinate file (required)");

  t_nml->addKeyword("--nghost", NamelistType::INTEGER, "1000");
  t_nml->addHelp("--nghost", "Number of ghost templates for the fragment (default: 1000)");

  t_nml->addKeyword("--moves", NamelistType::INTEGER, "1000");
  t_nml->addKeyword("-n", NamelistType::INTEGER, "1000");
  t_nml->addHelp("--moves", "Number of GCMC cycles");
  t_nml->addHelp("-n", "Alias for --moves");

  t_nml->addKeyword("--temp", NamelistType::REAL, "300.0");
  t_nml->addHelp("--temp", "Simulation temperature in Kelvin");

  t_nml->addKeyword("--bvalue", NamelistType::REAL, "5.0");
  t_nml->addKeyword("-b", NamelistType::REAL, "5.0");
  t_nml->addHelp("--bvalue", "Adams B parameter");
  t_nml->addHelp("-b", "Alias for --bvalue");

  t_nml->addKeyword("--mu-ex", NamelistType::REAL, "0.0");
  t_nml->addKeyword("--standard-volume", NamelistType::REAL, "30.0");

  t_nml->addKeyword("--adaptive-b", NamelistType::BOOLEAN, "false");
  t_nml->addHelp("--adaptive-b", "Enable three-stage adaptive Adams-B control");
  t_nml->addKeyword("--stage1-moves", NamelistType::INTEGER, "300");
  t_nml->addKeyword("--stage2-moves", NamelistType::INTEGER, "300");
  t_nml->addKeyword("--stage3-moves", NamelistType::INTEGER, "300");
  t_nml->addKeyword("--b-discovery", NamelistType::REAL, "8.0");
  t_nml->addKeyword("--target-occupancy", NamelistType::REAL, "0.5");
  t_nml->addKeyword("--coarse-rate", NamelistType::REAL, "0.5");
  t_nml->addKeyword("--fine-rate", NamelistType::REAL, "0.25");
  t_nml->addKeyword("--b-min", NamelistType::REAL, "-5.0");
  t_nml->addKeyword("--b-max", NamelistType::REAL, "10.0");

  t_nml->addKeyword("--timestep", NamelistType::REAL, "2.0");
  t_nml->addKeyword("--md-steps", NamelistType::INTEGER, "50");

  t_nml->addKeyword("--mc-translation", NamelistType::REAL, "1.0");
  t_nml->addKeyword("--mc-rotation", NamelistType::REAL, "30.0");
  t_nml->addKeyword("--mc-frequency", NamelistType::INTEGER, "5");
  t_nml->addHelp("--mc-translation", "Max translation displacement in Å for MC moves");
  t_nml->addHelp("--mc-rotation", "Max rotation angle in degrees for MC moves");
  t_nml->addHelp("--mc-frequency", "Number of MC attempts per GCMC cycle");

  t_nml->addKeyword("--log-memory", NamelistType::BOOLEAN, "true");
  t_nml->addHelp("--log-memory", "Record GPU / RSS memory telemetry each cycle");

  clip.parseUserInput(argc, argv);

  const std::string protein_prmtop = t_nml->getStringValue("-p");
  const std::string protein_inpcrd = t_nml->getStringValue("-c");
  std::string output_prefix = t_nml->getStringValue("-o");
  const std::string fragment_prmtop = t_nml->getStringValue("--fragment-prmtop");
  const std::string fragment_inpcrd = t_nml->getStringValue("--fragment-inpcrd");

  if (protein_prmtop.empty() || protein_inpcrd.empty()) {
    rtErr("Protein prmtop and inpcrd must be supplied for the hybrid application.",
          "gcmc_hybrid_runner");
  }
  if (fragment_prmtop.empty() || fragment_inpcrd.empty()) {
    rtErr("Fragment prmtop / inpcrd must be supplied.", "gcmc_hybrid_runner");
  }
  if (output_prefix.empty()) {
    output_prefix = "gcmc_hybrid_output";
  }

  const int n_moves = std::max(t_nml->getIntValue("--moves"), t_nml->getIntValue("-n"));
  const double temperature = t_nml->getRealValue("--temp");
  const int n_ghosts = t_nml->getIntValue("--nghost");
  const double timestep = t_nml->getRealValue("--timestep");
  const int md_steps = t_nml->getIntValue("--md-steps");
  const double b_value = std::max(t_nml->getRealValue("--bvalue"), t_nml->getRealValue("-b"));
  const double mu_ex = t_nml->getRealValue("--mu-ex");
  const double standard_volume = t_nml->getRealValue("--standard-volume");

  const double mc_translation = t_nml->getRealValue("--mc-translation");
  const double mc_rotation = t_nml->getRealValue("--mc-rotation");
  const int mc_frequency = t_nml->getIntValue("--mc-frequency");
  const bool log_memory = t_nml->getBoolValue("--log-memory");

  const bool use_adaptive_b = t_nml->getBoolValue("--adaptive-b");
  const int stage1_moves = t_nml->getIntValue("--stage1-moves");
  const int stage2_moves = t_nml->getIntValue("--stage2-moves");
  const int stage3_moves = t_nml->getIntValue("--stage3-moves");
  const double b_discovery = t_nml->getRealValue("--b-discovery");
  const double target_occupancy = t_nml->getRealValue("--target-occupancy");
  const double coarse_rate = t_nml->getRealValue("--coarse-rate");
  const double fine_rate = t_nml->getRealValue("--fine-rate");
  const double b_min = t_nml->getRealValue("--b-min");
  const double b_max = t_nml->getRealValue("--b-max");

  std::cout << "\n========================================\n";
  std::cout << "Hybrid GPU GCMC + MC Sampler\n";
  std::cout << "========================================\n";
  std::cout << "Protein topology:  " << protein_prmtop << "\n";
  std::cout << "Protein coords:    " << protein_inpcrd << "\n";
  std::cout << "Fragment topology: " << fragment_prmtop << "\n";
  std::cout << "Fragment coords:   " << fragment_inpcrd << "\n";
  std::cout << "GCMC cycles:       " << n_moves << "\n";
  std::cout << "Ghost templates:   " << n_ghosts << "\n";
  std::cout << "Temperature:       " << temperature << " K\n";
  std::cout << "MD steps / cycle:  " << md_steps << "\n";
  std::cout << "MC translation:    " << mc_translation << " Å\n";
  std::cout << "MC rotation:       " << mc_rotation << " deg\n";
  std::cout << "MC attempts/cycle: " << mc_frequency << "\n";
  std::cout << "Output prefix:     " << output_prefix << "\n";
  std::cout << "Adaptive B:        " << (use_adaptive_b ? "ON" : "OFF") << "\n";
  std::cout << "Log memory:        " << (log_memory ? "ON" : "OFF") << "\n\n";

  try {
    AtomGraph combined_topology;
    PhaseSpace phase_space;
    std::vector<double> box_dims(6, 0.0);

    GhostMoleculeMetadata ghost_metadata =
        buildSystemWithGhosts(protein_prmtop, protein_inpcrd,
                              fragment_prmtop, fragment_inpcrd,
                              n_ghosts, &combined_topology, &phase_space, &box_dims);

    const int protein_atom_count = ghost_metadata.base_atom_count;
    std::cout << "Combined system:   " << combined_topology.getAtomCount() << " atoms, "
              << combined_topology.getMoleculeCount() << " molecules\n";
    std::cout << "Protein atoms:     " << protein_atom_count << "\n";
    std::cout << "Fragment templates:" << ghost_metadata.n_ghost_molecules << "\n";

    combined_topology.modifyAtomMobility(0, protein_atom_count, MobilitySetting::OFF);
    zeroProteinMasses(&combined_topology, protein_atom_count);
    zeroProteinVelocities(&phase_space, protein_atom_count);

    StaticExclusionMask exclusions(&combined_topology);

    Thermostat thermostat(combined_topology, ThermostatKind::LANGEVIN, temperature);
    thermostat.setTimeStep(timestep / 1000.0);
    thermostat.upload();

    std::string ghost_file = output_prefix + "_ghosts.txt";
    std::string log_file = output_prefix + "_gcmc.log";

    GCMCSystemSampler sampler(&combined_topology, &phase_space, &exclusions, &thermostat,
                              temperature, ghost_metadata, mu_ex, standard_volume,
                              b_value, 0.0, ImplicitSolventModel::NONE, "LIG",
                              ghost_file, log_file);

    if (mc_translation > 0.0) {
      sampler.enableTranslationMoves(mc_translation);
    }
    if (mc_rotation > 0.0) {
      sampler.enableRotationMoves(mc_rotation);
    }

    if (use_adaptive_b) {
      sampler.enableAdaptiveB(stage1_moves, stage2_moves, stage3_moves,
                              b_discovery, target_occupancy,
                              coarse_rate, fine_rate, b_min, b_max);
      std::cout << "Adaptive B schedule: [" << b_min << ", " << b_max << "] "
                << "discovery=" << b_discovery << " target="
                << target_occupancy << " (stage lengths: "
                << stage1_moves << ", " << stage2_moves << ", " << stage3_moves << ")\n";
    }
    else {
      std::cout << "Using constant Adams B = " << b_value << "\n";
    }

    sampler.invalidateEnergyCache();

    std::string occupancy_file = output_prefix + "_occupancy.dat";
    std::ofstream occ_out(occupancy_file);
    occ_out << "# Cycle  Active_Molecules\n";

    std::string memory_file = output_prefix + "_gpu_memory.dat";
    std::ofstream mem_out;
    if (log_memory) {
      mem_out.open(memory_file);
      if (!mem_out.is_open()) {
        rtErr("Cannot open memory file: " + memory_file, "gcmc_hybrid_runner");
      }
      mem_out << "# Cycle  GPU_Free_MB  GPU_Used_MB  GPU_Total_MB  RSS_MB  VMS_MB\n";
    }

    std::cout << "\nStarting simulation...\n";

    for (int cycle = 0; cycle < n_moves; cycle++) {
#ifdef STORMM_USE_HPC
      size_t free_bytes_before = 0UL;
      size_t total_bytes = 0UL;
      cudaMemGetInfo(&free_bytes_before, &total_bytes);
#endif
      const bool accepted = sampler.runGCMCCycle();

      for (int mc_attempt = 0; mc_attempt < mc_frequency; mc_attempt++) {
        sampler.attemptMCMovesOnAllMolecules();
      }

      sampler.propagateSystem(md_steps);

      if (log_memory) {
#ifdef STORMM_USE_HPC
        size_t free_bytes_after = 0UL;
        size_t total_bytes_after = 0UL;
        cudaError_t cuda_err = cudaMemGetInfo(&free_bytes_after, &total_bytes_after);
        double rss_mb = 0.0;
        double vms_mb = 0.0;
        std::ifstream status_file("/proc/self/status");
        if (status_file.is_open()) {
          std::string line;
          while (std::getline(status_file, line)) {
            if (line.rfind("VmRSS", 0) == 0) {
              size_t pos = line.find_first_of("0123456789");
              if (pos != std::string::npos) {
                rss_mb = std::stod(line.substr(pos)) / 1024.0;
              }
            }
            else if (line.rfind("VmSize", 0) == 0) {
              size_t pos = line.find_first_of("0123456789");
              if (pos != std::string::npos) {
                vms_mb = std::stod(line.substr(pos)) / 1024.0;
              }
            }
          }
          status_file.close();
        }
        if (cuda_err == cudaSuccess) {
          const double free_mb = static_cast<double>(free_bytes_after) / (1024.0 * 1024.0);
          const double total_mb = static_cast<double>(total_bytes_after) / (1024.0 * 1024.0);
          const double used_mb = total_mb - free_mb;
          mem_out << cycle << " "
                  << std::fixed << std::setprecision(2)
                  << free_mb << " " << used_mb << " " << total_mb << " "
                  << rss_mb << " " << vms_mb << "\n";
        }
        else {
          mem_out << cycle << " 0 0 0 0 0\n";
        }
#else
        mem_out << cycle << " 0 0 0 0 0\n";
#endif
        mem_out.flush();
      }

      occ_out << cycle << " " << sampler.getActiveCount() << "\n";
      occ_out.flush();

      std::cout << "Cycle " << std::setw(6) << cycle
                << "  Active=" << std::setw(4) << sampler.getActiveCount()
                << "  Move=" << (accepted ? "ACCEPT" : "REJECT") << "\n";
    }

    std::cout << "\nSimulation complete.\n";
  }
  catch (std::exception &e) {
    rtErr("Hybrid GCMC encountered an error: " + std::string(e.what()),
          "gcmc_hybrid_runner");
  }

  return 0;
}
