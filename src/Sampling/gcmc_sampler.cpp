#include <algorithm>
#include <memory>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <limits>
#include "copyright.h"
#include "Constants/behavior.h"
#include "Constants/generalized_born.h"
#include "Constants/symbol_values.h"
#include "Math/vector_ops.h"
#include "Potential/nonbonded_potential.h"
#include "Potential/soft_core_potentials.h"
#include "Potential/valence_potential.h"
#include "Potential/pme_util_lambda.h"
#include "Potential/map_density_lambda.h"
#include "Synthesis/atomgraph_synthesis.h"
#include "Synthesis/phasespace_synthesis.h"
#include "Synthesis/static_mask_synthesis.h"
#include "Synthesis/nonbonded_workunit.h"
#include "Potential/energy_enumerators.h"
#include "Synthesis/synthesis_enumerators.h"
#ifdef STORMM_USE_CUDA
#include "Potential/hpc_lambda_nonbonded.h"
#include "Potential/cacheresource.h"
#include "MolecularMechanics/hpc_lambda_dynamics.h"
#include "Accelerator/core_kernel_manager.h"
#include "Accelerator/gpu_details.h"
#include "hpc_gcmc_lambda.h"
#include <cuda_runtime.h>
#endif
#include "Reporting/error_format.h"
#include "Restraints/restraint_apparatus.h"
#include "gcmc_sampler.h"
#include "mc_mover.h"

namespace stormm {
namespace sampling {

using card::HybridTargetLevel;
using card::GpuDetails;
using constants::ExceptionResponse;
using constants::PrecisionModel;
using energy::evaluateBondTerms;
using energy::evaluateAngleTerms;
using energy::evaluateDihedralTerms;
using energy::evaluateLambdaScaledNonbonded;
using energy::NonbondedTheme;
using energy::StateVariable;
using energy::EvaluateEnergy;
using energy::ClashResponse;
using energy::AccumulationMethod;
using errors::rtErr;
using mm::dynaStep;
using mm::lambdaDynaStep;
using mm::launchLambdaDynamicsStep;
using restraints::RestraintApparatus;
using restraints::RestraintKit;
using stmath::computeBoxTransform;
using stmath::mean;
using symbols::avogadro_number;
using symbols::boltzmann_constant_gafs;
using symbols::boltzmann_constant_md;
using symbols::pi;
using synthesis::AtomGraphSynthesis;
using synthesis::PhaseSpaceSynthesis;
using synthesis::VwuGoal;
using synthesis::maximum_valence_work_unit_atoms;
using synthesis::small_block_max_atoms;
using topology::UnitCellType;
using topology::ImplicitSolventModel;
using trajectory::CartesianDimension;

namespace {
constexpr bool kGcmcDebugLogs = false;
}

//-------------------------------------------------------------------------------------------------
// GCMCMolecule implementation
//-------------------------------------------------------------------------------------------------
GCMCMolecule::GCMCMolecule() :
  resid{-1},
  status{GCMCMoleculeStatus::GHOST},
  lambda_vdw{0.0},
  lambda_ele{0.0},
  gpu_indices_cached{false}
{}

//-------------------------------------------------------------------------------------------------
GCMCMolecule::GCMCMolecule(int resid_in) :
  resid{resid_in},
  status{GCMCMoleculeStatus::GHOST},
  lambda_vdw{0.0},
  lambda_ele{0.0},
  gpu_indices_cached{false}
{}

//-------------------------------------------------------------------------------------------------
bool GCMCMolecule::isActive() const {
  return (std::abs(lambda_vdw - 1.0) < 1.0e-6 && std::abs(lambda_ele - 1.0) < 1.0e-6);
}

//-------------------------------------------------------------------------------------------------
bool GCMCMolecule::isGhost() const {
  return (std::abs(lambda_vdw) < 1.0e-6 && std::abs(lambda_ele) < 1.0e-6);
}

//-------------------------------------------------------------------------------------------------
double GCMCMolecule::getCombinedLambda() const {
  const double vdw_end = VDW_COUPLING_THRESHOLD;
  if (lambda_vdw < 1.0) {
    return lambda_vdw * vdw_end;
  } else {
    return vdw_end + lambda_ele * (1.0 - vdw_end);
  }
}

//-------------------------------------------------------------------------------------------------
// GCMCStatistics implementation
//-------------------------------------------------------------------------------------------------
GCMCStatistics::GCMCStatistics() :
  n_moves{0},
  n_accepted{0},
  n_inserts{0},
  n_deletes{0},
  n_accepted_inserts{0},
  n_accepted_deletes{0},
  n_explosions{0},
  n_left_sphere{0}
{}

//-------------------------------------------------------------------------------------------------
double GCMCStatistics::getAcceptanceRate() const {
  if (n_moves == 0) return 0.0;
  return 100.0 * static_cast<double>(n_accepted) / static_cast<double>(n_moves);
}

//-------------------------------------------------------------------------------------------------
double GCMCStatistics::getInsertionAcceptanceRate() const {
  if (n_inserts == 0) return 0.0;
  return 100.0 * static_cast<double>(n_accepted_inserts) / static_cast<double>(n_inserts);
}

//-------------------------------------------------------------------------------------------------
double GCMCStatistics::getDeletionAcceptanceRate() const {
  if (n_deletes == 0) return 0.0;
  return 100.0 * static_cast<double>(n_accepted_deletes) / static_cast<double>(n_deletes);
}

//-------------------------------------------------------------------------------------------------
void GCMCStatistics::reset() {
  n_moves = 0;
  n_accepted = 0;
  n_inserts = 0;
  n_deletes = 0;
  n_accepted_inserts = 0;
  n_accepted_deletes = 0;
  n_explosions = 0;
  n_left_sphere = 0;
  N_history.clear();
  acc_rate_history.clear();
  acceptance_probs.clear();
  insert_acceptance_probs.clear();
  delete_acceptance_probs.clear();
  move_resids.clear();
  outcomes.clear();
  insert_works.clear();
  delete_works.clear();
  accepted_insert_works.clear();
  accepted_delete_works.clear();
}

//-------------------------------------------------------------------------------------------------
// GCMCSampler implementation
//-------------------------------------------------------------------------------------------------
GCMCSampler::GCMCSampler(AtomGraph* topology,
                         PhaseSpace* ps,
                         StaticExclusionMask* exclusions,
                         Thermostat* thermostat,
                         double temperature,
                         const GhostMoleculeMetadata& ghost_metadata,
                         topology::ImplicitSolventModel gb_model,
                         const std::string& resname,
                         const std::string& ghost_file,
                         const std::string& log_file) :
  topology_{topology},
  phase_space_{ps},
  exclusions_{exclusions},  // IMPORTANT: This exclusion mask must be built from the
                            // combined topology (base + ghosts) returned by buildTopologyWithGhosts().
                            // It should include all internal exclusions for ghost molecules.
  thermostat_{thermostat},
  temperature_{temperature},
  resname_{resname},
  N_active_{0},
  scorecard_{1},  // 1 system
  gb_model_{gb_model},
  gb_workspace_{nullptr},
  se_synthesis_{nullptr},
  launcher_{nullptr},
  mmctrl_{nullptr},
  gpu_lambda_arrays_dirty_{false},
  coupled_indices_valid_{false},
  coupled_atom_count_{0}
{
  // Validate input pointers
  if (topology == nullptr || ps == nullptr || exclusions == nullptr) {
    rtErr("Cannot create GCMCSampler with null pointers", "GCMCSampler");
  }

  // Validate temperature for numerical stability
  if (temperature < 1.0) {
    rtErr("Temperature must be at least 1.0 K for numerical stability", "GCMCSampler");
  }

  // Calculate thermodynamic constants
  const double kB = symbols::boltzmann_constant;  // kcal/(mol·K)
  kT_ = kB * temperature_;  // kcal/mol
  beta_ = 1.0 / kT_;        // mol/kcal

  // Initialize molecules from metadata, not by scanning
  molecules_.reserve(ghost_metadata.n_ghost_molecules);

  // Get nonbonded kit to access per-atom LJ type indices
  const NonbondedKit<double> nbk_init = topology_->getDoublePrecisionNonbondedKit();

  for (int i = 0; i < ghost_metadata.n_ghost_molecules; i++) {
    GCMCMolecule mol;
    // Store molecule index (not residue index) for tracking purposes
    // ghost_residue_indices is a flat list of all residues, not per-molecule mapping
    mol.resid = ghost_metadata.ghost_molecule_indices[i];  // Use molecule index for tracking
    mol.status = GCMCMoleculeStatus::GHOST;  // All start as ghosts
    mol.lambda_vdw = 0.0;
    mol.lambda_ele = 0.0;

    // Get atom indices for this ghost molecule from the ranges
    const int2& atom_range = ghost_metadata.ghost_atom_ranges[i];

    // Validate atom range is within topology bounds
    if (atom_range.y > topology_->getAtomCount()) {
      rtErr("Ghost atom range exceeds topology size: [" +
            std::to_string(atom_range.x) + ", " + std::to_string(atom_range.y) +
            ") but topology has only " + std::to_string(topology_->getAtomCount()) + " atoms",
            "GCMCSampler");
    }

    // Pre-allocate all vectors for efficiency
    const int n_atoms_in_molecule = atom_range.y - atom_range.x;
    mol.atom_indices.reserve(n_atoms_in_molecule);
    mol.heavy_atom_indices.reserve(n_atoms_in_molecule);  // Worst case: all heavy
    mol.original_charges.reserve(n_atoms_in_molecule);
    mol.original_sigma.reserve(n_atoms_in_molecule);
    mol.original_epsilon.reserve(n_atoms_in_molecule);

    for (int atom_idx = atom_range.x; atom_idx < atom_range.y; atom_idx++) {
      mol.atom_indices.push_back(atom_idx);

      // Track heavy atoms for COM calculations
      if (topology_->getAtomicNumber(atom_idx) > 1) {
        mol.heavy_atom_indices.push_back(atom_idx);
      }

      // Store original parameters - use LJ type index, not atom index
      const int lj_type = nbk_init.lj_idx[atom_idx];
      mol.original_charges.push_back(topology_->getPartialCharge<double>(atom_idx));
      mol.original_sigma.push_back(topology_->getLennardJonesSigma<double>(lj_type));
      mol.original_epsilon.push_back(topology_->getLennardJonesEpsilon<double>(lj_type));
    }

    // Initialize GPU cache flag
    mol.gpu_indices_cached = false;

    molecules_.push_back(mol);
  }

  // Open log file
  log_stream_.open(log_file);
  log_stream_ << "# GCMC Sampler initialized\n";
  log_stream_ << "# Temperature: " << temperature_ << " K\n";
  log_stream_ << "# kT: " << kT_ << " kcal/mol\n";
  log_stream_ << "# Number of " << resname_ << " molecules: " << molecules_.size() << "\n";
  log_stream_ << "#\n";
  log_stream_ << "# Step MoveType ResID Work(kcal/mol) AcceptProb Outcome\n";
  log_stream_.flush();

  // Open ghost file
  ghost_stream_.open(ghost_file);
  ghost_stream_ << "# Frame, Ghost Residue IDs\n";
  ghost_stream_.flush();

  // Initialize statistics
  stats_ = GCMCStatistics();

  // Initialize RNG with random seed
  rng_ = Xoshiro256ppGenerator();

  // Initialize immutable LJ parameter cache
  // Cache all LJ type parameters from topology at construction time
  // This prevents topology corruption from affecting GCMC energy evaluations
  const int n_lj_types = topology_->getLJTypeCount();
  cached_lj_sigma_.reserve(n_lj_types);
  cached_lj_epsilon_.reserve(n_lj_types);

  log_stream_ << "# Caching " << n_lj_types << " LJ types for immutable topology:\n";
  for (int lj_type = 0; lj_type < n_lj_types; lj_type++) {
    const double sigma = topology_->getLennardJonesSigma<double>(lj_type);
    const double epsilon = topology_->getLennardJonesEpsilon<double>(lj_type);
    cached_lj_sigma_.push_back(sigma);
    cached_lj_epsilon_.push_back(epsilon);
    log_stream_ << "#   LJ type " << lj_type << ": sigma=" << sigma
                << ", epsilon=" << epsilon << "\n";
  }
  log_stream_ << "# LJ parameter cache initialized - topology now immutable\n";
  log_stream_ << "# Cache addresses: sigma=" << (void*)cached_lj_sigma_.data()
              << " epsilon=" << (void*)cached_lj_epsilon_.data() << "\n";
  log_stream_.flush();

  // Store original cache values as member variables to detect corruption
  original_cached_sigma_ = cached_lj_sigma_;
  original_cached_epsilon_ = cached_lj_epsilon_;

  // Initialize GPU-resident Hybrid arrays for lambda-scaled energy evaluation
  // These arrays maintain synchronized CPU/GPU copies for efficient GPU computation
  const int n_atoms = topology_->getAtomCount();

  lambda_vdw_.resize(n_atoms);
  lambda_ele_.resize(n_atoms);
  atom_sigma_.resize(n_atoms);
  atom_epsilon_.resize(n_atoms);

  // Maximum possible coupled atoms (conservative estimate)
  coupled_indices_.resize(n_atoms);
  energy_output_elec_.resize(n_atoms);
  energy_output_vdw_.resize(n_atoms);

  // Scalar outputs for GPU-side reduction
  total_elec_.resize(1);
  total_vdw_.resize(1);

  // Work accumulation arrays for GPU-side NCMC protocol
  work_accumulator_.resize(1);
  energy_before_elec_.resize(1);
  energy_before_vdw_.resize(1);
  energy_after_elec_.resize(1);
  energy_after_vdw_.resize(1);

  // Lambda scheduling arrays for GPU-side NCMC protocol
  // Will be dynamically resized when used (n_pert_steps+1 for schedule, n_mol_atoms for indices)
  lambda_schedule_.resize(101);  // Default: 100 perturbation steps + initial
  molecule_atom_indices_.resize(n_atoms);  // Max possible size
  molecule_atom_count_.resize(1);

  // Initialize MC move workspace arrays (GPU-accelerated moves)
  // Pre-allocate to maximum molecule size for efficiency
  // These arrays enable fully GPU-accelerated MC moves with no coordinate transfers
  int max_mol_atoms = 0;
  for (const auto& mol : molecules_) {
    if (static_cast<int>(mol.atom_indices.size()) > max_mol_atoms) {
      max_mol_atoms = mol.atom_indices.size();
    }
  }
  if (max_mol_atoms == 0) {
    max_mol_atoms = 100;  // Default fallback for empty molecule list
  }
  mc_atom_indices_.resize(max_mol_atoms);
  mc_saved_x_.resize(max_mol_atoms);
  mc_saved_y_.resize(max_mol_atoms);
  mc_saved_z_.resize(max_mol_atoms);
  mc_rotation_matrix_.resize(9);  // 3x3 rotation matrix
  mc_rotating_atoms_.resize(max_mol_atoms);

  log_stream_ << "# MC workspace initialized for max " << max_mol_atoms << " atoms per molecule\n";

  // Initialize GPU-side lambda modification workspace
  // This cache holds atom indices for GPU-direct lambda updates (eliminates 600+ uploads per 100 cycles)
  gpu_molecule_indices_.resize(max_mol_atoms);
  log_stream_ << "# GPU lambda modification cache initialized for max " << max_mol_atoms << " atoms\n";

  // Initialize GPU lambda state tracking
  gpu_lambda_arrays_dirty_ = false;

  // Initialize energy cache
  energy_cached_ = false;
  cached_energy_ = 0.0;
  cached_lambda_hash_ = 0;

  // Initialize Ewald electrostatics infrastructure for periodic systems
  // Note: Sets up PME grid (for future FFT use) and computes Ewald coefficient.
  // Currently only DIRECT SPACE (erfc splitting) is used; reciprocal space awaits FFT.
  // For single-system GCMC, we create lightweight wrappers around topology and phase space.
  if (topology_->getUnitCellType() != UnitCellType::NONE) {
    log_stream_ << "# Periodic system detected - initializing Ewald direct space\n";

    // Create topology synthesis from single topology
    std::vector<AtomGraph*> topo_vec = {topology_};
    std::vector<RestraintApparatus*> empty_restraints;
    topology_synthesis_ = new AtomGraphSynthesis(topo_vec, empty_restraints);

    // Create phase space synthesis - use wrapper constructor
    // PhaseSpaceSynthesis needs references to PhaseSpace objects, not pointers
    std::vector<PhaseSpace> ps_by_value;
    ps_by_value.push_back(*phase_space_);  // Copy for synthesis
    std::vector<AtomGraph*> ps_topo_vec = {topology_};
    // FIX: Use default precision bits (don't override with 1!)
    ps_synthesis_ = new PhaseSpaceSynthesis(ps_by_value, ps_topo_vec);

    // Create CellGrid for spatial decomposition
    const double cutoff = 12.0;  // Typical cutoff in Angstroms
    const double padding = 1.0;  // Cell padding
    const int mesh_subdivisions = 1;  // Standard value
    // FIX: Use llint for accumulator type (neighbor list images require integers)
    cell_grid_ = new CellGrid<double, llint, double, double4>(
        *ps_synthesis_, *topology_synthesis_,
        cutoff, padding, mesh_subdivisions,
        NonbondedTheme::ELECTROSTATIC);

    // Create PMIGrid from CellGrid
    pme_grid_ = new PMIGrid(*cell_grid_, NonbondedTheme::ELECTROSTATIC,
                            6);  // B-spline order 6

    // Compute Ewald coefficient from cutoff
    // Standard formula: α ≈ 3.2 / cutoff for good accuracy
    ewald_coeff_ = 3.2 / cutoff;

    log_stream_ << "#   Cutoff: " << cutoff << " A\n";
    log_stream_ << "#   B-spline order: 6\n";
    log_stream_ << "#   Ewald coefficient: " << ewald_coeff_ << " Angstrom^-1\n";
    log_stream_ << "#   Electrostatics: Ewald DIRECT SPACE only (reciprocal space awaits STORMM FFT)\n";
  } else {
    // Non-periodic system: use cutoff electrostatics without Ewald splitting
    topology_synthesis_ = nullptr;
    ps_synthesis_ = nullptr;
    cell_grid_ = nullptr;
    pme_grid_ = nullptr;
    ewald_coeff_ = 0.0;  // No Ewald for non-periodic systems
    log_stream_ << "# Non-periodic system: cutoff Coulomb electrostatics (1/r)\n";
  }

  log_stream_ << "# GPU arrays initialized for " << n_atoms << " atoms\n";
  log_stream_ << "# Energy caching enabled with invalidation on coordinate changes\n";

  // Initialize Generalized Born workspace if enabled
  if (gb_model_ != topology::ImplicitSolventModel::NONE) {
    // Create atom starts and counts arrays for the single system
    std::vector<int> atom_starts_vec = {0};
    std::vector<int> atom_counts_vec = {n_atoms};
    Hybrid<int> atom_starts(atom_starts_vec, "atom_starts");
    Hybrid<int> atom_counts(atom_counts_vec, "atom_counts");

    // Create GB workspace with automatic precision selection
    gb_workspace_ = new synthesis::ImplicitSolventWorkspace(atom_starts, atom_counts,
                                                            constants::PrecisionModel::DOUBLE);

    // Log GB model info
    const char* gb_model_name = "Unknown";
    switch (gb_model_) {
      case topology::ImplicitSolventModel::HCT_GB:
        gb_model_name = "HCT (Hawkins-Cramer-Truhlar)";
        break;
      case topology::ImplicitSolventModel::OBC_GB:
        gb_model_name = "OBC (Onufriev-Bashford-Case)";
        break;
      case topology::ImplicitSolventModel::OBC_GB_II:
        gb_model_name = "OBC-II";
        break;
      case topology::ImplicitSolventModel::NECK_GB:
        gb_model_name = "Neck GB";
        break;
      case topology::ImplicitSolventModel::NECK_GB_II:
        gb_model_name = "Neck GB-II";
        break;
      default:
        break;
    }
    log_stream_ << "# Generalized Born implicit solvent enabled: " << gb_model_name << "\n";
  } else {
    log_stream_ << "# Generalized Born implicit solvent: DISABLED\n";
  }

#ifdef STORMM_USE_CUDA
  // Create StaticExclusionMaskSynthesis for lambda dynamics kernel
  std::vector<StaticExclusionMask*> se_vec = {exclusions_};
  std::vector<int> topo_indices = {0};
  se_synthesis_ = new StaticExclusionMaskSynthesis(se_vec, topo_indices);

  // Create synthesis objects if they don't exist (for non-periodic systems)
  // The lambda dynamics kernel works for both periodic and non-periodic systems
  if (topology_synthesis_ == nullptr) {
    std::vector<AtomGraph*> topo_vec = {topology_};
    std::vector<RestraintApparatus*> empty_restraints;
    topology_synthesis_ = new AtomGraphSynthesis(topo_vec, empty_restraints);

    std::vector<PhaseSpace> ps_by_value;
    ps_by_value.push_back(*phase_space_);
    std::vector<AtomGraph*> ps_topo_vec = {topology_};
    ps_synthesis_ = new PhaseSpaceSynthesis(ps_by_value, ps_topo_vec);

    log_stream_ << "# Non-periodic system: Created synthesis objects for lambda dynamics\n";
    log_stream_.flush();
  }

  // Load nonbonded work units (required before creating CoreKlManager)
  // Now topology_synthesis_ is guaranteed to exist
  if (topology_synthesis_ != nullptr) {
    topology_synthesis_->loadNonbondedWorkUnits(*se_synthesis_);

    // Create GPU kernel launcher
    const GpuDetails gpu;
    launcher_ = new CoreKlManager(gpu, *topology_synthesis_);

    // Create MolecularMechanicsControls
    const double timestep = (thermostat_ != nullptr) ? thermostat_->getTimeStep() : 0.001;
    mmctrl_ = new MolecularMechanicsControls(timestep, 1);

    // Prime work unit counters for lambda dynamics
    mmctrl_->primeWorkUnitCounters(
        *launcher_,
        EvaluateForce::YES,
        EvaluateEnergy::NO,
        ClashResponse::NONE,
        VwuGoal::MOVE_PARTICLES,
        PrecisionModel::DOUBLE,
        PrecisionModel::DOUBLE,
        *topology_synthesis_);

    // Create cache resources for kernel thread blocks
    const PrecisionModel valence_prec = PrecisionModel::DOUBLE;
    // NOTE: CacheResource objects are now created locally in evaluateTotalEnergy()
    // and propagateSystem() to avoid CUDA pinned memory fragmentation issues.
    // Regular STORMM dynamics uses the same pattern - creating caches as stack
    // variables that are destroyed after each use.

    log_stream_ << "# Lambda dynamics GPU infrastructure initialized\n";
    log_stream_ << "#   Kernel launcher created\n";
    log_stream_ << "#   CacheResource objects will be created locally as needed\n";
    log_stream_.flush();
  }
#else
  se_synthesis_ = nullptr;
  launcher_ = nullptr;
  mmctrl_ = nullptr;
#endif

  log_stream_.flush();
}

//-------------------------------------------------------------------------------------------------
GCMCSampler::~GCMCSampler() {
  if (log_stream_.is_open()) {
    log_stream_.close();
  }
  if (ghost_stream_.is_open()) {
    ghost_stream_.close();
  }
#ifdef STORMM_USE_CUDA
  // NOTE: CacheResource objects are now created locally and don't need cleanup
  if (launcher_ != nullptr) {
    delete launcher_;
    launcher_ = nullptr;
  }
  if (mmctrl_ != nullptr) {
    delete mmctrl_;
    mmctrl_ = nullptr;
  }
  if (se_synthesis_ != nullptr) {
    delete se_synthesis_;
    se_synthesis_ = nullptr;
  }
#endif
  // Clean up GB workspace if allocated
  if (gb_workspace_ != nullptr) {
    delete gb_workspace_;
    gb_workspace_ = nullptr;
  }
  // Clean up Ewald infrastructure (PMI grid, CellGrid, syntheses) in reverse order
  if (pme_grid_ != nullptr) {
    delete pme_grid_;
    pme_grid_ = nullptr;
  }
  if (cell_grid_ != nullptr) {
    delete cell_grid_;
    cell_grid_ = nullptr;
  }
  if (ps_synthesis_ != nullptr) {
    delete ps_synthesis_;
    ps_synthesis_ = nullptr;
  }
  if (topology_synthesis_ != nullptr) {
    delete topology_synthesis_;
    topology_synthesis_ = nullptr;
  }
}

//-------------------------------------------------------------------------------------------------
void GCMCSampler::invalidateEnergyCache() {
  energy_cached_ = false;
}

//-------------------------------------------------------------------------------------------------
void GCMCSampler::propagateSystem(int n_steps) {
  if (thermostat_ == nullptr) {
    rtErr("Cannot propagate system: thermostat is nullptr", "GCMCSampler::propagateSystem");
  }

#ifdef STORMM_USE_CUDA
  // GPU path: Use new launchLambdaDynamicsStep() kernel with synthesis objects
  if (launcher_ != nullptr && topology_synthesis_ != nullptr && ps_synthesis_ != nullptr) {
    // Update GPU-resident lambda arrays from current molecule states
    const int n_atoms = topology_->getAtomCount();
    double* lambda_vdw_ptr = lambda_vdw_.data();
    double* lambda_ele_ptr = lambda_ele_.data();

    // Initialize all atoms to fully coupled (lambda=1.0)
    for (int i = 0; i < n_atoms; i++) {
      lambda_vdw_ptr[i] = 1.0;
      lambda_ele_ptr[i] = 1.0;
    }

    // OPTIMIZATION: If lambda arrays are dirty (modified on GPU), reuse existing device copies.
    int n_coupled = 0;
    if (!gpu_lambda_arrays_dirty_) {
      // Override lambda for GCMC-controlled molecules
      for (const auto& mol : molecules_) {
        for (int idx : mol.atom_indices) {
          lambda_vdw_ptr[idx] = mol.lambda_vdw;
          lambda_ele_ptr[idx] = mol.lambda_ele;
        }
      }

      // Build list of coupled atoms (lambda > threshold)
      constexpr double LAMBDA_THRESHOLD = 0.01;
      int* coupled_ptr = coupled_indices_.data();
      for (int i = 0; i < n_atoms; i++) {
        if (lambda_vdw_ptr[i] > LAMBDA_THRESHOLD || lambda_ele_ptr[i] > LAMBDA_THRESHOLD) {
          coupled_ptr[n_coupled++] = i;
        }
      }

      // Upload lambda arrays to GPU
      lambda_vdw_.upload();
      lambda_ele_.upload();
      coupled_indices_.upload(0, n_coupled);

      coupled_indices_valid_ = true;
      gpu_lambda_arrays_dirty_ = false;
      coupled_atom_count_ = n_coupled;
    }
    else {
      // Device arrays already hold updated lambdas; ensure coupled indices reflect them
      n_coupled = ensureCoupledAtomList();
    }

    // Get device pointers
    const double* d_lambda_vdw = lambda_vdw_.data(HybridTargetLevel::DEVICE);
    const double* d_lambda_ele = lambda_ele_.data(HybridTargetLevel::DEVICE);
    const int* d_coupled_indices = coupled_indices_.data(HybridTargetLevel::DEVICE);

    // Note: topology and phase space uploads were removed to avoid memory issues
    // Topology is static and should be uploaded once at initialization
    // Phase space is uploaded inside launchLambdaDynamicsStep() as needed

    // Create CacheResource as STATIC local variables to avoid repeated allocation/deallocation
    // Static locals are allocated once on first call and reused on subsequent calls
    // This prevents CUDA pinned memory fragmentation from repeated alloc/free cycles
    const int2 vale_lp = launcher_->getValenceKernelDims(
        constants::PrecisionModel::DOUBLE,
        EvaluateForce::YES,
        energy::EvaluateEnergy::NO,
        energy::AccumulationMethod::SPLIT,
        synthesis::VwuGoal::MOVE_PARTICLES,
        energy::ClashResponse::NONE);
    const int2 nonb_lp = launcher_->getNonbondedKernelDims(
        constants::PrecisionModel::DOUBLE,
        topology_synthesis_->getNonbondedWorkType(),
        EvaluateForce::YES,
        energy::EvaluateEnergy::NO,
        energy::AccumulationMethod::SPLIT,
        topology_synthesis_->getImplicitSolventModel(),
        energy::ClashResponse::NONE);
    static energy::CacheResource valence_cache(vale_lp.x, maximum_valence_work_unit_atoms);
    static energy::CacheResource nonb_cache(nonb_lp.x, synthesis::small_block_max_atoms);

    // Run MD steps using new lambda dynamics kernel
    for (int step = 0; step < n_steps; step++) {
      // Launch lambda dynamics step (no energy evaluation during propagation for speed)
      launchLambdaDynamicsStep(
          d_lambda_vdw,
          d_lambda_ele,
          d_coupled_indices,
          n_coupled,
          *topology_synthesis_,
          *se_synthesis_,
          thermostat_,
          ps_synthesis_,
          mmctrl_,
          &scorecard_,
          *launcher_,
          &valence_cache,
          &nonb_cache,
          EvaluateEnergy::NO,  // Don't evaluate energy during propagation (faster)
          gb_workspace_,
          gb_model_);

      // Increment step counter
      mmctrl_->incrementStep();
    }

    // Download coordinates back to CPU for energy evaluation
    phase_space_->download();

    // Invalidate energy cache since coordinates changed
    invalidateEnergyCache();

    return;
  }
#endif

  // CPU fallback path: Use old lambdaDynaStep() function
  // This is kept for systems without CUDA or non-periodic systems

  // Get abstracts once before the loop (major optimization)
  PhaseSpaceWriter psw = phase_space_->data();
  ThermostatWriter<double> tstw = thermostat_->dpData();
  const ValenceKit<double> vk = topology_->getDoublePrecisionValenceKit();
  const NonbondedKit<double> nbk = topology_->getDoublePrecisionNonbondedKit();
  const ImplicitSolventKit<double> isk = topology_->getDoublePrecisionImplicitSolventKit();
  const VirtualSiteKit<double> vsk = topology_->getDoublePrecisionVirtualSiteKit();
  const ChemicalDetailsKit cdk = topology_->getChemicalDetailsKit();
  const ConstraintKit<double> cnk = topology_->getDoublePrecisionConstraintKit();
  const StaticExclusionMaskReader ser = exclusions_->data();
  const restraints::RestraintApparatus empty_restraints(topology_);
  const RestraintKit<double, double2, double4> rar = empty_restraints.dpData();

  // Get lambda arrays (CPU-side pointers)
  const double* lambda_vdw_ptr = lambda_vdw_.data();
  const double* lambda_ele_ptr = lambda_ele_.data();

  // Create lambda-aware nonbonded kit
  const topology::LambdaNonbondedKit<double> lambda_nbk(
      nbk.natom, nbk.n_q_types, nbk.n_lj_types, nbk.coulomb_constant, nbk.charge,
      nbk.q_idx, nbk.lj_idx, nbk.q_parameter, nbk.lja_coeff, nbk.ljb_coeff, nbk.ljc_coeff,
      nbk.lja_14_coeff, nbk.ljb_14_coeff, nbk.ljc_14_coeff, nbk.lj_sigma, nbk.lj_14_sigma,
      nbk.nb11x, nbk.nb11_bounds, nbk.nb12x, nbk.nb12_bounds, nbk.nb13x, nbk.nb13_bounds,
      nbk.nb14x, nbk.nb14_bounds, nbk.lj_type_corr, lambda_vdw_ptr, lambda_ele_ptr);

  // Allocate work arrays for GB calculations (even though we're not using GB)
  std::vector<double> effective_gb_radii(topology_->getAtomCount(), 0.0);
  std::vector<double> psi(topology_->getAtomCount(), 0.0);
  std::vector<double> sumdeijda(topology_->getAtomCount(), 0.0);
  generalized_born_defaults::NeckGeneralizedBornKit<double> neck_gbk(0, 0.0, 0.0, nullptr, nullptr);

  // Create minimal DynamicsControls
  DynamicsControls dyn_controls;
  dyn_controls.setStepCount(n_steps);
  dyn_controls.setTimeStep(tstw.dt);

  // Run MD steps with lambda-aware forces
  for (int step = 0; step < n_steps; step++) {
    lambdaDynaStep(&psw, &scorecard_, tstw, vk, lambda_nbk, isk, neck_gbk,
                   effective_gb_radii.data(), psi.data(), sumdeijda.data(),
                   rar, vsk, cdk, cnk, ser, dyn_controls, 0,
                   0.75, 0.5);  // vdw_coupling_threshold=0.75, softcore_alpha=0.5
    tstw.step += 1;
    thermostat_->incrementStep();
  }

  // Invalidate energy cache since coordinates changed
  invalidateEnergyCache();
}

//-------------------------------------------------------------------------------------------------
void GCMCSampler::registerMCMover(std::unique_ptr<MCMover> mover) {
  if (mover) {
    mc_movers_.push_back(std::move(mover));
  }
}

//-------------------------------------------------------------------------------------------------
void GCMCSampler::enableTranslationMoves(double max_displacement) {
  auto mover = std::make_unique<TranslationMover>(this, beta_, &rng_, max_displacement);
  registerMCMover(std::move(mover));
}

//-------------------------------------------------------------------------------------------------
void GCMCSampler::enableRotationMoves(double max_angle) {
  // max_angle is in degrees, convert to radians for mover constructor
  double max_angle_rad = max_angle * pi / 180.0;
  auto mover = std::make_unique<RotationMover>(this, beta_, &rng_, max_angle_rad);
  registerMCMover(std::move(mover));
}

//-------------------------------------------------------------------------------------------------
void GCMCSampler::enableTorsionMoves(double max_angle) {
  // max_angle is in degrees, convert to radians for mover constructor
  double max_angle_rad = max_angle * pi / 180.0;
  auto mover = std::make_unique<TorsionMover>(this, beta_, &rng_, topology_, max_angle_rad);
  registerMCMover(std::move(mover));
}

//-------------------------------------------------------------------------------------------------
bool GCMCSampler::attemptMCMove(GCMCMolecule& mol) {
  if (mc_movers_.empty()) {
    return false;  // No movers registered
  }

  // Randomly select a mover
  const int mover_idx = static_cast<int>(rng_.uniformRandomNumber() * mc_movers_.size());

  // Clamp to valid range (in case of numerical issues)
  const int safe_idx = std::min(mover_idx, static_cast<int>(mc_movers_.size()) - 1);

  // Attempt the move
  return mc_movers_[safe_idx]->attemptMove(mol);
}

//-------------------------------------------------------------------------------------------------
int GCMCSampler::attemptMCMovesOnAllMolecules() {
  // Collect active molecules
  std::vector<GCMCMolecule*> active_molecules;
  for (auto& mol : molecules_) {
    if (mol.isActive()) {
      active_molecules.push_back(&mol);
    }
  }

  // Return 0 if no active molecules
  if (active_molecules.empty()) {
    return 0;
  }

  // Select ONE random active molecule
  const size_t mol_idx = static_cast<size_t>(rng_.uniformRandomNumber() * active_molecules.size());
  const size_t safe_idx = std::min(mol_idx, active_molecules.size() - 1);

  // Attempt ONE MC move on the selected molecule
  const bool accepted = attemptMCMove(*active_molecules[safe_idx]);

  return accepted ? 1 : 0;
}

//-------------------------------------------------------------------------------------------------
std::vector<std::pair<std::string, MCMoveStatistics>> GCMCSampler::getMCStatistics() const {
  std::vector<std::pair<std::string, MCMoveStatistics>> stats;

  for (const auto& mover : mc_movers_) {
    stats.emplace_back(mover->getMoveType(), mover->getStatistics());
  }

  return stats;
}

//-------------------------------------------------------------------------------------------------
// Helper wrapper: Auto-select GPU or CPU path for lambda modification
// This private method is not declared in header - it's just a convenience wrapper
static inline void adjustMoleculeLambdaAuto(GCMCSampler* sampler, GCMCMolecule& mol, double new_lambda) {
#ifdef STORMM_USE_HPC
  // Use GPU path if available for maximum performance
  sampler->adjustMoleculeLambdaGPU(mol, new_lambda);
#else
  // Fall back to CPU path if no HPC support
  sampler->adjustMoleculeLambda(mol, new_lambda);
#endif
}

//-------------------------------------------------------------------------------------------------
void GCMCSampler::adjustMoleculeLambda(GCMCMolecule& mol, double new_lambda) {
#ifdef STORMM_USE_HPC
  if (launcher_ != nullptr) {
    adjustMoleculeLambdaGPU(mol, new_lambda);
    return;
  }
#endif

  // Two-stage coupling for improved acceptance rates
  //
  // Stage 1 (λ ∈ [0, VDW_COUPLING_THRESHOLD]):
  //   - VDW interactions gradually turned on: λ_vdw = λ / 0.75
  //   - Electrostatics remain off: λ_ele = 0
  //   - Softcore potential prevents singularities during insertion
  //
  // Stage 2 (λ ∈ (VDW_COUPLING_THRESHOLD, 1]):
  //   - VDW fully on: λ_vdw = 1.0
  //   - Electrostatics turned on: λ_ele = (λ - 0.75) / 0.25
  //
  // Rationale: VDW first creates space, then electrostatics are added
  // This ordering improves acceptance for polar molecules (e.g., water)
  const double vdw_end = VDW_COUPLING_THRESHOLD;
  double lambda_vdw, lambda_ele;

  if (new_lambda <= vdw_end) {
    lambda_vdw = new_lambda / vdw_end;
    lambda_ele = 0.0;
  } else {
    lambda_vdw = 1.0;
    lambda_ele = (new_lambda - vdw_end) / (1.0 - vdw_end);
  }

  // Store lambda values in molecule - DO NOT modify topology!
  mol.lambda_vdw = lambda_vdw;
  mol.lambda_ele = lambda_ele;

  // Update host-side lambda arrays for the affected atoms
  double* lambda_vdw_ptr = lambda_vdw_.data();
  double* lambda_ele_ptr = lambda_ele_.data();
  for (int idx : mol.atom_indices) {
    lambda_vdw_ptr[idx] = lambda_vdw;
    lambda_ele_ptr[idx] = lambda_ele;
  }

  // Mark caches for rebuild before next GPU operation
  coupled_indices_valid_ = false;
  gpu_lambda_arrays_dirty_ = true;
}

//-------------------------------------------------------------------------------------------------
void GCMCSampler::adjustMoleculeLambdaGPU(GCMCMolecule& mol, double new_lambda) {
#ifdef STORMM_USE_HPC
  // Calculate lambda_vdw and lambda_ele using same two-stage coupling logic as CPU path
  const double vdw_end = VDW_COUPLING_THRESHOLD;
  double lambda_vdw, lambda_ele;

  if (new_lambda <= vdw_end) {
    lambda_vdw = new_lambda / vdw_end;
    lambda_ele = 0.0;
  } else {
    lambda_vdw = 1.0;
    lambda_ele = (new_lambda - vdw_end) / (1.0 - vdw_end);
  }

  // Store lambda values in molecule (CPU-side tracking)
  mol.lambda_vdw = lambda_vdw;
  mol.lambda_ele = lambda_ele;

  // Keep host-side lambda arrays in sync with the molecule state
  double* lambda_vdw_host = lambda_vdw_.data();
  double* lambda_ele_host = lambda_ele_.data();
  for (int idx : mol.atom_indices) {
    lambda_vdw_host[idx] = lambda_vdw;
    lambda_ele_host[idx] = lambda_ele;
  }

  // DEBUG: Print first time GPU path is used
  if (kGcmcDebugLogs) {
    static bool first_gpu_call = true;
    if (first_gpu_call) {
      std::cerr << "DEBUG: GPU lambda modification active! Eliminating uploads.\n";
      first_gpu_call = false;
    }
  }

  // Upload molecule's atom indices to GPU cache if not already done
  // This is a one-time operation per molecule - subsequent calls reuse the cached GPU data
  if (!mol.gpu_indices_cached) {
    // Copy atom indices to CPU side of Hybrid array
    int* gpu_indices_ptr = gpu_molecule_indices_.data();
    for (size_t i = 0; i < mol.atom_indices.size(); i++) {
      gpu_indices_ptr[i] = mol.atom_indices[i];
    }

    // Upload to GPU (one-time cost)
    gpu_molecule_indices_.upload();
    mol.gpu_indices_cached = true;
  }

  // Get device pointers for lambda arrays and molecule indices
  double* d_lambda_vdw = lambda_vdw_.data(HybridTargetLevel::DEVICE);
  double* d_lambda_ele = lambda_ele_.data(HybridTargetLevel::DEVICE);
  const int* d_molecule_indices = gpu_molecule_indices_.data(HybridTargetLevel::DEVICE);

  // Launch kernel to modify lambda values on GPU
  const int n_atoms_in_mol = static_cast<int>(mol.atom_indices.size());
  const int n_atoms_total = topology_->getAtomCount();
  launchUpdateMoleculeLambda(
      n_atoms_in_mol,
      d_molecule_indices,
      lambda_vdw,
      lambda_ele,
      d_lambda_vdw,
      d_lambda_ele,
      n_atoms_total);  // FIX: Pass total atom count for bounds checking

  // NOTE: Kernel includes cudaDeviceSynchronize(), so lambda arrays are safe to use immediately
  // NOTE: We do NOT rebuild coupled_indices here - that's done once per energy evaluation
  //       to avoid redundant work (multiple lambda changes may occur before next energy eval)

  // Mark that GPU lambda arrays have been modified and need coupled indices rebuild
  coupled_indices_valid_ = false;
  gpu_lambda_arrays_dirty_ = true;

#else
  // No HPC support - use CPU path
  adjustMoleculeLambda(mol, new_lambda);
#endif
}

//-------------------------------------------------------------------------------------------------
int GCMCSampler::ensureCoupledAtomList(bool download_to_host) {
  const int n_atoms = topology_->getAtomCount();
  if (n_atoms == 0) {
    coupled_atom_count_ = 0;
    coupled_indices_valid_ = true;
    gpu_lambda_arrays_dirty_ = false;
    return 0;
  }

  if (coupled_indices_valid_ && !gpu_lambda_arrays_dirty_) {
    return coupled_atom_count_;
  }

  constexpr double LAMBDA_THRESHOLD = 0.01;
  int n_coupled = 0;

#ifdef STORMM_USE_CUDA
  if (launcher_ != nullptr) {
    using card::HybridTargetLevel;
    const double* d_lambda_vdw = lambda_vdw_.data(HybridTargetLevel::DEVICE);
    const double* d_lambda_ele = lambda_ele_.data(HybridTargetLevel::DEVICE);
    int* d_coupled_indices = const_cast<int*>(coupled_indices_.data(HybridTargetLevel::DEVICE));

    launchRebuildCoupledIndices(n_atoms,
                                d_lambda_vdw,
                                d_lambda_ele,
                                LAMBDA_THRESHOLD,
                                d_coupled_indices,
                                &n_coupled);

    if (download_to_host) {
      coupled_indices_.download(0, n_coupled);
    }

    coupled_atom_count_ = n_coupled;
    coupled_indices_valid_ = true;
    gpu_lambda_arrays_dirty_ = false;
    return n_coupled;
  }
#endif

  const double* lambda_vdw_ptr = lambda_vdw_.data();
  const double* lambda_ele_ptr = lambda_ele_.data();
  int* coupled_ptr = coupled_indices_.data();
  for (int i = 0; i < n_atoms; i++) {
    if (lambda_vdw_ptr[i] > LAMBDA_THRESHOLD || lambda_ele_ptr[i] > LAMBDA_THRESHOLD) {
      coupled_ptr[n_coupled++] = i;
    }
  }

#ifdef STORMM_USE_CUDA
  if (launcher_ != nullptr) {
    coupled_indices_.upload(0, n_coupled);
  }
#endif

  coupled_atom_count_ = n_coupled;
  coupled_indices_valid_ = true;
  gpu_lambda_arrays_dirty_ = false;
  return n_coupled;
}

//-------------------------------------------------------------------------------------------------
double GCMCSampler::evaluateTotalEnergy() {
  if (kGcmcDebugLogs) {
    log_stream_ << "# DEBUG evaluateTotalEnergy() invoked (energy_cached="
                << (energy_cached_ ? "true" : "false") << ")\n";
    log_stream_.flush();
  }

  // Check energy cache first
  // Compute hash of current lambda state across all molecules
  size_t lambda_hash = 0;
  for (const auto& mol : molecules_) {
    // Combine VDW and electrostatic lambda into hash
    // Use bit patterns to ensure different lambdas produce different hashes
    size_t vdw_bits = *reinterpret_cast<const size_t*>(&mol.lambda_vdw);
    size_t ele_bits = *reinterpret_cast<const size_t*>(&mol.lambda_ele);
    lambda_hash ^= vdw_bits + 0x9e3779b9 + (lambda_hash << 6) + (lambda_hash >> 2);
    lambda_hash ^= ele_bits + 0x9e3779b9 + (lambda_hash << 6) + (lambda_hash >> 2);
  }

  // If cache is valid and lambda state matches, return cached energy
  if (energy_cached_ && lambda_hash == cached_lambda_hash_) {
    static int cache_hit_count = 0;
    if (cache_hit_count < 10) {  // Log first 10 cache hits
      log_stream_ << "# CACHE HIT: Returning cached energy " << cached_energy_
                  << " kcal/mol (hash=" << lambda_hash << ")\n";
      log_stream_.flush();
      cache_hit_count++;
    }
    return cached_energy_;
  }

  // OPTIMIZATION: Download coordinates from GPU once before energy evaluation
  // This replaces the 100+ downloads that were happening after each propagateSystem() call
  #ifdef STORMM_USE_CUDA
  phase_space_->download();
  #endif

  // Build per-atom lambda and LJ parameter arrays for custom evaluation
  //
  // Strategy:
  // 1. Create arrays initialized to 1.0 (fully interacting) for all atoms
  // 2. For ghost molecules, override with their lambda_vdw/lambda_ele values
  // 3. Pass these arrays to evaluateLambdaScaledNonbonded() which applies scaling
  //
  // This approach keeps the topology immutable while allowing per-molecule
  // lambda control. Original parameters are stored in each GCMCMolecule.
  const int n_atoms = topology_->getAtomCount();
  std::vector<double> lambda_vdw_per_atom(n_atoms, 1.0);
  std::vector<double> lambda_ele_per_atom(n_atoms, 1.0);
  std::vector<double> atom_sigma(n_atoms);
  std::vector<double> atom_epsilon(n_atoms);

  // Get nonbonded kit to access per-atom LJ type indices
  const NonbondedKit<double> nbk_init = topology_->getDoublePrecisionNonbondedKit();

  // CORRUPTION DETECTOR: Check if cache has been modified
  static int corruption_check_count = 0;
  if (corruption_check_count < 20) {
    bool corrupted = false;
    for (size_t i = 0; i < cached_lj_sigma_.size(); i++) {
      if (std::abs(cached_lj_sigma_[i] - original_cached_sigma_[i]) > 1.0e-10 ||
          std::abs(cached_lj_epsilon_[i] - original_cached_epsilon_[i]) > 1.0e-10) {
        std::cout << "!!! CORRUPTION DETECTED at eval " << corruption_check_count
                  << " for LJ type " << i << " !!!\n";
        std::cout << "  Original: sigma=" << original_cached_sigma_[i]
                  << ", epsilon=" << original_cached_epsilon_[i] << "\n";
        std::cout << "  Current:  sigma=" << cached_lj_sigma_[i]
                  << ", epsilon=" << cached_lj_epsilon_[i] << "\n";
        std::cout << "  Memory addresses: cached_sigma=" << (void*)&cached_lj_sigma_[i]
                  << " cached_epsilon=" << (void*)&cached_lj_epsilon_[i] << "\n";
        corrupted = true;
      }
    }
    if (corrupted) {
      std::cout << "  Cache was corrupted! This happened between energy evaluations.\n";
      std::cout << "  Set a gdb watchpoint on these addresses to catch the write:\n";
      std::cout << "    watch *" << (void*)cached_lj_epsilon_.data() << "\n";
    }
    corruption_check_count++;
  }

  // DEBUG: Print topology LJ types AND cache values for first few evals
  // Check if both topology and cache are being corrupted
  static int topo_debug_count = 0;
  if (false && topo_debug_count < 10) {  // Disabled DEBUG output
    std::cout << "DEBUG eval " << topo_debug_count << ":\n";
    std::cout << "  CACHE addresses: sigma=" << (void*)cached_lj_sigma_.data()
              << " epsilon=" << (void*)cached_lj_epsilon_.data() << "\n";
    std::cout << "  CACHE values vs Topology values:\n";
    for (int t = 0; t < topology_->getLJTypeCount(); t++) {
      const double cache_sigma = cached_lj_sigma_[t];
      const double cache_eps = cached_lj_epsilon_[t];
      const double topo_sigma = topology_->getLennardJonesSigma<double>(t);
      const double topo_eps = topology_->getLennardJonesEpsilon<double>(t);

      std::cout << "  type" << t << ": "
                << "cache(s=" << cache_sigma << ",e=" << cache_eps << ") "
                << "topo(s=" << topo_sigma << ",e=" << topo_eps << ")\n";
    }
    topo_debug_count++;
  }

  // Initialize with default values (all atoms)
  // Use CACHED LJ parameters instead of reading from topology
  // This prevents topology corruption from affecting energy calculations
  for (int i = 0; i < n_atoms; i++) {
    // Get LJ type index for this atom, then lookup sigma/epsilon from CACHE
    const int lj_type = nbk_init.lj_idx[i];
    atom_sigma[i] = cached_lj_sigma_[lj_type];
    atom_epsilon[i] = cached_lj_epsilon_[lj_type];

    // DEBUG: Print epsilon values from CACHE for first few atoms and evals
    static int eps_debug_count = 0;
    if (false && eps_debug_count < 30 && i < 3) {  // Disabled DEBUG output
      std::cout << "DEBUG atom " << i << ": lj_type=" << lj_type
                << " sigma=" << atom_sigma[i] << " eps=" << atom_epsilon[i]
                << " (from CACHE)\n";
      eps_debug_count++;
    }
  }

  // Override lambda for ALL molecules
  // But only override LJ parameters for GHOST molecules (active use CACHED values)
  for (const auto& mol : molecules_) {
    if (kGcmcDebugLogs) {
      static int lambda_debug_count = 0;
      if (lambda_debug_count < 20 && mol.status == GCMCMoleculeStatus::ACTIVE) {
        log_stream_ << "# DEBUG ENERGY EVAL: ACTIVE molecule resid=" << mol.resid
                    << " lambda_vdw=" << mol.lambda_vdw << " lambda_ele=" << mol.lambda_ele << "\n";
        lambda_debug_count++;
      }
    }

    for (size_t i = 0; i < mol.atom_indices.size(); i++) {
      int idx = mol.atom_indices[i];
      lambda_vdw_per_atom[idx] = mol.lambda_vdw;
      lambda_ele_per_atom[idx] = mol.lambda_ele;

      // Only override sigma/epsilon for ghost molecules
      // Active molecules keep cached values (already initialized above from cache)
      if (mol.status == GCMCMoleculeStatus::GHOST) {
        atom_sigma[idx] = mol.original_sigma[i];
        atom_epsilon[idx] = mol.original_epsilon[i];

        // DEBUG: Print first few overrides
        static int override_count = 0;
        if (false && override_count < 3) {  // Disabled DEBUG output
          const int lj_type = nbk_init.lj_idx[idx];
          std::cout << "DEBUG override atom " << idx << ": orig_eps=" << mol.original_epsilon[i]
                    << " (was " << cached_lj_epsilon_[lj_type] << " from cache)\n";
          override_count++;
        }
      }
    }
  }

  // DEBUG: Print epsilon for candidate molecule atoms after all overrides
  static int eps_eval_count = 0;
  if (false && eps_eval_count < 10) {  // Disabled DEBUG output
    std::cout << "DEBUG atom_epsilon after overrides (eval " << eps_eval_count
              << ", n_atoms=" << n_atoms << "):\n";
    for (int i = 0; i < std::min(24, n_atoms); i++) {
      std::cout << "  atom " << i << ": eps=" << atom_epsilon[i] << "\n";
    }
    eps_eval_count++;
  }

  // CRITICAL: Regenerate abstracts (parameters may have changed)
  // Abstracts are snapshots - must create fresh ones
  const ValenceKit<double> vk = topology_->getDoublePrecisionValenceKit();
  const NonbondedKit<double> nbk = topology_->getDoublePrecisionNonbondedKit();
  PhaseSpaceWriter psw = phase_space_->data();
  const StaticExclusionMaskReader ser = exclusions_->data();

  // Zero forces (not accumulating)
  phase_space_->initializeForces();

  // Evaluate ALL energies (valence + nonbonded) using the GPU kernel
  // Note: Valence terms are NOT lambda-scaled. Intramolecular interactions
  // remain active for both coupled and ghost molecules to maintain correct
  // molecular geometries. Ghost intramolecular energy cancels in acceptance.

  double bond_energy = 0.0;
  double angle_energy = 0.0;
  double2 dihedral_energy = {0.0, 0.0};
  double bonded_energy = 0.0;
  double2 nonbonded;

#ifdef STORMM_USE_CUDA
  // GPU path: Use new lambda dynamics kernel with ScoreCard energy accumulation
  if (launcher_ != nullptr && topology_synthesis_ != nullptr && ps_synthesis_ != nullptr) {
    const int n_atoms = topology_->getAtomCount();
    double* lambda_vdw_ptr = lambda_vdw_.data();
    double* lambda_ele_ptr = lambda_ele_.data();
    for (int i = 0; i < n_atoms; i++) {
      lambda_vdw_ptr[i] = lambda_vdw_per_atom[i];
      lambda_ele_ptr[i] = lambda_ele_per_atom[i];
    }

    // If the device arrays were last updated from the host, refresh them now
    if (!gpu_lambda_arrays_dirty_) {
      lambda_vdw_.upload();
      lambda_ele_.upload();
    }

    // Force coupled index rebuild to reflect updated lambda values
    coupled_indices_valid_ = false;
    gpu_lambda_arrays_dirty_ = true;

    const int n_coupled = ensureCoupledAtomList();
    const double* d_lambda_vdw = lambda_vdw_.data(HybridTargetLevel::DEVICE);
    const double* d_lambda_ele = lambda_ele_.data(HybridTargetLevel::DEVICE);
    const int* d_coupled_indices = coupled_indices_.data(HybridTargetLevel::DEVICE);

    if (kGcmcDebugLogs) {
      log_stream_ << "# DEBUG evaluateTotalEnergy() GPU path: n_coupled=" << n_coupled
                  << " dirty=" << (gpu_lambda_arrays_dirty_ ? "true" : "false") << "\n";
      log_stream_.flush();
    }

    // REMOVED: topology_->upload() and phase_space_->upload()
    // These were causing memory issues by repeatedly uploading on every energy evaluation
    // Topology and phase space are already on GPU from initialization
    // Only phase_space coordinates change and are managed by launchLambdaDynamicsStep

    // Clear scorecard for fresh energy evaluation
    // Zero all accumulators on the device before kernel accumulates energies
#ifdef STORMM_USE_HPC
    scorecard_.initialize(HybridTargetLevel::DEVICE);
#else
    scorecard_.initialize(HybridTargetLevel::HOST);
#endif

    // Evaluate energy using lambda dynamics kernel (no integration, just energy)
    // Use a static MolecularMechanicsControls to avoid memory leak from repeated allocations
    static MolecularMechanicsControls energy_mmctrl(0.0, 0);  // Zero timestep, zero steps
    static bool initialized = false;
    if (!initialized) {
      energy_mmctrl.primeWorkUnitCounters(
          *launcher_,
          EvaluateForce::NO,   // Don't need forces for energy evaluation
          energy::EvaluateEnergy::YES, // DO evaluate energy
          energy::ClashResponse::NONE,
          synthesis::VwuGoal::ACCUMULATE, // Accumulate mode (no integration)
          constants::PrecisionModel::DOUBLE,
          constants::PrecisionModel::DOUBLE,
          *topology_synthesis_);
      initialized = true;
    }

    // Create CacheResource as STATIC local variables to avoid repeated allocation/deallocation
    // Static locals are allocated once on first call and reused on subsequent calls
    // This prevents CUDA pinned memory fragmentation from 100s of alloc/free cycles
    const int2 vale_lp = launcher_->getValenceKernelDims(
        constants::PrecisionModel::DOUBLE,
        EvaluateForce::NO,
        energy::EvaluateEnergy::YES,
        energy::AccumulationMethod::SPLIT,
        synthesis::VwuGoal::ACCUMULATE,
        energy::ClashResponse::NONE);
    const int2 nonb_lp = launcher_->getNonbondedKernelDims(
        constants::PrecisionModel::DOUBLE,
        topology_synthesis_->getNonbondedWorkType(),
        EvaluateForce::NO,
        energy::EvaluateEnergy::YES,
        energy::AccumulationMethod::SPLIT,
        topology_synthesis_->getImplicitSolventModel(),
        energy::ClashResponse::NONE);
    static energy::CacheResource valence_cache(vale_lp.x, maximum_valence_work_unit_atoms);
    static energy::CacheResource nonb_cache(nonb_lp.x, synthesis::small_block_max_atoms);

    if (kGcmcDebugLogs) {
      log_stream_ << "# DEBUG evaluateTotalEnergy(): invoking launchLambdaDynamicsStep with n_coupled="
                  << n_coupled << "\n";
      log_stream_.flush();
    }

    launchLambdaDynamicsStep(
        d_lambda_vdw,
        d_lambda_ele,
        d_coupled_indices,
        n_coupled,
        *topology_synthesis_,
        *se_synthesis_,
        thermostat_,  // Must pass thermostat even for energy-only (kernel accesses it)
        ps_synthesis_,
        &energy_mmctrl,
        &scorecard_,
        *launcher_,
        &valence_cache,
        &nonb_cache,
        energy::EvaluateEnergy::YES,
        gb_workspace_,
        gb_model_);

    // Download energies from ScoreCard
    scorecard_.download();

    // Extract ALL energies from ScoreCard (GPU kernel computes both valence and nonbonded)
    bond_energy = scorecard_.reportInstantaneousStates(StateVariable::BOND, 0);
    angle_energy = scorecard_.reportInstantaneousStates(StateVariable::ANGLE, 0);
    const double proper_dihedral = scorecard_.reportInstantaneousStates(StateVariable::PROPER_DIHEDRAL, 0);
    const double improper_dihedral = scorecard_.reportInstantaneousStates(StateVariable::IMPROPER_DIHEDRAL, 0);
    dihedral_energy.x = proper_dihedral;
    dihedral_energy.y = improper_dihedral;
    bonded_energy = bond_energy + angle_energy + proper_dihedral + improper_dihedral;

    const double elec_energy = scorecard_.reportInstantaneousStates(StateVariable::ELECTROSTATIC, 0);
    const double vdw_energy = scorecard_.reportInstantaneousStates(StateVariable::VDW, 0);

    nonbonded.x = elec_energy;
    nonbonded.y = vdw_energy;
  } else {
    // Fallback: No GPU infrastructure available, use zeros
    nonbonded.x = 0.0;
    nonbonded.y = 0.0;
  }
#else
  // CPU path: use original CPU function
  nonbonded = evaluateLambdaScaledNonbonded(nbk, ser, psw,
                                            lambda_vdw_per_atom,
                                            lambda_ele_per_atom,
                                            atom_sigma,
                                            atom_epsilon,
                                            &scorecard_,
                                            EvaluateForce::NO, 0);
#endif

  // NOTE: PME reciprocal space energy is NOT included here because STORMM lacks FFT implementation.
  // When STORMM FFT is implemented, the full PME calculation would be:
  //   1. Spread lambda-scaled charges: mapDensityLambda(pme_grid_, ...)
  //   2. FFT convolution (NOT IMPLEMENTED - getTotalOnGrid() only sums for charge checks)
  //   3. Self-energy correction: pmeSelfEnergyLambda(...)
  // Currently using Ewald DIRECT SPACE only via ewald_coeff_ in nonbonded kernel.

  // Total energy: bonded + direct space nonbonded (Ewald direct space for periodic systems)
  // NOTE: Reciprocal space energy is missing until STORMM implements FFT
  double total = bonded_energy + nonbonded.x + nonbonded.y;

  // Update energy cache for next evaluation
  energy_cached_ = true;
  cached_energy_ = total;
  cached_lambda_hash_ = lambda_hash;

  if (kGcmcDebugLogs) {
    log_stream_ << "# DEBUG evaluateTotalEnergy() complete: total=" << total
                << " bonded=" << bonded_energy
                << " elec=" << nonbonded.x
                << " vdw=" << nonbonded.y << "\n";
    log_stream_.flush();
  }

  return total;
}

//-------------------------------------------------------------------------------------------------
GCMCMolecule* GCMCSampler::selectRandomGhostMolecule() {
  std::vector<int> ghost_indices;
  for (size_t i = 0; i < molecules_.size(); i++) {
    if (molecules_[i].status == GCMCMoleculeStatus::GHOST) {
      ghost_indices.push_back(i);
    }
  }

  if (ghost_indices.empty()) {
    return nullptr;
  }

  // Select random ghost with safe index calculation
  // Use std::min to ensure index is always in valid range [0, size-1]
  // This handles the edge case where uniformRandomNumber() returns exactly 1.0
  size_t random_idx = std::min(
      static_cast<size_t>(std::floor(rng_.uniformRandomNumber() * static_cast<double>(ghost_indices.size()))),
      ghost_indices.size() - 1
  );

  return &molecules_[ghost_indices[random_idx]];
}

//-------------------------------------------------------------------------------------------------
GCMCMolecule* GCMCSampler::selectRandomActiveMolecule() {
  std::vector<int> active_indices;
  for (size_t i = 0; i < molecules_.size(); i++) {
    if (molecules_[i].status == GCMCMoleculeStatus::ACTIVE) {
      active_indices.push_back(i);
    }
  }

  if (active_indices.empty()) {
    return nullptr;
  }

  // Select random active molecule with safe index calculation
  // Use std::min to ensure index is always in valid range [0, size-1]
  // This handles the edge case where uniformRandomNumber() returns exactly 1.0
  size_t random_idx = std::min(
      static_cast<size_t>(std::floor(rng_.uniformRandomNumber() * static_cast<double>(active_indices.size()))),
      active_indices.size() - 1
  );

  return &molecules_[active_indices[random_idx]];
}

//-------------------------------------------------------------------------------------------------
double3 GCMCSampler::calculateMoleculeCOG(const GCMCMolecule& mol) const {
  // Download coordinates from GPU if needed
  #ifdef STORMM_USE_CUDA
  phase_space_->download();
  #endif

  // Get host pointers using correct PhaseSpace API
  PhaseSpaceWriter psw = phase_space_->data();
  const double* xcrd = psw.xcrd;
  const double* ycrd = psw.ycrd;
  const double* zcrd = psw.zcrd;

  double3 cog = {0.0, 0.0, 0.0};

  // Use heavy atoms if available, otherwise all atoms
  const std::vector<int>& atoms_to_use =
    mol.heavy_atom_indices.empty() ? mol.atom_indices : mol.heavy_atom_indices;

  // Check for division by zero
  if (atoms_to_use.empty()) {
    rtErr("Cannot calculate COG for molecule with no atoms", "calculateMoleculeCOG");
  }

  for (int atom_idx : atoms_to_use) {
    cog.x += xcrd[atom_idx];
    cog.y += ycrd[atom_idx];
    cog.z += zcrd[atom_idx];
  }

  const double n_atoms = static_cast<double>(atoms_to_use.size());
  cog.x /= n_atoms;
  cog.y /= n_atoms;
  cog.z /= n_atoms;

  return cog;
}

//-------------------------------------------------------------------------------------------------
std::vector<double3> GCMCSampler::saveCoordinates(const GCMCMolecule& mol) const {
  std::vector<double3> saved;

  // Download from GPU if needed
  #ifdef STORMM_USE_CUDA
  phase_space_->download();
  #endif

  // Get coordinates using correct PhaseSpace API
  PhaseSpaceWriter psw = phase_space_->data();
  const double* xcrd = psw.xcrd;
  const double* ycrd = psw.ycrd;
  const double* zcrd = psw.zcrd;

  for (int atom_idx : mol.atom_indices) {
    double3 coord;
    coord.x = xcrd[atom_idx];
    coord.y = ycrd[atom_idx];
    coord.z = zcrd[atom_idx];
    saved.push_back(coord);
  }

  return saved;
}

//-------------------------------------------------------------------------------------------------
void GCMCSampler::applyRandomRotation(const GCMCMolecule& mol) {
  // Generate uniform random rotation using quaternions (Shoemake method)
  // Reference: K. Shoemake, "Uniform random rotations", Graphics Gems III, 1992

  const double u1 = rng_.uniformRandomNumber();
  const double u2 = rng_.uniformRandomNumber();
  const double u3 = rng_.uniformRandomNumber();

  const double sqrt1_u1 = std::sqrt(1.0 - u1);
  const double sqrtu1 = std::sqrt(u1);
  const double two_pi_u2 = 2.0 * M_PI * u2;
  const double two_pi_u3 = 2.0 * M_PI * u3;

  // Generate uniform random unit quaternion
  const double qw = sqrt1_u1 * std::sin(two_pi_u2);
  const double qx = sqrt1_u1 * std::cos(two_pi_u2);
  const double qy = sqrtu1 * std::sin(two_pi_u3);
  const double qz = sqrtu1 * std::cos(two_pi_u3);

  // Convert quaternion to rotation matrix
  const double xx = qx * qx;
  const double yy = qy * qy;
  const double zz = qz * qz;
  const double xy = qx * qy;
  const double xz = qx * qz;
  const double yz = qy * qz;
  const double wx = qw * qx;
  const double wy = qw * qy;
  const double wz = qw * qz;

  const double r00 = 1.0 - 2.0 * (yy + zz);
  const double r01 = 2.0 * (xy - wz);
  const double r02 = 2.0 * (xz + wy);
  const double r10 = 2.0 * (xy + wz);
  const double r11 = 1.0 - 2.0 * (xx + zz);
  const double r12 = 2.0 * (yz - wx);
  const double r20 = 2.0 * (xz - wy);
  const double r21 = 2.0 * (yz + wx);
  const double r22 = 1.0 - 2.0 * (xx + yy);

  // Get center of geometry
  const double3 cog = calculateMoleculeCOG(mol);

  // Get coordinate pointers
  PhaseSpaceWriter psw = phase_space_->data();
  double* xcrd = psw.xcrd;
  double* ycrd = psw.ycrd;
  double* zcrd = psw.zcrd;

  // Rotate each atom about the COG
  for (int atom_idx : mol.atom_indices) {
    // Translate to COG origin
    const double dx = xcrd[atom_idx] - cog.x;
    const double dy = ycrd[atom_idx] - cog.y;
    const double dz = zcrd[atom_idx] - cog.z;

    // Apply rotation
    const double rx = r00 * dx + r01 * dy + r02 * dz;
    const double ry = r10 * dx + r11 * dy + r12 * dz;
    const double rz = r20 * dx + r21 * dy + r22 * dz;

    // Translate back
    xcrd[atom_idx] = rx + cog.x;
    ycrd[atom_idx] = ry + cog.y;
    zcrd[atom_idx] = rz + cog.z;
  }

  // Note: Coordinates/velocities will be uploaded by evaluateTotalEnergy() when needed
  // Manual uploadPositions() here was causing CUDA driver state exhaustion

  // Invalidate energy cache since coordinates changed
  invalidateEnergyCache();
}

//-------------------------------------------------------------------------------------------------
std::vector<double3> GCMCSampler::saveVelocities(const GCMCMolecule& mol) const {
  std::vector<double3> saved;

  // Download from GPU if needed
  #ifdef STORMM_USE_CUDA
  phase_space_->download();
  #endif

  // Get velocities using correct PhaseSpace API
  PhaseSpaceWriter psw = phase_space_->data();
  const double* xvel = psw.xvel;
  const double* yvel = psw.yvel;
  const double* zvel = psw.zvel;

  for (int atom_idx : mol.atom_indices) {
    double3 vel;
    vel.x = xvel[atom_idx];
    vel.y = yvel[atom_idx];
    vel.z = zvel[atom_idx];
    saved.push_back(vel);
  }

  return saved;
}

//-------------------------------------------------------------------------------------------------
void GCMCSampler::restoreCoordinates(GCMCMolecule& mol, const std::vector<double3>& saved_coords) {
  // Get coordinate arrays using correct PhaseSpace API
  PhaseSpaceWriter psw = phase_space_->data();
  double* xcrd = psw.xcrd;
  double* ycrd = psw.ycrd;
  double* zcrd = psw.zcrd;

  for (size_t i = 0; i < mol.atom_indices.size(); i++) {
    int atom_idx = mol.atom_indices[i];
    xcrd[atom_idx] = saved_coords[i].x;
    ycrd[atom_idx] = saved_coords[i].y;
    zcrd[atom_idx] = saved_coords[i].z;
  }

  // PhaseSpace manages GPU synchronization internally
  // No manual upload/download needed

  // Invalidate energy cache since coordinates changed
  invalidateEnergyCache();
}

//-------------------------------------------------------------------------------------------------
void GCMCSampler::restoreVelocities(GCMCMolecule& mol, const std::vector<double3>& saved_vels,
                                     bool reverse) {
  // Get velocity arrays using correct PhaseSpace API
  PhaseSpaceWriter psw = phase_space_->data();
  double* xvel = psw.xvel;
  double* yvel = psw.yvel;
  double* zvel = psw.zvel;

  const double sign = reverse ? -1.0 : 1.0;

  for (size_t i = 0; i < mol.atom_indices.size(); i++) {
    int atom_idx = mol.atom_indices[i];
    xvel[atom_idx] = sign * saved_vels[i].x;
    yvel[atom_idx] = sign * saved_vels[i].y;
    zvel[atom_idx] = sign * saved_vels[i].z;
  }

  // PhaseSpace manages GPU synchronization internally
  // No manual upload/download needed
}

//-------------------------------------------------------------------------------------------------
double3 GCMCSampler::generateMaxwellBoltzmannVelocity(double mass) {
  // Generate Maxwell-Boltzmann velocity
  // For units: kB in amu*A^2/fs^2/K, mass in amu, T in K
  // Result: velocity in A/fs
  const double sigma = std::sqrt(boltzmann_constant_md * temperature_ / mass);

  double3 vel;
  vel.x = rng_.gaussianRandomNumber() * sigma;
  vel.y = rng_.gaussianRandomNumber() * sigma;
  vel.z = rng_.gaussianRandomNumber() * sigma;

  return vel;
}

//-------------------------------------------------------------------------------------------------
void GCMCSampler::applyPBC(const GCMCMolecule& mol) {
  // Get coordinate arrays and box dimensions using correct PhaseSpace API
  PhaseSpaceWriter psw = phase_space_->data();

  // Get box dimensions from PhaseSpace
  const double* box_dims = phase_space_->getBoxDimensionsHandle()->data();
  const double box_x = box_dims[0];
  const double box_y = box_dims[1];
  const double box_z = box_dims[2];

  double* xcrd = psw.xcrd;
  double* ycrd = psw.ycrd;
  double* zcrd = psw.zcrd;

  // Calculate molecule center of geometry (COG) first
  // This ensures the entire molecule is wrapped as a unit, preserving molecular geometry
  double cog_x = 0.0;
  double cog_y = 0.0;
  double cog_z = 0.0;
  int n_atoms = mol.atom_indices.size();

  for (int atom_idx : mol.atom_indices) {
    cog_x += xcrd[atom_idx];
    cog_y += ycrd[atom_idx];
    cog_z += zcrd[atom_idx];
  }

  cog_x /= n_atoms;
  cog_y /= n_atoms;
  cog_z /= n_atoms;

  // Calculate shift to bring COG into primary box [0, L)
  // Using floor to wrap coordinates into the range [0, box_size)
  double shift_x = -std::floor(cog_x / box_x) * box_x;
  double shift_y = -std::floor(cog_y / box_y) * box_y;
  double shift_z = -std::floor(cog_z / box_z) * box_z;

  // Apply the same shift to ALL atoms in the molecule
  // This preserves molecular geometry while wrapping the entire molecule
  for (int atom_idx : mol.atom_indices) {
    xcrd[atom_idx] += shift_x;
    ycrd[atom_idx] += shift_y;
    zcrd[atom_idx] += shift_z;
  }

  // Upload modified coordinates back to GPU if needed
  #ifdef STORMM_USE_HPC
  #ifdef STORMM_USE_HPC

  phase_space_->uploadPositions();

  #endif
  #endif
}

//-------------------------------------------------------------------------------------------------
void GCMCSampler::applyPBCToAllMolecules() {
  // Wrap ALL molecules back into the primary unit cell
  // This is needed after MD propagation to ensure consistent PBC treatment
  for (auto& mol : molecules_) {
    // Only wrap coupled molecules (lambda > 0)
    if (mol.lambda_vdw > 0.01 || mol.lambda_ele > 0.01) {
      applyPBC(mol);
    }
  }
}

//-------------------------------------------------------------------------------------------------
int GCMCSampler::getActiveCount() const {
  return N_active_;
}

//-------------------------------------------------------------------------------------------------
int GCMCSampler::getGhostCount() const {
  int count = 0;
  for (const auto& mol : molecules_) {
    if (mol.status == GCMCMoleculeStatus::GHOST) {
      count++;
    }
  }
  return count;
}

//-------------------------------------------------------------------------------------------------
std::vector<int> GCMCSampler::getActiveAtomIndices() const {
  std::vector<int> active_atoms;
  for (const auto& mol : molecules_) {
    if (mol.status == GCMCMoleculeStatus::ACTIVE) {
      // Add all atom indices from this active molecule
      active_atoms.insert(active_atoms.end(), mol.atom_indices.begin(), mol.atom_indices.end());
    }
  }
  return active_atoms;
}

//-------------------------------------------------------------------------------------------------
int GCMCSampler::getTotalMoleculeCount() const {
  return molecules_.size();
}

//-------------------------------------------------------------------------------------------------
const GCMCStatistics& GCMCSampler::getStatistics() const {
  return stats_;
}

//-------------------------------------------------------------------------------------------------
void GCMCSampler::writeGhostSnapshot() {
  if (!ghost_stream_.is_open()) return;

  ghost_stream_ << stats_.n_moves;
  for (const auto& mol : molecules_) {
    if (mol.status == GCMCMoleculeStatus::GHOST) {
      ghost_stream_ << " " << mol.resid;
    }
  }
  ghost_stream_ << "\n";
  ghost_stream_.flush();
}

//-------------------------------------------------------------------------------------------------
void GCMCSampler::logMove(const std::string& move_type, int resid, double work, double accept_prob) {
  if (!log_stream_.is_open()) return;

  log_stream_ << std::setw(6) << stats_.n_moves << " "
              << std::setw(20) << move_type << " "
              << std::setw(6) << resid << " "
              << std::setw(12) << std::setprecision(4) << std::fixed << work << " "
              << std::setw(10) << std::setprecision(6) << accept_prob << " ";

  if (!stats_.outcomes.empty()) {
    log_stream_ << stats_.outcomes.back();
  }

  log_stream_ << "\n";
  log_stream_.flush();
}

//-------------------------------------------------------------------------------------------------
// GCMCSphereSampler implementation
//-------------------------------------------------------------------------------------------------
GCMCSphereSampler::GCMCSphereSampler(AtomGraph* topology,
                                     PhaseSpace* ps,
                                     StaticExclusionMask* exclusions,
                                     Thermostat* thermostat,
                                     double temperature,
                                     const GhostMoleculeMetadata& ghost_metadata,
                                     const std::vector<int>& ref_atoms,
                                     double sphere_radius,
                                     double mu_ex,
                                     double standard_volume,
                                     double adams,
                                     double adams_shift,
                                     int max_N,
                                     topology::ImplicitSolventModel gb_model,
                                     const std::string& resname,
                                     const std::string& ghost_file,
                                     const std::string& log_file) :
  GCMCSampler(topology, ps, exclusions, thermostat, temperature, ghost_metadata, gb_model, resname, ghost_file, log_file),
  sphere_(ref_atoms, sphere_radius),
  mu_ex_{mu_ex},
  standard_volume_{standard_volume},
  max_N_{max_N}
{
  // Validate sphere radius for numerical stability
  if (sphere_radius < 1.0) {
    rtErr("Sphere radius must be at least 1.0 Angstroms", "GCMCSphereSampler");
  }

  // Calculate Adams B parameter
  if (!std::isnan(adams)) {
    B_ = adams + adams_shift;
  } else {
    const double sphere_volume = (4.0 / 3.0) * pi * sphere_radius * sphere_radius * sphere_radius;
    B_ = beta_ * mu_ex_ + std::log(sphere_volume / standard_volume_) + adams_shift;
  }

  // Update sphere center
  updateSphereCenter();

  // Classify molecules by sphere membership
  classifyMolecules();

  // Count initial N
  N_active_ = 0;
  for (const auto& mol : molecules_) {
    if (mol.status == GCMCMoleculeStatus::ACTIVE) {
      N_active_++;
    }
  }

  log_stream_ << "# Sphere radius: " << sphere_radius << " A\n";
  log_stream_ << "# Sphere volume: " << sphere_.getVolume() << " A^3\n";
  log_stream_ << "# Adams B parameter: " << B_ << "\n";
  log_stream_ << "# Initial molecules in sphere: " << N_active_ << "\n";
  log_stream_.flush();
}

//-------------------------------------------------------------------------------------------------
void GCMCSphereSampler::updateSphereCenter() {
  // Download coordinates from GPU if needed
  #ifdef STORMM_USE_CUDA
  phase_space_->download();
  #endif

  // Get host pointers using correct PhaseSpace API
  PhaseSpaceWriter psw = phase_space_->data();
  const double* xcrd = psw.xcrd;
  const double* ycrd = psw.ycrd;
  const double* zcrd = psw.zcrd;

  // Update sphere center based on reference atoms
  sphere_.updateCenter(xcrd, ycrd, zcrd);
}

//-------------------------------------------------------------------------------------------------
void GCMCSphereSampler::classifyMolecules() {
  // Update sphere center first
  updateSphereCenter();

  // Check each molecule's position relative to sphere
  for (auto& mol : molecules_) {
    double3 cog = calculateMoleculeCOG(mol);

    if (sphere_.contains(cog)) {
      if (mol.status == GCMCMoleculeStatus::UNTRACKED) {
        // Molecule entered sphere region
        mol.status = GCMCMoleculeStatus::ACTIVE;
        N_active_++;
      }
    } else {
      if (mol.status == GCMCMoleculeStatus::ACTIVE) {
        // Molecule left sphere region
        mol.status = GCMCMoleculeStatus::UNTRACKED;
        N_active_--;
      }
    }
  }
}

//-------------------------------------------------------------------------------------------------
bool GCMCSphereSampler::isMoleculeInSphere(const GCMCMolecule& mol) const {
  double3 cog = calculateMoleculeCOG(mol);
  return sphere_.contains(cog);
}

//-------------------------------------------------------------------------------------------------
double3 GCMCSphereSampler::selectInsertionSite() {
  // Generate random point within sphere using rejection sampling
  double3 site;
  const double3 center = sphere_.getCenter();
  const double radius = sphere_.getRadius();

  bool found = false;
  int attempts = 0;
  const int max_attempts = 1000;

  while (!found && attempts < max_attempts) {
    // Generate random point in cube
    site.x = center.x + (2.0 * rng_.uniformRandomNumber() - 1.0) * radius;
    site.y = center.y + (2.0 * rng_.uniformRandomNumber() - 1.0) * radius;
    site.z = center.z + (2.0 * rng_.uniformRandomNumber() - 1.0) * radius;

    // Check if within sphere
    if (sphere_.contains(site)) {
      found = true;
    }
    attempts++;
  }

  if (!found) {
    rtErr("Failed to find insertion site within sphere after " + std::to_string(max_attempts) +
          " attempts", "GCMCSphereSampler::selectInsertionSite");
  }

  return site;
}

//-------------------------------------------------------------------------------------------------
bool GCMCSphereSampler::attemptInsertion() {
  stats_.n_moves++;

  // Check if sphere is full
  if (N_active_ >= max_N_) {
    log_stream_ << "# Sphere full, rejecting insertion\n";
    stats_.outcomes.push_back("rejected insert (sphere full)");
    return false;
  }

  // Select random ghost molecule
  GCMCMolecule* ghost_mol = selectRandomGhostMolecule();
  if (ghost_mol == nullptr) {
    log_stream_ << "# No ghost molecules available\n";
    stats_.outcomes.push_back("rejected insert (no ghosts)");
    return false;
  }

  // Select insertion site
  double3 insertion_site = selectInsertionSite();

  // Save current state
  std::vector<double3> saved_coords = saveCoordinates(*ghost_mol);
  std::vector<double3> saved_vels = saveVelocities(*ghost_mol);

  // Calculate current COG
  double3 current_cog = calculateMoleculeCOG(*ghost_mol);

  // Move molecule to insertion site - use correct PhaseSpace API
  PhaseSpaceWriter psw = phase_space_->data();
  double* xcrd = psw.xcrd;
  double* ycrd = psw.ycrd;
  double* zcrd = psw.zcrd;

  for (size_t i = 0; i < ghost_mol->atom_indices.size(); i++) {
    int atom_idx = ghost_mol->atom_indices[i];
    xcrd[atom_idx] = insertion_site.x + (saved_coords[i].x - current_cog.x);
    ycrd[atom_idx] = insertion_site.y + (saved_coords[i].y - current_cog.y);
    zcrd[atom_idx] = insertion_site.z + (saved_coords[i].z - current_cog.z);
  }

  // Apply PBC
  applyPBC(*ghost_mol);

  // Assign Maxwell-Boltzmann velocities
  double* xvel = psw.xvel;
  double* yvel = psw.yvel;
  double* zvel = psw.zvel;

  for (int atom_idx : ghost_mol->atom_indices) {
    double mass = topology_->getAtomicMass<double>(atom_idx);
    double3 vel = generateMaxwellBoltzmannVelocity(mass);
    xvel[atom_idx] = vel.x;
    yvel[atom_idx] = vel.y;
    zvel[atom_idx] = vel.z;
  }

  // Note: Coordinates/velocities will be uploaded by evaluateTotalEnergy() when needed
  // Manual uploadPositions() here was causing CUDA driver state exhaustion

  // Calculate energy change for GCMC insertion
  // Molecules are already at correct lambdas: ACTIVE at 1.0, GHOST at 0.0
  // Only need to evaluate inserting molecule at lambda=0 vs lambda≈1

  // E_initial: Ghost at lambda=0 has no interactions (should be 0)
  // Ghost molecule is already at lambda=0, so just evaluate
  double E_initial = evaluateTotalEnergy();

  // E_final: Set inserting molecule to lambda≈1 to calculate interactions
  // Use 0.998 (< 0.999 threshold) so evaluateLambdaScaledNonbonded sees it as a "ghost"
  // and calculates its interactions with ACTIVE molecules (which are at lambda=1.0)
  adjustMoleculeLambdaAuto(this, *ghost_mol, 0.998);
  double E_final = evaluateTotalEnergy();

  // Restore to lambda=0 before acceptance decision
  adjustMoleculeLambdaAuto(this, *ghost_mol, 0.0);

  double delta_E = E_final - E_initial;

  // DEBUG: Print energies (first 10 insertions only)
  if (kGcmcDebugLogs && stats_.n_inserts < 10) {
    log_stream_ << "# DEBUG Insert " << stats_.n_inserts << ": E_init=" << E_initial
                << " E_final=" << E_final << " dE=" << delta_E << " N_active=" << N_active_ << "\n";
  }

  // If accepted, will be set to lambda=1.0 below

  // Calculate acceptance probability for insertion
  // P_acc = min(1, exp(B) * exp(-beta*delta_E) / (N+1))
  double acc_prob = std::min(1.0, std::exp(B_ - beta_ * delta_E) / (N_active_ + 1.0));

  // Metropolis acceptance
  stats_.n_inserts++;
  stats_.insert_acceptance_probs.push_back(acc_prob);
  stats_.move_resids.push_back(ghost_mol->resid);

  if (rng_.uniformRandomNumber() < acc_prob) {
    // ACCEPT - fully activate molecule
    adjustMoleculeLambdaAuto(this, *ghost_mol, 1.0);
    if (kGcmcDebugLogs) {
      log_stream_ << "# DEBUG ACCEPT: lambda set to 1.0 for resid " << ghost_mol->resid << "\n";
    }
    ghost_mol->status = GCMCMoleculeStatus::ACTIVE;
    N_active_++;
    stats_.n_accepted++;
    stats_.n_accepted_inserts++;
    stats_.N_history.push_back(N_active_);
    stats_.acc_rate_history.push_back(stats_.getAcceptanceRate());

    logMove("ACCEPTED insertion", ghost_mol->resid, delta_E, acc_prob);
    stats_.outcomes.push_back("accepted insert");

    return true;
  } else {
    // REJECT - restore original state
    adjustMoleculeLambdaAuto(this, *ghost_mol, 0.0);
    restoreCoordinates(*ghost_mol, saved_coords);
    restoreVelocities(*ghost_mol, saved_vels, false);  // Don't reverse for standard MC

    stats_.N_history.push_back(N_active_);
    stats_.acc_rate_history.push_back(stats_.getAcceptanceRate());

    logMove("REJECTED insertion", ghost_mol->resid, delta_E, acc_prob);
    stats_.outcomes.push_back("rejected insert");

    return false;
  }
}

//-------------------------------------------------------------------------------------------------
bool GCMCSphereSampler::attemptDeletion() {
  stats_.n_moves++;

  // Select random active molecule
  GCMCMolecule* active_mol = selectRandomActiveMolecule();
  if (active_mol == nullptr) {
    log_stream_ << "# No active molecules available\n";
    stats_.outcomes.push_back("rejected delete (no active)");
    return false;
  }

  // Check if molecule is in sphere
  if (!isMoleculeInSphere(*active_mol)) {
    log_stream_ << "# Molecule not in sphere, cannot delete\n";
    stats_.outcomes.push_back("rejected delete (outside sphere)");
    return false;
  }

  // Calculate energy with molecule present
  double E_initial = evaluateTotalEnergy();

  // Turn off interactions
  adjustMoleculeLambdaAuto(this, *active_mol, 0.0);
  double E_final = evaluateTotalEnergy();

  double delta_E = E_final - E_initial;

  // Calculate acceptance probability for deletion
  // P_acc = min(1, N * exp(-B) * exp(-beta*delta_E))
  double acc_prob = std::min(1.0, N_active_ * std::exp(-B_ - beta_ * delta_E));

  // Metropolis acceptance
  stats_.n_deletes++;
  stats_.delete_acceptance_probs.push_back(acc_prob);
  stats_.move_resids.push_back(active_mol->resid);

  if (rng_.uniformRandomNumber() < acc_prob) {
    // ACCEPT
    active_mol->status = GCMCMoleculeStatus::GHOST;
    N_active_--;
    stats_.n_accepted++;
    stats_.n_accepted_deletes++;
    stats_.N_history.push_back(N_active_);
    stats_.acc_rate_history.push_back(stats_.getAcceptanceRate());

    logMove("ACCEPTED deletion", active_mol->resid, delta_E, acc_prob);
    stats_.outcomes.push_back("accepted delete");

    return true;
  } else {
    // REJECT - restore interactions
    adjustMoleculeLambdaAuto(this, *active_mol, 1.0);

    stats_.N_history.push_back(N_active_);
    stats_.acc_rate_history.push_back(stats_.getAcceptanceRate());

    logMove("REJECTED deletion", active_mol->resid, delta_E, acc_prob);
    stats_.outcomes.push_back("rejected delete");

    return false;
  }
}

//-------------------------------------------------------------------------------------------------
bool GCMCSphereSampler::runGCMCCycle() {
  // Decide whether to attempt insertion or deletion
  // Use 50/50 probability or bias based on current occupancy

  if (N_active_ == 0 || (N_active_ < max_N_ && rng_.uniformRandomNumber() < 0.5)) {
    return attemptInsertion();
  } else {
    return attemptDeletion();
  }
}

//-------------------------------------------------------------------------------------------------
// NCMCSampler implementation
//-------------------------------------------------------------------------------------------------
NCMCSampler::NCMCSampler(AtomGraph* topology,
                         PhaseSpace* ps,
                         StaticExclusionMask* exclusions,
                         Thermostat* thermostat,
                         double temperature,
                         const GhostMoleculeMetadata& ghost_metadata,
                         int n_pert_steps,
                         int n_prop_steps_per_pert,
                         double timestep,
                         const std::vector<double>& lambdas,
                         bool record_traj,
                         const std::vector<int>& ref_atoms,
                         double sphere_radius,
                         double mu_ex,
                         double standard_volume,
                         double adams,
                         double adams_shift,
                         int max_N,
                         topology::ImplicitSolventModel gb_model,
                         const std::string& resname,
                         const std::string& ghost_file,
                         const std::string& log_file) :
  GCMCSphereSampler(topology, ps, exclusions, thermostat, temperature, ghost_metadata, ref_atoms, sphere_radius,
                    mu_ex, standard_volume, adams, adams_shift, max_N,
                    gb_model, resname, ghost_file, log_file),
  record_traj_{record_traj}
{
  // Setup protocol
  protocol_.setPropagationStepsPerPerturbation(n_prop_steps_per_pert);
  protocol_.setTimestep(timestep);

  // Create or validate lambda schedule
  if (lambdas.empty()) {
    protocol_.generateLinearSchedule(n_pert_steps);
  } else {
    if (static_cast<int>(lambdas.size()) != n_pert_steps + 1) {
      rtErr("Lambda schedule must have n_pert_steps + 1 points", "NCMCSampler");
    }
    if (std::abs(lambdas[0]) > 1.0e-6 || std::abs(lambdas.back() - 1.0) > 1.0e-6) {
      rtErr("Lambda schedule must start at 0 and end at 1", "NCMCSampler");
    }
    protocol_.setLambdaSchedule(lambdas);
  }

  // DynamicsControls uses default settings - dynaStep will use timestep from protocol

  log_stream_ << "# NCMC protocol:\n";
  log_stream_ << "#   Perturbation steps: " << n_pert_steps << "\n";
  log_stream_ << "#   Propagation steps per pert: " << n_prop_steps_per_pert << "\n";
  log_stream_ << "#   Timestep: " << timestep << " fs\n";
  log_stream_ << "#   Switching time: " << protocol_.getSwitchingTime() << " ps\n";
  log_stream_.flush();

  // Note: Trajectory recording would be initialized here if needed
  // Since we're using raw pointer, the calling code must manage it
}

//-------------------------------------------------------------------------------------------------
void NCMCSampler::propagateSystem(int n_steps) {
  // OPTIMIZATION: During propagateSystem, lambda is CONSTANT (changes happen between calls)
  // Profiling showed abstract regeneration was taking 88% of runtime, not 1% as originally thought!
  // Generate abstracts ONCE before the loop, eliminating 10,000+ regenerations per GCMC move

  static int propagate_call_count = 0;
  static double total_abstract_time = 0.0;
  static double total_loop_time = 0.0;
  static double total_download_time = 0.0;

  auto abstract_start = std::chrono::high_resolution_clock::now();
  const ValenceKit<double> vk = topology_->getDoublePrecisionValenceKit();
  const NonbondedKit<double> nbk = topology_->getDoublePrecisionNonbondedKit();
  const StaticExclusionMaskReader ser = exclusions_->data();
  const ImplicitSolventKit<double> isk = topology_->getDoublePrecisionImplicitSolventKit();
  const VirtualSiteKit<double> vsk = topology_->getDoublePrecisionVirtualSiteKit();
  const ChemicalDetailsKit cdk = topology_->getChemicalDetailsKit();
  const ConstraintKit<double> cnk = topology_->getDoublePrecisionConstraintKit();
  const restraints::RestraintApparatus empty_restraints(topology_);
  const RestraintKit<double, double2, double4> rar = empty_restraints.dpData();

  // Get lambda arrays (CPU-side pointers)
  const double* lambda_vdw_ptr = lambda_vdw_.data();
  const double* lambda_ele_ptr = lambda_ele_.data();

  // Create lambda-aware nonbonded kit
  const topology::LambdaNonbondedKit<double> lambda_nbk(
      nbk.natom, nbk.n_q_types, nbk.n_lj_types, nbk.coulomb_constant, nbk.charge,
      nbk.q_idx, nbk.lj_idx, nbk.q_parameter, nbk.lja_coeff, nbk.ljb_coeff, nbk.ljc_coeff,
      nbk.lja_14_coeff, nbk.ljb_14_coeff, nbk.ljc_14_coeff, nbk.lj_sigma, nbk.lj_14_sigma,
      nbk.nb11x, nbk.nb11_bounds, nbk.nb12x, nbk.nb12_bounds, nbk.nb13x, nbk.nb13_bounds,
      nbk.nb14x, nbk.nb14_bounds, nbk.lj_type_corr, lambda_vdw_ptr, lambda_ele_ptr);

  std::vector<double> effective_gb_radii(topology_->getAtomCount(), 0.0);
  std::vector<double> psi(topology_->getAtomCount(), 0.0);
  std::vector<double> sumdeijda(topology_->getAtomCount(), 0.0);
  generalized_born_defaults::NeckGeneralizedBornKit<double> neck_gbk(0, 0.0, 0.0, nullptr, nullptr);
  auto abstract_end = std::chrono::high_resolution_clock::now();
  total_abstract_time += std::chrono::duration<double, std::milli>(abstract_end - abstract_start).count();

  auto loop_start = std::chrono::high_resolution_clock::now();

#ifdef STORMM_USE_CUDA
  // ==========================================================================================
  // GPU-ACCELERATED DYNAMICS PATH (Periodic systems only)
  // ==========================================================================================
  // IMPORTANT: GPU propagation requires AtomGraphSynthesis which is only created
  // for periodic systems. Non-periodic systems fall back to CPU path below.

  if (topology_synthesis_ != nullptr) {
    // Get timestep from dynamics controls
  const double dt = dyn_controls_.getTimeStep();
  const int n_atoms = topology_->getAtomCount();

  // Count coupled atoms (lambda > threshold)
  int n_coupled = 0;
  std::vector<int> coupled_atom_indices;
  for (int i = 0; i < n_atoms; i++) {
    if (lambda_vdw_.data()[i] > 0.01 || lambda_ele_.data()[i] > 0.01) {
      coupled_atom_indices.push_back(i);
      n_coupled++;
    }
  }

  // Early exit if no coupled atoms
  if (n_coupled > 0) {
    // Update coupled indices array
    if (n_coupled > coupled_indices_.size()) {
      rtErr("Coupled atom list of length " + std::to_string(n_coupled) +
            " exceeds pre-allocated workspace of " + std::to_string(coupled_indices_.size()) +
            " atoms. Increase the initial allocation.", "GCMCSampler::propagateSystem");
    }
    for (int i = 0; i < n_coupled; i++) {
      coupled_indices_.data()[i] = coupled_atom_indices[i];
    }
    if (kGcmcDebugLogs) {
      log_stream_ << "# DEBUG propagateSystem(): n_coupled=" << n_coupled
                  << " lambda_dirty=" << (gpu_lambda_arrays_dirty_ ? "true" : "false") << "\n";
      log_stream_.flush();
    }

    // Upload all necessary data to GPU (only the active portion of coupled_indices_)
    #ifdef STORMM_USE_HPC
    #ifdef STORMM_USE_HPC

    phase_space_->uploadPositions();

    #endif
    #endif
    lambda_vdw_.upload();
    lambda_ele_.upload();
    atom_sigma_.upload();
    atom_epsilon_.upload();
    coupled_indices_.upload(0, n_coupled);
    topology_->upload();

    coupled_indices_valid_ = true;
    gpu_lambda_arrays_dirty_ = false;
    coupled_atom_count_ = n_coupled;

    // Get device pointers from Hybrid arrays
    const int* d_coupled_indices = coupled_indices_.data(HybridTargetLevel::DEVICE);
    const double* d_lambda_vdw = lambda_vdw_.data(HybridTargetLevel::DEVICE);
    const double* d_lambda_ele = lambda_ele_.data(HybridTargetLevel::DEVICE);
    const double* d_atom_sigma = atom_sigma_.data(HybridTargetLevel::DEVICE);
    const double* d_atom_epsilon = atom_epsilon_.data(HybridTargetLevel::DEVICE);
    const double* d_masses = cdk.masses;  // Already points to GPU memory

    // Get exclusion mask data (already on GPU)
    const uint* d_exclusion_mask = ser.mask_data;
    const int* d_supertile_map = ser.supertile_map_idx;
    const int* d_tile_map = ser.tile_map_idx;
    const int supertile_stride = ser.supertile_stride_count;

    // Get PhaseSpace device pointers - note: STORMM stores x,y,z separately
    PhaseSpaceWriter psw = phase_space_->data();
    double* d_xcrd = psw.xcrd;  // Points to GPU memory when CUDA enabled
    double* d_ycrd = psw.ycrd;
    double* d_zcrd = psw.zcrd;
    double* d_xvel = psw.xvel;
    double* d_yvel = psw.yvel;
    double* d_zvel = psw.zvel;
    double* d_xfrc = psw.xfrc;
    double* d_yfrc = psw.yfrc;
    double* d_zfrc = psw.zfrc;

    // Get unit cell info
    const double* d_umat = psw.umat;  // Already points to GPU memory
    const UnitCellType unit_cell = psw.unit_cell;

    // Allocate temporary GPU arrays for energy outputs
    energy_output_elec_.resize(n_coupled);
    energy_output_vdw_.resize(n_coupled);
    double* d_per_atom_elec = energy_output_elec_.data(HybridTargetLevel::DEVICE);
    double* d_per_atom_vdw = energy_output_vdw_.data(HybridTargetLevel::DEVICE);

    // Create kernel launcher for GPU computation
    const card::GpuDetails gpu;  // Default constructor for GPU details
    const card::CoreKlManager launcher(gpu, *topology_synthesis_);

    // Create MM controls for timestep management and integration mode
    // Note: We now use MOVE_PARTICLES mode instead of ACCUMULATE, which enables
    // constraints, virtual sites, and thermostat integration
    // Use static to avoid memory leak from repeated allocations
    static mm::MolecularMechanicsControls mmctrl(0.0, 1);  // Will be updated below
    mmctrl.primeWorkUnitCounters(
        launcher,
        energy::EvaluateForce::YES,
        energy::EvaluateEnergy::NO,  // Will override for energy steps if needed
        mm::ClashResponse::NONE,
        synthesis::VwuGoal::MOVE_PARTICLES,  // CRITICAL: Changed from ACCUMULATE
        energy::PrecisionModel::DOUBLE,
        energy::PrecisionModel::DOUBLE,
        *topology_synthesis_);

    // Create cache resource for valence kernel thread blocks
    const energy::PrecisionModel valence_prec = energy::PrecisionModel::DOUBLE;
    const int2 vale_lp = launcher.getValenceKernelDims(
        valence_prec,
        energy::EvaluateForce::YES,
        energy::EvaluateEnergy::NO,
        numerics::AccumulationMethod::SPLIT,
        synthesis::VwuGoal::MOVE_PARTICLES,  // CRITICAL: Changed from ACCUMULATE
        mm::ClashResponse::NONE);

    // Allocate cache for thread blocks
    static energy::CacheResource* valence_cache_ptr = nullptr;
    static int cached_valence_blocks = 0;
    static int cached_valence_atoms = 0;
    const int required_valence_blocks = vale_lp.x;
    const int required_valence_atoms = mm::maximum_valence_work_unit_atoms;
    if (valence_cache_ptr == nullptr ||
        cached_valence_blocks < required_valence_blocks ||
        cached_valence_atoms < required_valence_atoms) {
      cached_valence_blocks = std::max(cached_valence_blocks, required_valence_blocks);
      cached_valence_atoms = std::max(cached_valence_atoms, required_valence_atoms);
      valence_cache_ptr = new energy::CacheResource(cached_valence_blocks,
                                                    cached_valence_atoms);
      if (kGcmcDebugLogs) {
        log_stream_ << "# DEBUG evaluateTotalEnergy(): allocated valence CacheResource blocks="
                    << cached_valence_blocks << " atoms=" << cached_valence_atoms << "\n";
        log_stream_.flush();
        std::cout << "DEBUG: allocated valence CacheResource blocks=" << cached_valence_blocks
                  << " atoms=" << cached_valence_atoms << std::endl;
      }
    }
    energy::CacheResource* valence_cache = valence_cache_ptr;

    // Get nonbonded launch parameters and create cache
    const int2 nonb_lp = launcher.getNonbondedKernelDims(energy::PrecisionModel::DOUBLE,
                                                        topology_synthesis_->getNonbondedWorkType(),
                                                        energy::EvaluateForce::YES,
                                                        energy::EvaluateEnergy::NO,
                                                        energy::AccumulationMethod::SPLIT,
                                                        gb_model_,
                                                        energy::ClashResponse::NONE);
    static energy::CacheResource* nonb_cache_ptr = nullptr;
    static int cached_nonb_blocks = 0;
    static int cached_nonb_atoms = 0;
    const int required_nonb_blocks = nonb_lp.x;
    const int required_nonb_atoms = synthesis::small_block_max_atoms;
    if (nonb_cache_ptr == nullptr ||
        cached_nonb_blocks < required_nonb_blocks ||
        cached_nonb_atoms < required_nonb_atoms) {
      cached_nonb_blocks = std::max(cached_nonb_blocks, required_nonb_blocks);
      cached_nonb_atoms = std::max(cached_nonb_atoms, required_nonb_atoms);
      nonb_cache_ptr = new energy::CacheResource(cached_nonb_blocks,
                                                 cached_nonb_atoms);
      if (kGcmcDebugLogs) {
        log_stream_ << "# DEBUG evaluateTotalEnergy(): allocated nonbonded CacheResource blocks="
                    << cached_nonb_blocks << " atoms=" << cached_nonb_atoms << "\n";
        log_stream_.flush();
        std::cout << "DEBUG: allocated nonbonded CacheResource blocks=" << cached_nonb_blocks
                  << " atoms=" << cached_nonb_atoms << std::endl;
      }
    }
    energy::CacheResource* nonb_cache = nonb_cache_ptr;

    if (se_synthesis_ == nullptr) {
      rtErr("Static exclusion mask synthesis is not initialized for GPU propagation.",
            "GCMCSampler::propagateSystem");
    }
    se_synthesis_->upload();

    // Run GPU dynamics loop with full integration (constraints + virtual sites + thermostat)
    for (int step = 0; step < n_steps; step++) {
      // Determine if this is an energy evaluation step
      const bool on_energy_step = false;  // Could be made configurable
      const energy::EvaluateEnergy eval_energy = on_energy_step ?
          energy::EvaluateEnergy::YES : energy::EvaluateEnergy::NO;

      mm::launchLambdaDynamicsStep(
          d_lambda_vdw, d_lambda_ele,
          d_coupled_indices, n_coupled,
          *topology_synthesis_,    // Topology synthesis (const reference)
          *se_synthesis_,          // Static exclusion mask synthesis (const reference)
          thermostat_,             // Thermostat (integrated automatically in MOVE_PARTICLES)
          ps_synthesis_,           // Phase space synthesis (coordinates, velocities, forces)
          &mmctrl,                 // MM controls (now required for MOVE_PARTICLES mode)
          &scorecard_,             // Score card for energy tracking
          launcher,                // Kernel launcher (const reference)
        valence_cache,           // Cache resource for valence kernel thread blocks
        nonb_cache,              // Cache resource for nonbonded kernel thread blocks
          eval_energy,             // Energy evaluation control
          gb_workspace_,           // GB workspace (nullptr if GB disabled)
          gb_model_);              // GB model (NONE if GB disabled)

      // Increment step counters
      mmctrl.incrementStep();
      if (thermostat_ != nullptr) {
        thermostat_->incrementStep();
      }
    }

    // Synchronize GPU to ensure all operations complete
    cudaDeviceSynchronize();
  }
  } // End if (topology_synthesis_ != nullptr) - GPU path for periodic systems

#else
  // ==========================================================================================
  // CPU-ONLY DYNAMICS PATH (fallback)
  // ==========================================================================================

  // OPTIMIZATION: Get data writers ONCE before loop to avoid repeated GPU synchronization
  ThermostatWriter<double> tstw = thermostat_->dpData();
  PhaseSpaceWriter psw = phase_space_->data();

  for (int step = 0; step < n_steps; step++) {
    lambdaDynaStep(&psw, &scorecard_, tstw, vk, lambda_nbk, isk, neck_gbk,
                   effective_gb_radii.data(), psi.data(), sumdeijda.data(),
                   rar, vsk, cdk, cnk, ser, dyn_controls_, 0,
                   0.75, 0.5);  // vdw_coupling_threshold=0.75, softcore_alpha=0.5
  }
#endif

  auto loop_end = std::chrono::high_resolution_clock::now();
  total_loop_time += std::chrono::duration<double, std::milli>(loop_end - loop_start).count();

  // OPTIMIZATION: Don't download after every propagate - batch downloads before energy evals
  auto download_start = std::chrono::high_resolution_clock::now();
  #ifdef STORMM_USE_CUDA
  // phase_space_->download();  // DISABLED - batch downloads to reduce overhead
  #endif
  auto download_end = std::chrono::high_resolution_clock::now();
  total_download_time += std::chrono::duration<double, std::milli>(download_end - download_start).count();

  propagate_call_count++;
  if (propagate_call_count % 100 == 0) {
    log_stream_ << "# PROPAGATE PROFILE (100 calls): Abstract=" << (total_abstract_time/100.0)
                << "ms, Loop=" << (total_loop_time/100.0)
                << "ms, Download=" << (total_download_time/100.0)
                << "ms, Total=" << ((total_abstract_time + total_loop_time + total_download_time)/100.0) << "ms\n";
    log_stream_.flush();
    total_abstract_time = 0.0;
    total_loop_time = 0.0;
    total_download_time = 0.0;
  }

  // Invalidate energy cache since coordinates changed during MD
  invalidateEnergyCache();
}

//-------------------------------------------------------------------------------------------------
double NCMCSampler::performNCMCProtocol(GCMCMolecule& mol, bool forward, bool propagate) {
  double work = 0.0;
  const auto& lambdas = protocol_.getLambdaSchedule();
  const int n_pert_steps = protocol_.getPerturbationSteps();
  const int n_prop_steps = protocol_.getPropagationStepsPerPerturbation();

  // Determine lambda schedule direction
  std::vector<double> lambda_values;
  if (forward) {
    // Insertion: 0 -> 1
    lambda_values = lambdas;
  } else {
    // Deletion: 1 -> 0
    lambda_values.reserve(lambdas.size());
    for (auto it = lambdas.rbegin(); it != lambdas.rend(); ++it) {
      lambda_values.push_back(*it);
    }
  }

  // Initial equilibration at starting lambda
  if (propagate && n_prop_steps > 0) {
    propagateSystem(n_prop_steps);
    // NOTE: PBC wrapping removed for performance - energy calculations already use PBC correctly
    // applyPBCToAllMolecules();
  }

  // Handle instant insertion/deletion (npert=0 case)
  if (n_pert_steps == 0) {
    // Instantaneous lambda change: evaluate energy difference in one step
    double E_before = evaluateTotalEnergy();
    adjustMoleculeLambdaGPU(mol, lambda_values[1]);  // Jump directly to final lambda (GPU-optimized)
    double E_after = evaluateTotalEnergy();

    work = E_after - E_before;

    if (std::isnan(E_after) || std::isnan(work)) {
      log_stream_ << "# NaN detected during instant insertion/deletion\n";
      stats_.n_explosions++;
      return std::numeric_limits<double>::infinity();
    }

    return work;
  }

  // NCMC switching protocol with GPU-side work accumulation
  //
  // The NCMC method improves acceptance by switching lambda gradually
  // while allowing MD relaxation. Work is accumulated as:
  //   W = Σ ΔE(λ_i → λ_{i+1}) at fixed coordinates
  //
  // Key steps for each perturbation:
  //   1. Evaluate energy at current lambda (E_before)
  //   2. Change lambda (topology parameters unchanged, only scaling changes)
  //   3. Evaluate energy at NEW lambda but SAME coordinates (E_after)
  //   4. Accumulate work: W += (E_after - E_before) **ON GPU**
  //   5. Propagate system with MD at the new lambda
  //
  // This ensures work is calculated at fixed coords (requirement for NCMC theory)
  //
  // GPU OPTIMIZATION: Work accumulation happens entirely on GPU, eliminating
  // 100 CPU↔GPU transfers per NCMC move (2 per perturbation × 50 steps).
  // Only the final work value is downloaded.

#ifdef STORMM_USE_CUDA
  // GPU-accelerated NCMC with on-device work accumulation and lambda scheduling
  using card::HybridTargetLevel;

  // Zero work accumulator on GPU
  cudaMemset(work_accumulator_.data(HybridTargetLevel::DEVICE), 0, sizeof(double));

  // Prepare GPU arrays for energy evaluation
  const int n_atoms = topology_->getAtomCount();
  PhaseSpaceWriter psw = phase_space_->data();
  const StaticExclusionMaskReader ser = exclusions_->data();
  const NonbondedKit<double> nbk = topology_->getDoublePrecisionNonbondedKit();

  // Build per-atom lambda and LJ arrays (same as evaluateTotalEnergy)
  std::vector<double> lambda_vdw_per_atom(n_atoms, 1.0);
  std::vector<double> lambda_ele_per_atom(n_atoms, 1.0);
  std::vector<double> atom_sigma(n_atoms);
  std::vector<double> atom_epsilon(n_atoms);

  // Initialize with cached LJ parameters
  for (int i = 0; i < n_atoms; i++) {
    const int lj_type = nbk.lj_idx[i];
    atom_sigma[i] = cached_lj_sigma_[lj_type];
    atom_epsilon[i] = cached_lj_epsilon_[lj_type];
  }

  // Override lambda for all molecules, LJ for ghosts only
  for (const auto& m : molecules_) {
    for (size_t i = 0; i < m.atom_indices.size(); i++) {
      int idx = m.atom_indices[i];
      lambda_vdw_per_atom[idx] = m.lambda_vdw;
      lambda_ele_per_atom[idx] = m.lambda_ele;

      if (m.status == GCMCMoleculeStatus::GHOST) {
        atom_sigma[idx] = m.original_sigma[i];
        atom_epsilon[idx] = m.original_epsilon[i];
      }
    }
  }

  // Copy to Hybrid arrays
  double* lambda_vdw_ptr = lambda_vdw_.data();
  double* lambda_ele_ptr = lambda_ele_.data();
  double* sigma_ptr = atom_sigma_.data();
  double* epsilon_ptr = atom_epsilon_.data();

  for (int i = 0; i < n_atoms; i++) {
    lambda_vdw_ptr[i] = lambda_vdw_per_atom[i];
    lambda_ele_ptr[i] = lambda_ele_per_atom[i];
    sigma_ptr[i] = atom_sigma[i];
    epsilon_ptr[i] = atom_epsilon[i];
  }

  // Build coupled atom list
  constexpr double LAMBDA_THRESHOLD = 0.01;
  int* coupled_ptr = coupled_indices_.data();
  int n_coupled = 0;
  for (int i = 0; i < n_atoms; i++) {
    if (lambda_vdw_per_atom[i] > LAMBDA_THRESHOLD || lambda_ele_per_atom[i] > LAMBDA_THRESHOLD) {
      coupled_ptr[n_coupled++] = i;
    }
  }

  // Prepare GPU-side lambda scheduling
  // Resize lambda schedule to match actual protocol size (minimum 2 for safety)
  const int schedule_size = std::max(2, n_pert_steps + 1);
  lambda_schedule_.resize(schedule_size);

  // Store the lambda schedule on GPU (all n_pert_steps+1 values)
  double* schedule_ptr = lambda_schedule_.data();
  for (int i = 0; i <= n_pert_steps; i++) {
    schedule_ptr[i] = lambda_values[i];
  }

  // Store molecule atom indices on GPU
  // Resize to actual molecule size (minimum 1 for safety)
  const int n_mol_atoms = mol.atom_indices.size();
  const int mol_indices_size = std::max(1, static_cast<int>(n_mol_atoms));
  molecule_atom_indices_.resize(mol_indices_size);

  int* mol_indices_ptr = molecule_atom_indices_.data();
  int* mol_count_ptr = molecule_atom_count_.data();
  mol_count_ptr[0] = n_mol_atoms;
  for (int i = 0; i < n_mol_atoms; i++) {
    mol_indices_ptr[i] = mol.atom_indices[i];
  }

  // Upload to GPU once (including new lambda scheduling arrays)
  lambda_vdw_.upload();
  lambda_ele_.upload();
  atom_sigma_.upload();
  atom_epsilon_.upload();
  coupled_indices_.upload(0, n_coupled);
  energy_output_elec_.upload();
  energy_output_vdw_.upload();
  energy_before_elec_.upload();
  energy_before_vdw_.upload();
  energy_after_elec_.upload();
  energy_after_vdw_.upload();
  work_accumulator_.upload();
  lambda_schedule_.upload();
  molecule_atom_indices_.upload();
  molecule_atom_count_.upload();

  coupled_indices_valid_ = true;
  gpu_lambda_arrays_dirty_ = false;
  coupled_atom_count_ = n_coupled;

  // Get device pointers
  const int* d_coupled_indices = coupled_indices_.data(HybridTargetLevel::DEVICE);
  double* d_lambda_vdw = lambda_vdw_.data(HybridTargetLevel::DEVICE);
  double* d_lambda_ele = lambda_ele_.data(HybridTargetLevel::DEVICE);
  const double* d_atom_sigma = atom_sigma_.data(HybridTargetLevel::DEVICE);
  const double* d_atom_epsilon = atom_epsilon_.data(HybridTargetLevel::DEVICE);
  double* d_per_atom_elec = energy_output_elec_.data(HybridTargetLevel::DEVICE);
  double* d_per_atom_vdw = energy_output_vdw_.data(HybridTargetLevel::DEVICE);
  double* d_energy_before_elec = energy_before_elec_.data(HybridTargetLevel::DEVICE);
  double* d_energy_before_vdw = energy_before_vdw_.data(HybridTargetLevel::DEVICE);
  double* d_energy_after_elec = energy_after_elec_.data(HybridTargetLevel::DEVICE);
  double* d_energy_after_vdw = energy_after_vdw_.data(HybridTargetLevel::DEVICE);
  double* d_work = work_accumulator_.data(HybridTargetLevel::DEVICE);
  const double* d_lambda_schedule = lambda_schedule_.data(HybridTargetLevel::DEVICE);
  const int* d_mol_indices = molecule_atom_indices_.data(HybridTargetLevel::DEVICE);

  // OPTIMIZATION: Get MD abstracts ONCE before NCMC loop to eliminate 100 synchronizations
  // This keeps coordinates on GPU throughout the entire NCMC protocol (psw already obtained above)
  const ValenceKit<double> vk = topology_->getDoublePrecisionValenceKit();
  const ImplicitSolventKit<double> isk = topology_->getDoublePrecisionImplicitSolventKit();
  const VirtualSiteKit<double> vsk = topology_->getDoublePrecisionVirtualSiteKit();
  const ChemicalDetailsKit cdk = topology_->getChemicalDetailsKit();
  const ConstraintKit<double> cnk = topology_->getDoublePrecisionConstraintKit();
  const restraints::RestraintApparatus empty_restraints(topology_);
  const RestraintKit<double, double2, double4> rar = empty_restraints.dpData();
  ThermostatWriter<double> tstw = thermostat_->dpData();

  std::vector<double> effective_gb_radii(n_atoms, 0.0);
  std::vector<double> psi(n_atoms, 0.0);
  std::vector<double> sumdeijda(n_atoms, 0.0);
  generalized_born_defaults::NeckGeneralizedBornKit<double> neck_gbk(0, 0.0, 0.0, nullptr, nullptr);

#ifdef STORMM_USE_CUDA
  if (launcher_ != nullptr && topology_synthesis_ != nullptr && ps_synthesis_ != nullptr) {
    const int system_id = 0;
    const synthesis::SyNonbondedKit<double, double2> synbk =
        topology_synthesis_->getDoublePrecisionNonbondedKit(HybridTargetLevel::DEVICE);
    const StaticExclusionMaskReader ser_dev = exclusions_->data(HybridTargetLevel::DEVICE);
    synthesis::PsSynthesisWriter psw_dev = ps_synthesis_->data(HybridTargetLevel::DEVICE);

    const int atom_offset   = synbk.atom_offsets[system_id];
    const int lj_offset     = synbk.ljabc_offsets[system_id];
    const int n_lj_types    = synbk.n_lj_types[system_id];
    const double* d_charges = synbk.charge + atom_offset;
    const int* d_lj_idx     = synbk.lj_idx + atom_offset;
    const double2* d_ljab   = synbk.ljab_coeff + lj_offset;

    const uint* d_exclusion_mask = ser_dev.mask_data;
    const int* d_supertile_map   = ser_dev.supertile_map_idx;
    const int* d_tile_map        = ser_dev.tile_map_idx;
    const int supertile_stride   = ser_dev.supertile_stride_count;

    llint* d_xcrd = psw_dev.xcrd;
    llint* d_ycrd = psw_dev.ycrd;
    llint* d_zcrd = psw_dev.zcrd;
    const double* d_umat = psw_dev.umat;
    const UnitCellType unit_cell = psw_dev.unit_cell;
    const float inv_gpos_scale = psw_dev.inv_gpos_scale;
    const float frc_scale      = psw_dev.frc_scale;

    double* d_work = work_accumulator_.data(HybridTargetLevel::DEVICE);
    cudaMemset(d_work, 0, sizeof(double));

    static energy::CacheResource* valence_cache_ptr = nullptr;
    static energy::CacheResource* nonb_cache_ptr    = nullptr;

    if (se_synthesis_ == nullptr) {
      rtErr("Static exclusion mask synthesis is not initialized for GPU NCMC.",
            "NCMCSampler::performNCMCProtocol");
    }
    se_synthesis_->upload();

    const card::GpuDetails gpu;
    const card::CoreKlManager launcher(gpu, *topology_synthesis_);
    static mm::MolecularMechanicsControls mmctrl(0.0, 1);

    const int2 vale_lp = launcher.getValenceKernelDims(
        constants::PrecisionModel::DOUBLE,
        EvaluateForce::YES,
        energy::EvaluateEnergy::NO,
        energy::AccumulationMethod::SPLIT,
        synthesis::VwuGoal::MOVE_PARTICLES,
        energy::ClashResponse::NONE);
    const int2 nonb_lp = launcher.getNonbondedKernelDims(
        constants::PrecisionModel::DOUBLE,
        topology_synthesis_->getNonbondedWorkType(),
        EvaluateForce::YES,
        energy::EvaluateEnergy::NO,
        energy::AccumulationMethod::SPLIT,
        topology_synthesis_->getImplicitSolventModel(),
        energy::ClashResponse::NONE);

    if (valence_cache_ptr == nullptr) {
      valence_cache_ptr = new energy::CacheResource(vale_lp.x, maximum_valence_work_unit_atoms);
    }
    if (nonb_cache_ptr == nullptr) {
      nonb_cache_ptr = new energy::CacheResource(nonb_lp.x, synthesis::small_block_max_atoms);
    }

    mmctrl.primeWorkUnitCounters(
        launcher,
        energy::EvaluateForce::YES,
        energy::EvaluateEnergy::NO,
        mm::ClashResponse::NONE,
        synthesis::VwuGoal::MOVE_PARTICLES,
        constants::PrecisionModel::DOUBLE,
        constants::PrecisionModel::DOUBLE,
        *topology_synthesis_);

    for (int i = 0; i < n_pert_steps; i++) {
      energy::launchLambdaScaledNonbondedWithReduction(
          n_atoms, n_coupled, d_coupled_indices,
          d_xcrd, d_ycrd, d_zcrd, d_charges,
          d_lambda_vdw, d_lambda_ele,
          d_lj_idx, n_lj_types, d_ljab,
          d_exclusion_mask, d_supertile_map, d_tile_map, supertile_stride,
          d_umat, unit_cell, synbk.coulomb, ewald_coeff_,
          inv_gpos_scale, frc_scale,
          d_per_atom_elec, d_per_atom_vdw,
          d_energy_before_elec, d_energy_before_vdw,
          nullptr, nullptr, nullptr,
          gb_workspace_, gb_model_);

      energy::launchUpdateLambdaFromSchedule(
          i + 1, d_lambda_schedule, d_mol_indices, n_mol_atoms,
          VDW_COUPLING_THRESHOLD, d_lambda_vdw, d_lambda_ele);

      energy::launchLambdaScaledNonbondedWithReduction(
          n_atoms, n_coupled, d_coupled_indices,
          d_xcrd, d_ycrd, d_zcrd, d_charges,
          d_lambda_vdw, d_lambda_ele,
          d_lj_idx, n_lj_types, d_ljab,
          d_exclusion_mask, d_supertile_map, d_tile_map, supertile_stride,
          d_umat, unit_cell, synbk.coulomb, ewald_coeff_,
          inv_gpos_scale, frc_scale,
          d_per_atom_elec, d_per_atom_vdw,
          d_energy_after_elec, d_energy_after_vdw,
          nullptr, nullptr, nullptr,
          gb_workspace_, gb_model_);

      energy::launchAccumulateWorkDelta(
          d_energy_before_elec, d_energy_before_vdw,
          d_energy_after_elec, d_energy_after_vdw,
          d_work);

      if (propagate && n_prop_steps > 0) {
        for (int step = 0; step < n_prop_steps; step++) {
          launchLambdaDynamicsStep(
              d_lambda_vdw,
              d_lambda_ele,
              d_coupled_indices,
              n_coupled,
              *topology_synthesis_,
              *se_synthesis_,
              thermostat_,
              ps_synthesis_,
              &mmctrl,
              &scorecard_,
              launcher,
              valence_cache_ptr,
              nonb_cache_ptr,
              EvaluateEnergy::NO,
              gb_workspace_,
              gb_model_);

          mmctrl.incrementStep();
          thermostat_->incrementStep();
        }
      }
    }

    work_accumulator_.download();
    work = work_accumulator_.data()[0];

    lambda_vdw_.download();
    lambda_ele_.download();
    gpu_lambda_arrays_dirty_ = true;
    coupled_indices_valid_ = false;
    coupled_atom_count_ = 0;

    adjustMoleculeLambda(mol, lambda_values[n_pert_steps]);

    phase_space_->download();
    ps_synthesis_->download();
    invalidateEnergyCache();

    if (std::isnan(work)) {
      log_stream_ << "# NaN detected during NCMC protocol\n";
      stats_.n_explosions++;
      return std::numeric_limits<double>::infinity();
    }

    return work;
  }
#endif

  // GPU-accelerated loop disabled - fall through to use CPU evaluateTotalEnergy approach below
  // (Same as the #else clause, but using the already-initialized variables from GPU setup)
  for (int i = 0; i < n_pert_steps; i++) {
    double E_before = evaluateTotalEnergy();
    adjustMoleculeLambda(mol, lambda_values[i + 1]);
    double E_after = evaluateTotalEnergy();
    double delta_E = E_after - E_before;
    work += delta_E;

    if (std::isnan(E_after) || std::isnan(work)) {
      log_stream_ << "# NaN detected during NCMC protocol\n";
      stats_.n_explosions++;
      return std::numeric_limits<double>::infinity();
    }

    if (propagate && n_prop_steps > 0) {
      try {
        propagateSystem(n_prop_steps);
        applyPBCToAllMolecules();
      } catch (...) {
        log_stream_ << "# Integration failure during NCMC protocol\n";
        stats_.n_explosions++;
        return std::numeric_limits<double>::infinity();
      }
    }
  }

#else
  // CPU fallback: original implementation
  for (int i = 0; i < n_pert_steps; i++) {
    double E_before = evaluateTotalEnergy();
    adjustMoleculeLambda(mol, lambda_values[i + 1]);
    double E_after = evaluateTotalEnergy();
    double delta_E = E_after - E_before;
    work += delta_E;

    if (std::isnan(E_after) || std::isnan(work)) {
      log_stream_ << "# NaN detected during NCMC protocol\n";
      stats_.n_explosions++;
      return std::numeric_limits<double>::infinity();
    }

    if (propagate && n_prop_steps > 0) {
      try {
        propagateSystem(n_prop_steps);
        applyPBCToAllMolecules();
      } catch (...) {
        log_stream_ << "# Integration failure during NCMC protocol\n";
        stats_.n_explosions++;
        return std::numeric_limits<double>::infinity();
      }
    }
  }
#endif

  return work;
}

//-------------------------------------------------------------------------------------------------
bool NCMCSampler::attemptInsertion() {
  stats_.n_moves++;

  // Check if sphere is full
  if (N_active_ >= max_N_) {
    log_stream_ << "# Sphere full, rejecting NCMC insertion\n";
    stats_.outcomes.push_back("rejected NCMC insert (sphere full)");
    return false;
  }

  // Select random ghost molecule
  GCMCMolecule* ghost_mol = selectRandomGhostMolecule();
  if (ghost_mol == nullptr) {
    log_stream_ << "# No ghost molecules available\n";
    stats_.outcomes.push_back("rejected NCMC insert (no ghosts)");
    return false;
  }

  // Select insertion site
  double3 insertion_site = selectInsertionSite();

  // Save current state
  std::vector<double3> saved_coords = saveCoordinates(*ghost_mol);
  std::vector<double3> saved_vels = saveVelocities(*ghost_mol);

  // Move molecule to insertion site
  double3 current_cog = calculateMoleculeCOG(*ghost_mol);
  PhaseSpaceWriter psw = phase_space_->data();
  double* xcrd = psw.xcrd;
  double* ycrd = psw.ycrd;
  double* zcrd = psw.zcrd;

  for (size_t i = 0; i < ghost_mol->atom_indices.size(); i++) {
    int atom_idx = ghost_mol->atom_indices[i];
    xcrd[atom_idx] = insertion_site.x + (saved_coords[i].x - current_cog.x);
    ycrd[atom_idx] = insertion_site.y + (saved_coords[i].y - current_cog.y);
    zcrd[atom_idx] = insertion_site.z + (saved_coords[i].z - current_cog.z);
  }

  applyPBC(*ghost_mol);

  // Assign Maxwell-Boltzmann velocities
  double* xvel = psw.xvel;
  double* yvel = psw.yvel;
  double* zvel = psw.zvel;

  for (int atom_idx : ghost_mol->atom_indices) {
    double mass = topology_->getAtomicMass<double>(atom_idx);
    double3 vel = generateMaxwellBoltzmannVelocity(mass);
    xvel[atom_idx] = vel.x;
    yvel[atom_idx] = vel.y;
    zvel[atom_idx] = vel.z;
  }

  // Note: Coordinates/velocities will be uploaded by evaluateTotalEnergy() when needed
  // Manual uploadPositions() here was causing CUDA driver state exhaustion

  // Set initial lambda from NCMC protocol schedule (0.05 with LAMBDA_GHOST_THRESHOLD optimization)
  const auto& lambdas = protocol_.getLambdaSchedule();
  adjustMoleculeLambdaAuto(this, *ghost_mol, lambdas[0]);

  if (kGcmcDebugLogs) {
    log_stream_ << "# DEBUG: Initial lambda for insertion = " << lambdas[0]
                << " (vdw=" << ghost_mol->lambda_vdw << ", ele=" << ghost_mol->lambda_ele << ")\n";
  }

  // Save thermostat state before NCMC protocol
  // The thermostat contains internal state that must be preserved for proper detailed balance
  Thermostat thermostat_backup = *thermostat_;

  // Run NCMC protocol (0 -> 1)
  double protocol_work = performNCMCProtocol(*ghost_mol, true);

  // Check for protocol failure
  if (std::isinf(protocol_work)) {
    // Protocol failed, restore state including thermostat
    *thermostat_ = thermostat_backup;
    adjustMoleculeLambdaAuto(this, *ghost_mol, 0.0);
    restoreCoordinates(*ghost_mol, saved_coords);
    restoreVelocities(*ghost_mol, saved_vels, true);  // Reverse velocities
    stats_.outcomes.push_back("rejected NCMC insert (protocol failed)");
    return false;
  }

  // Download final coordinates
  #ifdef STORMM_USE_CUDA
  phase_space_->download();
  #endif

  // Check if molecule still in sphere
  if (!isMoleculeInSphere(*ghost_mol)) {
    log_stream_ << "# Molecule left sphere during NCMC insertion\n";
    stats_.n_left_sphere++;
    *thermostat_ = thermostat_backup;  // Restore thermostat state
    adjustMoleculeLambdaAuto(this, *ghost_mol, 0.0);
    restoreCoordinates(*ghost_mol, saved_coords);
    restoreVelocities(*ghost_mol, saved_vels, true);
    stats_.outcomes.push_back("rejected NCMC insert (left sphere)");
    return false;
  }

  // NCMC insertion acceptance probability
  //
  // P_accept = min(1, exp(B - βW) / (N + 1))
  //
  // Where:
  //   B = β×μ_ex + ln(V_sphere/V_std)  [Adams parameter]
  //   W = protocol work in kcal/mol
  //   β = 1/kT
  //   N = current number of active molecules
  //
  // The (N+1) term accounts for the increased molecule count after insertion
  double acc_prob = std::min(1.0, std::exp(B_ - beta_ * protocol_work) / (N_active_ + 1.0));

  // Metropolis acceptance
  stats_.n_inserts++;
  stats_.insert_works.push_back(protocol_work);
  stats_.insert_acceptance_probs.push_back(acc_prob);
  stats_.move_resids.push_back(ghost_mol->resid);

  if (rng_.uniformRandomNumber() < acc_prob) {
    // ACCEPT - fully activate molecule
    adjustMoleculeLambdaAuto(this, *ghost_mol, 1.0);
    if (kGcmcDebugLogs) {
      log_stream_ << "# DEBUG ACCEPT: lambda set to 1.0 for resid " << ghost_mol->resid << "\n";
    }
    ghost_mol->status = GCMCMoleculeStatus::ACTIVE;
    N_active_++;
    stats_.n_accepted++;
    stats_.n_accepted_inserts++;
    stats_.accepted_insert_works.push_back(protocol_work);
    stats_.N_history.push_back(N_active_);
    stats_.acc_rate_history.push_back(stats_.getAcceptanceRate());

    logMove("ACCEPTED NCMC insertion", ghost_mol->resid, protocol_work, acc_prob);
    stats_.outcomes.push_back("accepted NCMC insert");

    return true;
  } else {
    // REJECT - restore state with velocity reversal and thermostat state
    *thermostat_ = thermostat_backup;  // Restore thermostat state
    adjustMoleculeLambdaAuto(this, *ghost_mol, 0.0);
    restoreCoordinates(*ghost_mol, saved_coords);
    restoreVelocities(*ghost_mol, saved_vels, true);  // Reverse for detailed balance

    stats_.N_history.push_back(N_active_);
    stats_.acc_rate_history.push_back(stats_.getAcceptanceRate());

    logMove("REJECTED NCMC insertion", ghost_mol->resid, protocol_work, acc_prob);
    stats_.outcomes.push_back("rejected NCMC insert");

    return false;
  }
}

//-------------------------------------------------------------------------------------------------
bool NCMCSampler::attemptDeletion() {
  stats_.n_moves++;

  // Select random active molecule
  GCMCMolecule* active_mol = selectRandomActiveMolecule();
  if (active_mol == nullptr) {
    log_stream_ << "# No active molecules available\n";
    stats_.outcomes.push_back("rejected NCMC delete (no active)");
    return false;
  }

  // Check if molecule is in sphere
  if (!isMoleculeInSphere(*active_mol)) {
    log_stream_ << "# Molecule not in sphere, cannot delete\n";
    stats_.outcomes.push_back("rejected NCMC delete (outside sphere)");
    return false;
  }

  // Save current state
  std::vector<double3> saved_coords = saveCoordinates(*active_mol);
  std::vector<double3> saved_vels = saveVelocities(*active_mol);

  // Save thermostat state before NCMC protocol for detailed balance
  Thermostat thermostat_backup = *thermostat_;

  // Run NCMC protocol (1 -> 0)
  double protocol_work = performNCMCProtocol(*active_mol, false);

  // Check for protocol failure
  if (std::isinf(protocol_work)) {
    // Protocol failed, restore state including thermostat
    *thermostat_ = thermostat_backup;
    adjustMoleculeLambdaAuto(this, *active_mol, 1.0);
    restoreCoordinates(*active_mol, saved_coords);
    restoreVelocities(*active_mol, saved_vels, true);
    stats_.outcomes.push_back("rejected NCMC delete (protocol failed)");
    return false;
  }

  // NCMC deletion acceptance probability
  //
  // P_accept = min(1, N × exp(-B - βW))
  //
  // Where:
  //   B = β×μ_ex + ln(V_sphere/V_std)  [Adams parameter]
  //   W = protocol work in kcal/mol
  //   β = 1/kT
  //   N = current number of active molecules (before deletion)
  //
  // The N term accounts for the decreased molecule count after deletion
  // Note the sign change: +B for insertion, -B for deletion (detailed balance)
  double acc_prob = std::min(1.0, N_active_ * std::exp(-B_ - beta_ * protocol_work));

  // Metropolis acceptance
  stats_.n_deletes++;
  stats_.delete_works.push_back(protocol_work);
  stats_.delete_acceptance_probs.push_back(acc_prob);
  stats_.move_resids.push_back(active_mol->resid);

  if (rng_.uniformRandomNumber() < acc_prob) {
    // ACCEPT
    active_mol->status = GCMCMoleculeStatus::GHOST;
    // Set lambda=0 to fully decouple ghost (NCMC ends at 0.05, but ghosts should be at 0.0)
    adjustMoleculeLambdaAuto(this, *active_mol, 0.0);
    N_active_--;
    stats_.n_accepted++;
    stats_.n_accepted_deletes++;
    stats_.accepted_delete_works.push_back(protocol_work);
    stats_.N_history.push_back(N_active_);
    stats_.acc_rate_history.push_back(stats_.getAcceptanceRate());

    logMove("ACCEPTED NCMC deletion", active_mol->resid, protocol_work, acc_prob);
    stats_.outcomes.push_back("accepted NCMC delete");

    return true;
  } else {
    // REJECT - restore state with velocity reversal and thermostat state
    *thermostat_ = thermostat_backup;
    adjustMoleculeLambdaAuto(this, *active_mol, 1.0);
    restoreCoordinates(*active_mol, saved_coords);
    restoreVelocities(*active_mol, saved_vels, true);

    stats_.N_history.push_back(N_active_);
    stats_.acc_rate_history.push_back(stats_.getAcceptanceRate());

    logMove("REJECTED NCMC deletion", active_mol->resid, protocol_work, acc_prob);
    stats_.outcomes.push_back("rejected NCMC delete");

    return false;
  }
}

//-------------------------------------------------------------------------------------------------
bool NCMCSampler::runGCMCCycle() {
  // Decide whether to attempt insertion or deletion
  // Use 50/50 probability or bias based on current occupancy

  if (N_active_ == 0 || (N_active_ < max_N_ && rng_.uniformRandomNumber() < 0.5)) {
    return attemptInsertion();
  } else {
    return attemptDeletion();
  }
}

//-------------------------------------------------------------------------------------------------
const NCMCProtocol& NCMCSampler::getProtocol() const {
  return protocol_;
}

//-------------------------------------------------------------------------------------------------
// GCMCSystemSampler implementation
//-------------------------------------------------------------------------------------------------
GCMCSystemSampler::GCMCSystemSampler(AtomGraph* topology,
                                     PhaseSpace* ps,
                                     StaticExclusionMask* exclusions,
                                     Thermostat* thermostat,
                                     double temperature,
                                     const GhostMoleculeMetadata& ghost_metadata,
                                     double mu_ex,
                                     double standard_volume,
                                     double adams,
                                     double adams_shift,
                                     topology::ImplicitSolventModel gb_model,
                                     const std::string& resname,
                                     const std::string& ghost_file,
                                     const std::string& log_file) :
  GCMCSampler(topology, ps, exclusions, thermostat, temperature, ghost_metadata,
              gb_model, resname, ghost_file, log_file),
  mu_ex_{mu_ex},
  standard_volume_{standard_volume},
  adaptive_b_enabled_{false},
  current_stage_{AnnealingStage::DISCOVERY},
  stage1_moves_{0},
  stage2_moves_{0},
  stage3_moves_{0},
  b_discovery_{10.0},
  target_occupancy_{0.5},
  coarse_learning_rate_{0.1},
  fine_learning_rate_{0.01},
  b_min_{0.1},
  b_max_{15.0},
  n_max_fragments_{0},
  current_adaptive_b_{5.0},
  move_counter_{0}
{
  // Calculate box volume from PhaseSpace box dimensions
  // Box dimensions: [a, b, c, alpha, beta, gamma]
  const double* box_dims = phase_space_->getBoxDimensionsHandle()->data();
  const double a = box_dims[0];
  const double b = box_dims[1];
  const double c = box_dims[2];
  const double alpha = box_dims[3] * (pi / 180.0);  // Convert to radians
  const double beta = box_dims[4] * (pi / 180.0);
  const double gamma = box_dims[5] * (pi / 180.0);

  // Calculate box volume for general triclinic cell
  box_volume_ = a * b * c * std::sqrt(1.0 + 2.0 * std::cos(alpha) * std::cos(beta) * std::cos(gamma)
                                       - std::cos(alpha) * std::cos(alpha)
                                       - std::cos(beta) * std::cos(beta)
                                       - std::cos(gamma) * std::cos(gamma));

  // Calculate B parameter (Adams parameter)
  // B = beta * mu_ex + ln(V_box / V_std)
  if (std::isnan(adams)) {
    // Use mu_ex to calculate B
    B_ = beta_ * mu_ex_ + std::log(box_volume_ / standard_volume_);
  } else {
    // Use provided Adams B parameter directly
    B_ = adams + adams_shift;
  }

  // Log system-wide GCMC parameters
  log_stream_ << "# System-wide GCMC Sampler initialized\n";
  log_stream_ << "#   Temperature: " << temperature_ << " K\n";
  log_stream_ << "#   kT: " << kT_ << " kcal/mol\n";
  log_stream_ << "#   Box volume: " << box_volume_ << " Angstrom^3\n";
  log_stream_ << "#   Standard volume: " << standard_volume_ << " Angstrom^3\n";
  log_stream_ << "#   Excess chemical potential: " << mu_ex_ << " kcal/mol\n";
  log_stream_ << "#   B parameter: " << B_ << "\n";
  log_stream_ << "#   Total ghost molecules: " << molecules_.size() << "\n";
  log_stream_.flush();
}

//-------------------------------------------------------------------------------------------------
double GCMCSystemSampler::getBoxVolume() const {
  return box_volume_;
}

//-------------------------------------------------------------------------------------------------
double3 GCMCSystemSampler::selectInsertionSite() {
  // Get box dimensions
  const double* box_dims = phase_space_->getBoxDimensionsHandle()->data();

  // Generate random position within the box
  // For simplicity, use orthorhombic approximation for random placement
  double3 site;
  site.x = rng_.uniformRandomNumber() * box_dims[0];  // a dimension
  site.y = rng_.uniformRandomNumber() * box_dims[1];  // b dimension
  site.z = rng_.uniformRandomNumber() * box_dims[2];  // c dimension

  return site;
}

//-------------------------------------------------------------------------------------------------
bool GCMCSystemSampler::attemptInsertion() {
  stats_.n_moves++;

  // Select random ghost molecule
  GCMCMolecule* ghost_mol = selectRandomGhostMolecule();
  if (ghost_mol == nullptr) {
    log_stream_ << "# No ghost molecules available\n";
    stats_.outcomes.push_back("rejected insert (no ghosts)");
    return false;
  }

  // Apply random rotation to the molecule before insertion
  applyRandomRotation(*ghost_mol);

  // Select insertion site anywhere in the box
  double3 insertion_site = selectInsertionSite();

  // Save current state (after rotation)
  std::vector<double3> saved_coords = saveCoordinates(*ghost_mol);
  std::vector<double3> saved_vels = saveVelocities(*ghost_mol);

  // Calculate current COG
  double3 current_cog = calculateMoleculeCOG(*ghost_mol);

  // Move molecule to insertion site
  PhaseSpaceWriter psw = phase_space_->data();
  double* xcrd = psw.xcrd;
  double* ycrd = psw.ycrd;
  double* zcrd = psw.zcrd;

  for (size_t i = 0; i < ghost_mol->atom_indices.size(); i++) {
    int atom_idx = ghost_mol->atom_indices[i];
    xcrd[atom_idx] = insertion_site.x + (saved_coords[i].x - current_cog.x);
    ycrd[atom_idx] = insertion_site.y + (saved_coords[i].y - current_cog.y);
    zcrd[atom_idx] = insertion_site.z + (saved_coords[i].z - current_cog.z);
  }

  // Note: Coordinates will be uploaded by evaluateTotalEnergy() when needed
  // Manual uploadPositions() here was causing CUDA driver state exhaustion after ~100 cycles
  // with large systems (>1000 atoms) due to excessive cudaMemcpy operations

  // Apply PBC
  applyPBC(*ghost_mol);

  // Assign Maxwell-Boltzmann velocities
  double* xvel = psw.xvel;
  double* yvel = psw.yvel;
  double* zvel = psw.zvel;

  for (int atom_idx : ghost_mol->atom_indices) {
    double mass = topology_->getAtomicMass<double>(atom_idx);
    double3 vel = generateMaxwellBoltzmannVelocity(mass);
    xvel[atom_idx] = vel.x;
    yvel[atom_idx] = vel.y;
    zvel[atom_idx] = vel.z;
  }

  // Note: Coordinates/velocities will be uploaded by evaluateTotalEnergy() when needed
  // Manual uploadPositions() here was causing CUDA driver state exhaustion

  // Calculate energy change for GCMC insertion
  // Molecules are already at correct lambdas: ACTIVE at 1.0, GHOST at 0.0
  // Only need to evaluate inserting molecule at lambda=0 vs lambda≈1

  // E_initial: Ghost at lambda=0 has no interactions (should be 0)
  // Ghost molecule is already at lambda=0, so just evaluate
  double E_initial = evaluateTotalEnergy();

  // E_final: Set inserting molecule to lambda≈1 to calculate interactions
  // Use 0.998 (< 0.999 threshold) so evaluateLambdaScaledNonbonded sees it as a "ghost"
  // and calculates its interactions with ACTIVE molecules (which are at lambda=1.0)
  adjustMoleculeLambdaAuto(this, *ghost_mol, 0.998);
  double E_final = evaluateTotalEnergy();

  // Restore to lambda=0 before acceptance decision
  adjustMoleculeLambdaAuto(this, *ghost_mol, 0.0);

  double delta_E = E_final - E_initial;

  // DEBUG: Print energies (first 10 insertions only)
  if (kGcmcDebugLogs && stats_.n_inserts < 10) {
    log_stream_ << "# DEBUG Insert " << stats_.n_inserts << ": E_init=" << E_initial
                << " E_final=" << E_final << " dE=" << delta_E << " N_active=" << N_active_ << "\n";
  }

  // If accepted, will be set to lambda=1.0 below

  // Calculate acceptance probability for insertion
  // P_acc = min(1, exp(B) * exp(-beta*delta_E) / (N+1))
  // Note: N_active_ is the current number of active molecules in the entire system
  double acc_prob = std::min(1.0, std::exp(B_ - beta_ * delta_E) / (N_active_ + 1.0));

  // Metropolis acceptance
  stats_.n_inserts++;
  stats_.insert_acceptance_probs.push_back(acc_prob);
  stats_.move_resids.push_back(ghost_mol->resid);

  if (rng_.uniformRandomNumber() < acc_prob) {
    // ACCEPT - fully activate molecule
    adjustMoleculeLambdaAuto(this, *ghost_mol, 1.0);
    if (kGcmcDebugLogs) {
      log_stream_ << "# DEBUG ACCEPT: lambda set to 1.0 for resid " << ghost_mol->resid << "\n";
    }
    ghost_mol->status = GCMCMoleculeStatus::ACTIVE;
    N_active_++;
    stats_.n_accepted++;
    stats_.n_accepted_inserts++;
    stats_.N_history.push_back(N_active_);
    stats_.acc_rate_history.push_back(stats_.getAcceptanceRate());

    logMove("ACCEPTED insertion", ghost_mol->resid, delta_E, acc_prob);
    stats_.outcomes.push_back("accepted insert");

    return true;
  } else {
    // REJECT - restore original state
    adjustMoleculeLambdaAuto(this, *ghost_mol, 0.0);
    restoreCoordinates(*ghost_mol, saved_coords);
    restoreVelocities(*ghost_mol, saved_vels, false);  // Don't reverse for standard MC

    stats_.N_history.push_back(N_active_);
    stats_.acc_rate_history.push_back(stats_.getAcceptanceRate());

    logMove("REJECTED insertion", ghost_mol->resid, delta_E, acc_prob);
    stats_.outcomes.push_back("rejected insert");

    return false;
  }
}

//-------------------------------------------------------------------------------------------------
bool GCMCSystemSampler::attemptDeletion() {
  stats_.n_moves++;

  // Select random active molecule
  GCMCMolecule* active_mol = selectRandomActiveMolecule();
  if (active_mol == nullptr) {
    log_stream_ << "# No active molecules available\n";
    stats_.outcomes.push_back("rejected delete (no active)");
    return false;
  }

  // Save current state
  std::vector<double3> saved_coords = saveCoordinates(*active_mol);
  std::vector<double3> saved_vels = saveVelocities(*active_mol);

  // Calculate energy with molecule active
  double E_initial = evaluateTotalEnergy();

  // Turn off interactions
  adjustMoleculeLambdaAuto(this, *active_mol, 0.0);

  // Calculate energy with molecule as ghost
  double E_final = evaluateTotalEnergy();

  double delta_E = E_final - E_initial;

  // Calculate acceptance probability for deletion
  // P_acc = min(1, N * exp(-B) * exp(-beta*delta_E))
  double acc_prob = std::min(1.0, N_active_ * std::exp(-B_ - beta_ * delta_E));

  // Metropolis acceptance
  stats_.n_deletes++;
  stats_.delete_acceptance_probs.push_back(acc_prob);
  stats_.move_resids.push_back(active_mol->resid);

  if (rng_.uniformRandomNumber() < acc_prob) {
    // ACCEPT deletion
    active_mol->status = GCMCMoleculeStatus::GHOST;
    N_active_--;
    stats_.n_accepted++;
    stats_.n_accepted_deletes++;
    stats_.N_history.push_back(N_active_);
    stats_.acc_rate_history.push_back(stats_.getAcceptanceRate());

    logMove("ACCEPTED deletion", active_mol->resid, delta_E, acc_prob);
    stats_.outcomes.push_back("accepted delete");

    return true;
  } else {
    // REJECT - restore active state
    adjustMoleculeLambdaAuto(this, *active_mol, 1.0);

    stats_.N_history.push_back(N_active_);
    stats_.acc_rate_history.push_back(stats_.getAcceptanceRate());

    logMove("REJECTED deletion", active_mol->resid, delta_E, acc_prob);
    stats_.outcomes.push_back("rejected delete");

    return false;
  }
}

//-------------------------------------------------------------------------------------------------
bool GCMCSystemSampler::runGCMCCycle() {
  // Decide whether to attempt insertion or deletion
  // Use 50/50 probability or bias based on current occupancy

  if (N_active_ == 0 || (getGhostCount() > 0 && rng_.uniformRandomNumber() < 0.5)) {
    return attemptInsertion();
  } else {
    return attemptDeletion();
  }
}

//-------------------------------------------------------------------------------------------------
void GCMCSystemSampler::enableAdaptiveB(int stage1_moves, int stage2_moves, int stage3_moves,
                                        double b_discovery, double target_occupancy,
                                        double coarse_rate, double fine_rate,
                                        double b_min, double b_max) {
  adaptive_b_enabled_ = true;
  stage1_moves_ = stage1_moves;
  stage2_moves_ = stage2_moves;
  stage3_moves_ = stage3_moves;
  b_discovery_ = b_discovery;
  target_occupancy_ = target_occupancy;
  coarse_learning_rate_ = coarse_rate;
  fine_learning_rate_ = fine_rate;
  b_min_ = b_min;
  b_max_ = b_max;
  n_max_fragments_ = 0;
  current_adaptive_b_ = b_discovery;  // Start with discovery B
  current_stage_ = AnnealingStage::DISCOVERY;
  move_counter_ = 0;

  // Log adaptive B settings
  log_stream_ << "# Adaptive B Protocol enabled\n";
  log_stream_ << "#   Stage 1 (Discovery): " << stage1_moves_ << " moves, B = " << b_discovery_ << "\n";
  log_stream_ << "#   Stage 2 (Coarse): " << stage2_moves_ << " moves, target = "
              << (target_occupancy_ * 100) << "% of N_max, learning rate = " << coarse_rate << "\n";
  log_stream_ << "#   Stage 3 (Fine): " << stage3_moves_ << " moves, annealing to 0, learning rate = "
              << fine_rate << "\n";
  log_stream_ << "#   B clamps: [" << b_min_ << ", " << b_max_ << "]\n";
  log_stream_.flush();
}

//-------------------------------------------------------------------------------------------------
double GCMCSystemSampler::computeAdaptiveB(int move_number) {
  if (!adaptive_b_enabled_) {
    return B_;  // Use fixed B
  }

  updateStageProgress(move_number);
  int current_count = countActiveFragments();

  switch (current_stage_) {
    case AnnealingStage::DISCOVERY:
      // Stage 1: Fixed high B, track maximum
      if (current_count > n_max_fragments_) {
        n_max_fragments_ = current_count;
      }
      current_adaptive_b_ = b_discovery_;
      break;

    case AnnealingStage::COARSE: {
      // Stage 2: Adaptive to target occupancy (e.g., 50% of N_max)
      int target = static_cast<int>(target_occupancy_ * n_max_fragments_);
      double error = (target - current_count) / static_cast<double>(std::max(target, 1));
      current_adaptive_b_ += coarse_learning_rate_ * error;
      current_adaptive_b_ = std::clamp(current_adaptive_b_, b_min_, b_max_);
      break;
    }

    case AnnealingStage::FINE: {
      // Stage 3: Gradual decrease to zero occupancy
      int moves_in_stage3 = move_number - stage1_moves_ - stage2_moves_;
      double progress = static_cast<double>(moves_in_stage3) / stage3_moves_;
      // Linear annealing of target from target_occupancy to 0
      double current_target_fraction = (1.0 - progress) * target_occupancy_;
      int target = static_cast<int>(current_target_fraction * n_max_fragments_);

      double error = (target - current_count) / static_cast<double>(std::max(target, 1));
      current_adaptive_b_ += fine_learning_rate_ * error;
      // In fine stage, only clamp to minimum, allow it to go lower naturally
      current_adaptive_b_ = std::max(b_min_, current_adaptive_b_);
      break;
    }

    case AnnealingStage::PRODUCTION:
      // Stage 4: Fixed B (maintain last value from fine stage)
      // current_adaptive_b_ stays at its last value
      break;
  }

  // Update the B_ member for use in insertion/deletion calculations
  B_ = current_adaptive_b_;

  // Log progress every 1000 moves
  move_counter_++;
  if (move_counter_ % 1000 == 0) {
    const char* stage_names[] = {"DISCOVERY", "COARSE", "FINE", "PRODUCTION"};
    int stage_idx = static_cast<int>(current_stage_);

    log_stream_ << "# Move " << move_number
                << " | Stage: " << stage_names[stage_idx]
                << " | B: " << std::fixed << std::setprecision(3) << current_adaptive_b_
                << " | Active: " << current_count << "/" << molecules_.size();

    if (current_stage_ == AnnealingStage::COARSE || current_stage_ == AnnealingStage::FINE) {
      int target = 0;
      if (current_stage_ == AnnealingStage::COARSE) {
        target = static_cast<int>(target_occupancy_ * n_max_fragments_);
      } else {
        int moves_in_stage3 = move_number - stage1_moves_ - stage2_moves_;
        double progress = static_cast<double>(moves_in_stage3) / stage3_moves_;
        double current_target_fraction = (1.0 - progress) * target_occupancy_;
        target = static_cast<int>(current_target_fraction * n_max_fragments_);
      }
      log_stream_ << " | Target: " << target;
    }

    if (current_stage_ == AnnealingStage::DISCOVERY) {
      log_stream_ << " | N_max: " << n_max_fragments_;
    }

    // Add acceptance rate from last 1000 moves if available
    if (stats_.n_moves > 0) {
      double accept_rate = static_cast<double>(stats_.n_accepted) / stats_.n_moves;
      log_stream_ << " | Accept: " << std::fixed << std::setprecision(1)
                  << (accept_rate * 100) << "%";
    }

    log_stream_ << "\n";
    log_stream_.flush();
  }

  return current_adaptive_b_;
}

//-------------------------------------------------------------------------------------------------
int GCMCSystemSampler::countActiveFragments() const {
  int count = 0;
  for (const auto& mol : molecules_) {
    // Consider molecule active if lambda > 0.5
    if (mol.getCombinedLambda() > 0.5) {
      count++;
    }
  }
  return count;
}

//-------------------------------------------------------------------------------------------------
void GCMCSystemSampler::updateStageProgress(int move_number) {
  AnnealingStage prev_stage = current_stage_;

  if (move_number < stage1_moves_) {
    current_stage_ = AnnealingStage::DISCOVERY;
  } else if (move_number < stage1_moves_ + stage2_moves_) {
    current_stage_ = AnnealingStage::COARSE;
  } else if (move_number < stage1_moves_ + stage2_moves_ + stage3_moves_) {
    current_stage_ = AnnealingStage::FINE;
  } else {
    current_stage_ = AnnealingStage::PRODUCTION;
  }

  // Log stage transitions
  if (prev_stage != current_stage_) {
    const char* stage_names[] = {"DISCOVERY", "COARSE", "FINE", "PRODUCTION"};
    int prev_idx = static_cast<int>(prev_stage);
    int curr_idx = static_cast<int>(current_stage_);

    log_stream_ << "\n# === Stage transition: " << stage_names[prev_idx]
                << " -> " << stage_names[curr_idx] << " ===\n";

    if (current_stage_ == AnnealingStage::COARSE) {
      log_stream_ << "# Discovery complete. N_max = " << n_max_fragments_ << " fragments\n";
      log_stream_ << "# Starting coarse equilibration to "
                  << (target_occupancy_ * 100) << "% occupancy ("
                  << static_cast<int>(target_occupancy_ * n_max_fragments_) << " fragments)\n";
    } else if (current_stage_ == AnnealingStage::FINE) {
      log_stream_ << "# Starting fine annealing from "
                  << static_cast<int>(target_occupancy_ * n_max_fragments_)
                  << " to 0 fragments\n";
    } else if (current_stage_ == AnnealingStage::PRODUCTION) {
      log_stream_ << "# Annealing complete. Final B = " << current_adaptive_b_ << "\n";
      log_stream_ << "# Strongest binders: " << countActiveFragments() << " fragments remain\n";
    }

    log_stream_.flush();
  }
}

//-------------------------------------------------------------------------------------------------
// NCMCSystemSampler implementation
//-------------------------------------------------------------------------------------------------
NCMCSystemSampler::NCMCSystemSampler(AtomGraph* topology,
                                     PhaseSpace* ps,
                                     StaticExclusionMask* exclusions,
                                     Thermostat* thermostat,
                                     double temperature,
                                     const GhostMoleculeMetadata& ghost_metadata,
                                     int n_pert_steps,
                                     int n_prop_steps_per_pert,
                                     double timestep,
                                     const std::vector<double>& lambdas,
                                     bool record_traj,
                                     double mu_ex,
                                     double standard_volume,
                                     double adams,
                                     double adams_shift,
                                     topology::ImplicitSolventModel gb_model,
                                     const std::string& resname,
                                     const std::string& ghost_file,
                                     const std::string& log_file) :
  GCMCSystemSampler(topology, ps, exclusions, thermostat, temperature, ghost_metadata,
                    mu_ex, standard_volume, adams, adams_shift, gb_model, resname,
                    ghost_file, log_file),
  record_traj_{record_traj}
{
  // Validate thermostat
  if (thermostat == nullptr) {
    rtErr("Cannot create NCMCSystemSampler with null thermostat", "NCMCSystemSampler");
  }

  // Setup protocol
  protocol_.setPropagationStepsPerPerturbation(n_prop_steps_per_pert);
  protocol_.setTimestep(timestep);

  // Create or validate lambda schedule
  if (lambdas.empty()) {
    protocol_.generateLinearSchedule(n_pert_steps);
  } else {
    if (static_cast<int>(lambdas.size()) != n_pert_steps + 1) {
      rtErr("Lambda schedule must have n_pert_steps + 1 points", "NCMCSystemSampler");
    }
    if (std::abs(lambdas[0]) > 1.0e-6 || std::abs(lambdas.back() - 1.0) > 1.0e-6) {
      rtErr("Lambda schedule must start at 0 and end at 1", "NCMCSystemSampler");
    }
    protocol_.setLambdaSchedule(lambdas);
  }

  // Log NCMC system-wide parameters
  log_stream_ << "# NCMC System-wide Sampler initialized\n";
  log_stream_ << "#   Total switching time: " << protocol_.getSwitchingTime() << " ps\n";
  log_stream_ << "#   Perturbation steps: " << n_pert_steps << "\n";
  log_stream_ << "#   Propagation steps per pert: " << n_prop_steps_per_pert << "\n";
  log_stream_ << "#   Timestep: " << timestep << " fs\n";
  log_stream_.flush();
}

//-------------------------------------------------------------------------------------------------
void NCMCSystemSampler::propagateSystem(int n_steps) {
  // OPTIMIZATION: Lambda is constant during propagateSystem, so generate abstracts ONCE
  // This eliminates 10,000+ abstract regenerations per GCMC move (100 pert steps × 1 prop step × many regenerations)

  static int propagate_call_count = 0;
  static double total_abstract_time = 0.0;
  static double total_loop_time = 0.0;
  static double total_download_time = 0.0;

  auto abstract_start = std::chrono::high_resolution_clock::now();
  const ValenceKit<double> vk = topology_->getDoublePrecisionValenceKit();
  const NonbondedKit<double> nbk = topology_->getDoublePrecisionNonbondedKit();
  const StaticExclusionMaskReader ser = exclusions_->data();
  const ImplicitSolventKit<double> isk = topology_->getDoublePrecisionImplicitSolventKit();
  const VirtualSiteKit<double> vsk = topology_->getDoublePrecisionVirtualSiteKit();
  const ChemicalDetailsKit cdk = topology_->getChemicalDetailsKit();
  const ConstraintKit<double> cnk = topology_->getDoublePrecisionConstraintKit();
  const restraints::RestraintApparatus empty_restraints(topology_);
  const RestraintKit<double, double2, double4> rar = empty_restraints.dpData();

  // Get lambda arrays (CPU-side pointers)
  const double* lambda_vdw_ptr = lambda_vdw_.data();
  const double* lambda_ele_ptr = lambda_ele_.data();

  // Create lambda-aware nonbonded kit
  const topology::LambdaNonbondedKit<double> lambda_nbk(
      nbk.natom, nbk.n_q_types, nbk.n_lj_types, nbk.coulomb_constant, nbk.charge,
      nbk.q_idx, nbk.lj_idx, nbk.q_parameter, nbk.lja_coeff, nbk.ljb_coeff, nbk.ljc_coeff,
      nbk.lja_14_coeff, nbk.ljb_14_coeff, nbk.ljc_14_coeff, nbk.lj_sigma, nbk.lj_14_sigma,
      nbk.nb11x, nbk.nb11_bounds, nbk.nb12x, nbk.nb12_bounds, nbk.nb13x, nbk.nb13_bounds,
      nbk.nb14x, nbk.nb14_bounds, nbk.lj_type_corr, lambda_vdw_ptr, lambda_ele_ptr);

  std::vector<double> effective_gb_radii(topology_->getAtomCount(), 0.0);
  std::vector<double> psi(topology_->getAtomCount(), 0.0);
  std::vector<double> sumdeijda(topology_->getAtomCount(), 0.0);
  generalized_born_defaults::NeckGeneralizedBornKit<double> neck_gbk(0, 0.0, 0.0, nullptr, nullptr);
  auto abstract_end = std::chrono::high_resolution_clock::now();
  total_abstract_time += std::chrono::duration<double, std::milli>(abstract_end - abstract_start).count();

  auto loop_start = std::chrono::high_resolution_clock::now();

#ifdef STORMM_USE_CUDA
  // ==========================================================================================
  // GPU-ACCELERATED DYNAMICS PATH (Periodic systems only)
  // ==========================================================================================
  // IMPORTANT: GPU propagation requires AtomGraphSynthesis which is only created
  // for periodic systems. Non-periodic systems fall back to CPU path below.

  if (topology_synthesis_ != nullptr) {
    // Get timestep from dynamics controls
  const double dt = dyn_controls_.getTimeStep();
  const int n_atoms = topology_->getAtomCount();

  // Count coupled atoms (lambda > threshold)
  int n_coupled = 0;
  std::vector<int> coupled_atom_indices;
  for (int i = 0; i < n_atoms; i++) {
    if (lambda_vdw_.data()[i] > 0.01 || lambda_ele_.data()[i] > 0.01) {
      coupled_atom_indices.push_back(i);
      n_coupled++;
    }
  }

  // Early exit if no coupled atoms
  if (n_coupled > 0) {
    // Update coupled indices array
    if (n_coupled > coupled_indices_.size()) {
      rtErr("Coupled atom list of length " + std::to_string(n_coupled) +
            " exceeds pre-allocated workspace of " + std::to_string(coupled_indices_.size()) +
            " atoms. Increase the initial allocation.", "GCMCSampler::propagateSystem");
    }
    for (int i = 0; i < n_coupled; i++) {
      coupled_indices_.data()[i] = coupled_atom_indices[i];
    }

    // Upload all necessary data to GPU
    #ifdef STORMM_USE_HPC
  #ifdef STORMM_USE_HPC

  phase_space_->uploadPositions();

  #endif
  #endif
    lambda_vdw_.upload();
    lambda_ele_.upload();
    atom_sigma_.upload();
    atom_epsilon_.upload();
    coupled_indices_.upload(0, n_coupled);
    topology_->upload();

    // Get device pointers from Hybrid arrays
    const int* d_coupled_indices = coupled_indices_.data(HybridTargetLevel::DEVICE);
    const double* d_lambda_vdw = lambda_vdw_.data(HybridTargetLevel::DEVICE);
    const double* d_lambda_ele = lambda_ele_.data(HybridTargetLevel::DEVICE);
    const double* d_atom_sigma = atom_sigma_.data(HybridTargetLevel::DEVICE);
    const double* d_atom_epsilon = atom_epsilon_.data(HybridTargetLevel::DEVICE);
    const double* d_masses = cdk.masses;  // Already points to GPU memory

    // Get exclusion mask data (already on GPU)
    const uint* d_exclusion_mask = ser.mask_data;
    const int* d_supertile_map = ser.supertile_map_idx;
    const int* d_tile_map = ser.tile_map_idx;
    const int supertile_stride = ser.supertile_stride_count;

    // Get PhaseSpace device pointers - note: STORMM stores x,y,z separately
    PhaseSpaceWriter psw = phase_space_->data();
    double* d_xcrd = psw.xcrd;  // Points to GPU memory when CUDA enabled
    double* d_ycrd = psw.ycrd;
    double* d_zcrd = psw.zcrd;
    double* d_xvel = psw.xvel;
    double* d_yvel = psw.yvel;
    double* d_zvel = psw.zvel;
    double* d_xfrc = psw.xfrc;
    double* d_yfrc = psw.yfrc;
    double* d_zfrc = psw.zfrc;

    // Get unit cell info
    const double* d_umat = psw.umat;  // Already points to GPU memory
    const UnitCellType unit_cell = psw.unit_cell;

    // Allocate temporary GPU arrays for energy outputs
    energy_output_elec_.resize(n_coupled);
    energy_output_vdw_.resize(n_coupled);
    double* d_per_atom_elec = energy_output_elec_.data(HybridTargetLevel::DEVICE);
    double* d_per_atom_vdw = energy_output_vdw_.data(HybridTargetLevel::DEVICE);

    // Create kernel launcher for GPU computation
    const card::GpuDetails gpu;  // Default constructor for GPU details
    const card::CoreKlManager launcher(gpu, *topology_synthesis_);

    // Create MM controls for timestep management and integration mode
    // Note: We now use MOVE_PARTICLES mode instead of ACCUMULATE, which enables
    // constraints, virtual sites, and thermostat integration
    // Use static to avoid memory leak from repeated allocations
    static mm::MolecularMechanicsControls mmctrl(0.0, 1);  // Will be updated below
    mmctrl.primeWorkUnitCounters(
        launcher,
        energy::EvaluateForce::YES,
        energy::EvaluateEnergy::NO,  // Will override for energy steps if needed
        mm::ClashResponse::NONE,
        synthesis::VwuGoal::MOVE_PARTICLES,  // CRITICAL: Changed from ACCUMULATE
        energy::PrecisionModel::DOUBLE,
        energy::PrecisionModel::DOUBLE,
        *topology_synthesis_);

    // Create cache resource for valence kernel thread blocks
    const energy::PrecisionModel valence_prec = energy::PrecisionModel::DOUBLE;
    const int2 vale_lp = launcher.getValenceKernelDims(
        valence_prec,
        energy::EvaluateForce::YES,
        energy::EvaluateEnergy::NO,
        numerics::AccumulationMethod::SPLIT,
        synthesis::VwuGoal::MOVE_PARTICLES,  // CRITICAL: Changed from ACCUMULATE
        mm::ClashResponse::NONE);

    // Allocate cache for thread blocks
    // CRITICAL: Use static to avoid memory leak from repeated allocations
    // CacheResource contains Hybrid<llint> arrays (cache_llint_data) that use CUDA pinned memory
    // Repeated allocation/deallocation causes cudaFreeHost failures after ~100 cycles
    static energy::CacheResource valence_cache(vale_lp.x, mm::maximum_valence_work_unit_atoms);

    // Get nonbonded launch parameters and create cache
    const int2 nonb_lp = launcher.getNonbondedKernelDims(energy::PrecisionModel::DOUBLE,
                                                        topology_synthesis_->getNonbondedWorkType(),
                                                        energy::EvaluateForce::YES,
                                                        energy::EvaluateEnergy::NO,
                                                        energy::AccumulationMethod::SPLIT,
                                                        gb_model_,
                                                        energy::ClashResponse::NONE);
    // CRITICAL: Use static to avoid memory leak from repeated allocations
    static energy::CacheResource nonb_cache(nonb_lp.x, synthesis::small_block_max_atoms);

    if (se_synthesis_ == nullptr) {
      rtErr("Static exclusion mask synthesis is not initialized for GPU propagation.",
            "NCMCSystemSampler::propagateSystem");
    }
    se_synthesis_->upload();

    // Run GPU dynamics loop with full integration (constraints + virtual sites + thermostat)
    for (int step = 0; step < n_steps; step++) {
      // Determine if this is an energy evaluation step
      const bool on_energy_step = false;  // Could be made configurable
      const energy::EvaluateEnergy eval_energy = on_energy_step ?
          energy::EvaluateEnergy::YES : energy::EvaluateEnergy::NO;

      mm::launchLambdaDynamicsStep(
          d_lambda_vdw, d_lambda_ele,
          d_coupled_indices, n_coupled,
          *topology_synthesis_,    // Topology synthesis (const reference)
          *se_synthesis_,          // Static exclusion mask synthesis (const reference)
          thermostat_,             // Thermostat (integrated automatically in MOVE_PARTICLES)
          ps_synthesis_,           // Phase space synthesis (coordinates, velocities, forces)
          &mmctrl,                 // MM controls (now required for MOVE_PARTICLES mode)
          &scorecard_,             // Score card for energy tracking
          launcher,                // Kernel launcher (const reference)
          &valence_cache,          // Cache resource for valence kernel thread blocks
          &nonb_cache,             // Cache resource for nonbonded kernel thread blocks
          eval_energy,             // Energy evaluation control
          gb_workspace_,           // GB workspace (nullptr if GB disabled)
          gb_model_);              // GB model (NONE if GB disabled)

      // Increment step counters
      mmctrl.incrementStep();
      if (thermostat_ != nullptr) {
        thermostat_->incrementStep();
      }
    }

    // Synchronize GPU to ensure all operations complete
    cudaDeviceSynchronize();
  }
  } // End if (topology_synthesis_ != nullptr) - GPU path for periodic systems

#else
  // ==========================================================================================
  // CPU-ONLY DYNAMICS PATH (fallback)
  // ==========================================================================================

  // OPTIMIZATION: Get data writers ONCE before loop to avoid repeated GPU synchronization
  ThermostatWriter<double> tstw = thermostat_->dpData();
  PhaseSpaceWriter psw = phase_space_->data();

  for (int step = 0; step < n_steps; step++) {
    lambdaDynaStep(&psw, &scorecard_, tstw, vk, lambda_nbk, isk, neck_gbk,
                   effective_gb_radii.data(), psi.data(), sumdeijda.data(),
                   rar, vsk, cdk, cnk, ser, dyn_controls_, 0,
                   0.75, 0.5);  // vdw_coupling_threshold=0.75, softcore_alpha=0.5
  }
#endif

  auto loop_end = std::chrono::high_resolution_clock::now();
  total_loop_time += std::chrono::duration<double, std::milli>(loop_end - loop_start).count();

  // OPTIMIZATION: Don't download after every propagate - batch downloads before energy evals
  auto download_start = std::chrono::high_resolution_clock::now();
  #ifdef STORMM_USE_CUDA
  // phase_space_->download();  // DISABLED - batch downloads to reduce overhead
  #endif
  auto download_end = std::chrono::high_resolution_clock::now();
  total_download_time += std::chrono::duration<double, std::milli>(download_end - download_start).count();

  propagate_call_count++;
  if (propagate_call_count % 100 == 0) {
    log_stream_ << "# PROPAGATE PROFILE (100 calls): Abstract=" << (total_abstract_time/100.0)
                << "ms, Loop=" << (total_loop_time/100.0)
                << "ms, Download=" << (total_download_time/100.0)
                << "ms, Total=" << ((total_abstract_time + total_loop_time + total_download_time)/100.0) << "ms\n";
    log_stream_.flush();
    total_abstract_time = 0.0;
    total_loop_time = 0.0;
    total_download_time = 0.0;
  }

  // Invalidate energy cache since coordinates changed during MD
  invalidateEnergyCache();
}

//-------------------------------------------------------------------------------------------------
double NCMCSystemSampler::performNCMCProtocol(GCMCMolecule& mol, bool forward, bool propagate) {
  // Profiling timers
  static double total_energy_time = 0.0;
  static double total_md_time = 0.0;
  static double total_lambda_time = 0.0;
  static int ncmc_call_count = 0;
  auto ncmc_start = std::chrono::high_resolution_clock::now();

  double work = 0.0;
  const auto& lambdas = protocol_.getLambdaSchedule();
  const int n_pert_steps = protocol_.getPerturbationSteps();
  const int n_prop_steps = protocol_.getPropagationStepsPerPerturbation();

  // Determine lambda schedule direction
  std::vector<double> lambda_values;
  if (forward) {
    // Insertion: 0 -> 1
    lambda_values = lambdas;
  } else {
    // Deletion: 1 -> 0
    lambda_values.reserve(lambdas.size());
    for (auto it = lambdas.rbegin(); it != lambdas.rend(); ++it) {
      lambda_values.push_back(*it);
    }
  }

  // Initial equilibration at starting lambda
  if (propagate && n_prop_steps > 0) {
    propagateSystem(n_prop_steps);
    // NOTE: PBC wrapping removed for performance - energy calculations already use PBC correctly
    // applyPBCToAllMolecules();
  }

  // Handle instant insertion/deletion (npert=0 case)
  if (n_pert_steps == 0) {
    // Instantaneous lambda change: evaluate energy difference in one step
    double E_before = evaluateTotalEnergy();
    adjustMoleculeLambdaGPU(mol, lambda_values[1]);  // Jump directly to final lambda (GPU-optimized)
    double E_after = evaluateTotalEnergy();

    work = E_after - E_before;

    if (std::isnan(E_after) || std::isnan(work)) {
      log_stream_ << "# NaN detected during instant insertion/deletion\n";
      stats_.n_explosions++;
      return std::numeric_limits<double>::infinity();
    }

    return work;
  }

  // NCMC switching protocol with work accumulation
  for (int i = 0; i < n_pert_steps; i++) {
    // Energy BEFORE lambda change
    auto e1_start = std::chrono::high_resolution_clock::now();
    double E_before = evaluateTotalEnergy();
    auto e1_end = std::chrono::high_resolution_clock::now();
    total_energy_time += std::chrono::duration<double, std::milli>(e1_end - e1_start).count();

    // Change lambda
    auto lambda_start = std::chrono::high_resolution_clock::now();
    adjustMoleculeLambda(mol, lambda_values[i + 1]);
    auto lambda_end = std::chrono::high_resolution_clock::now();
    total_lambda_time += std::chrono::duration<double, std::milli>(lambda_end - lambda_start).count();

    // Energy AFTER lambda change at SAME coordinates
    auto e2_start = std::chrono::high_resolution_clock::now();
    double E_after = evaluateTotalEnergy();
    auto e2_end = std::chrono::high_resolution_clock::now();
    total_energy_time += std::chrono::duration<double, std::milli>(e2_end - e2_start).count();

    // Accumulate work
    double delta_E = E_after - E_before;
    work += delta_E;

    // Check for NaN
    if (std::isnan(E_after) || std::isnan(work)) {
      log_stream_ << "# NaN detected during NCMC protocol\n";
      stats_.n_explosions++;
      return std::numeric_limits<double>::infinity();
    }

    // Propagate at new lambda
    if (propagate && n_prop_steps > 0) {
      try {
        auto md_start = std::chrono::high_resolution_clock::now();
        propagateSystem(n_prop_steps);
        auto md_end = std::chrono::high_resolution_clock::now();
        total_md_time += std::chrono::duration<double, std::milli>(md_end - md_start).count();
        // NOTE: PBC wrapping removed for performance
        // applyPBCToAllMolecules();
      } catch (...) {
        log_stream_ << "# Integration failure during NCMC protocol\n";
        stats_.n_explosions++;
        return std::numeric_limits<double>::infinity();
      }
    }
  }

  // Print profiling every 10 NCMC calls
  ncmc_call_count++;
  if (ncmc_call_count % 10 == 0) {
    auto ncmc_end = std::chrono::high_resolution_clock::now();
    double ncmc_total_time = std::chrono::duration<double, std::milli>(ncmc_end - ncmc_start).count();
    log_stream_ << "# PROFILE (10 NCMCs): Energy=" << (total_energy_time/10.0)
                << "ms, MD=" << (total_md_time/10.0)
                << "ms, Lambda=" << (total_lambda_time/10.0)
                << "ms, Total=" << ncmc_total_time << "ms\n";
    log_stream_.flush();
    total_energy_time = 0.0;
    total_md_time = 0.0;
    total_lambda_time = 0.0;
  }

  return work;
}

//-------------------------------------------------------------------------------------------------
bool NCMCSystemSampler::attemptInsertion() {
  stats_.n_moves++;

  // Select random ghost molecule
  GCMCMolecule* ghost_mol = selectRandomGhostMolecule();
  if (ghost_mol == nullptr) {
    log_stream_ << "# No ghost molecules available\n";
    stats_.outcomes.push_back("rejected NCMC insert (no ghosts)");
    return false;
  }

  // Apply random rotation to the molecule before insertion
  applyRandomRotation(*ghost_mol);

  // Select insertion site anywhere in the box
  double3 insertion_site = selectInsertionSite();

  // Save current state (after rotation)
  std::vector<double3> saved_coords = saveCoordinates(*ghost_mol);
  std::vector<double3> saved_vels = saveVelocities(*ghost_mol);

  // Move molecule to insertion site
  double3 current_cog = calculateMoleculeCOG(*ghost_mol);

  PhaseSpaceWriter psw = phase_space_->data();
  double* xcrd = psw.xcrd;
  double* ycrd = psw.ycrd;
  double* zcrd = psw.zcrd;

  for (size_t i = 0; i < ghost_mol->atom_indices.size(); i++) {
    int atom_idx = ghost_mol->atom_indices[i];
    xcrd[atom_idx] = insertion_site.x + (saved_coords[i].x - current_cog.x);
    ycrd[atom_idx] = insertion_site.y + (saved_coords[i].y - current_cog.y);
    zcrd[atom_idx] = insertion_site.z + (saved_coords[i].z - current_cog.z);
  }

  // Upload modified coordinates to GPU before recalculating COG
  #ifdef STORMM_USE_HPC
  #ifdef STORMM_USE_HPC

  phase_space_->uploadPositions();

  #endif
  #endif

  applyPBC(*ghost_mol);

  // Assign Maxwell-Boltzmann velocities
  double* xvel = psw.xvel;
  double* yvel = psw.yvel;
  double* zvel = psw.zvel;

  for (int atom_idx : ghost_mol->atom_indices) {
    double mass = topology_->getAtomicMass<double>(atom_idx);
    double3 vel = generateMaxwellBoltzmannVelocity(mass);
    xvel[atom_idx] = vel.x;
    yvel[atom_idx] = vel.y;
    zvel[atom_idx] = vel.z;
  }

  // Note: Coordinates/velocities will be uploaded by evaluateTotalEnergy() when needed
  // Manual uploadPositions() here was causing CUDA driver state exhaustion

  // Set initial lambda = 0 (ghost state)
  adjustMoleculeLambdaAuto(this, *ghost_mol, 0.0);

  // Save thermostat state for detailed balance
  Thermostat thermostat_backup = *thermostat_;

  // Run NCMC protocol (0 -> 1)
  double protocol_work = performNCMCProtocol(*ghost_mol, true);

  // Check for protocol failure
  if (std::isinf(protocol_work)) {
    *thermostat_ = thermostat_backup;
    adjustMoleculeLambdaAuto(this, *ghost_mol, 0.0);
    restoreCoordinates(*ghost_mol, saved_coords);
    restoreVelocities(*ghost_mol, saved_vels, true);
    stats_.outcomes.push_back("rejected NCMC insert (protocol failed)");
    return false;
  }

  #ifdef STORMM_USE_CUDA
  phase_space_->download();
  #endif

  // NCMC insertion acceptance probability
  // P_accept = min(1, exp(B - βW) / (N + 1))
  double acc_prob = std::min(1.0, std::exp(B_ - beta_ * protocol_work) / (N_active_ + 1.0));

  // Metropolis acceptance
  stats_.n_inserts++;
  stats_.insert_works.push_back(protocol_work);
  stats_.insert_acceptance_probs.push_back(acc_prob);
  stats_.move_resids.push_back(ghost_mol->resid);

  if (rng_.uniformRandomNumber() < acc_prob) {
    // ACCEPT - fully activate molecule
    if (kGcmcDebugLogs) {
      log_stream_ << "# DEBUG BEFORE: resid=" << ghost_mol->resid << " lambda_vdw=" << ghost_mol->lambda_vdw << " lambda_ele=" << ghost_mol->lambda_ele << "\n";
    }
    adjustMoleculeLambdaAuto(this, *ghost_mol, 1.0);
    if (kGcmcDebugLogs) {
      log_stream_ << "# DEBUG AFTER:  resid=" << ghost_mol->resid << " lambda_vdw=" << ghost_mol->lambda_vdw << " lambda_ele=" << ghost_mol->lambda_ele << "\n";
    }
    ghost_mol->status = GCMCMoleculeStatus::ACTIVE;
    N_active_++;
    stats_.n_accepted++;
    stats_.n_accepted_inserts++;
    stats_.accepted_insert_works.push_back(protocol_work);
    stats_.N_history.push_back(N_active_);
    stats_.acc_rate_history.push_back(stats_.getAcceptanceRate());

    logMove("ACCEPTED NCMC insertion", ghost_mol->resid, protocol_work, acc_prob);
    stats_.outcomes.push_back("accepted NCMC insert");

    return true;
  } else {
    // REJECT - restore state with velocity reversal
    *thermostat_ = thermostat_backup;
    adjustMoleculeLambdaAuto(this, *ghost_mol, 0.0);
    restoreCoordinates(*ghost_mol, saved_coords);
    restoreVelocities(*ghost_mol, saved_vels, true);

    stats_.N_history.push_back(N_active_);
    stats_.acc_rate_history.push_back(stats_.getAcceptanceRate());

    logMove("REJECTED NCMC insertion", ghost_mol->resid, protocol_work, acc_prob);
    stats_.outcomes.push_back("rejected NCMC insert");

    return false;
  }
}

//-------------------------------------------------------------------------------------------------
bool NCMCSystemSampler::attemptDeletion() {
  stats_.n_moves++;

  // Select random active molecule
  GCMCMolecule* active_mol = selectRandomActiveMolecule();
  if (active_mol == nullptr) {
    log_stream_ << "# No active molecules available\n";
    stats_.outcomes.push_back("rejected NCMC delete (no active)");
    return false;
  }

  // Save current state
  std::vector<double3> saved_coords = saveCoordinates(*active_mol);
  std::vector<double3> saved_vels = saveVelocities(*active_mol);

  // Molecule starts at lambda = 1 (active)
  adjustMoleculeLambdaAuto(this, *active_mol, 1.0);

  // Save thermostat state for detailed balance
  Thermostat thermostat_backup = *thermostat_;

  // Run NCMC protocol (1 -> 0)
  double protocol_work = performNCMCProtocol(*active_mol, false);

  // Check for protocol failure
  if (std::isinf(protocol_work)) {
    *thermostat_ = thermostat_backup;
    adjustMoleculeLambdaAuto(this, *active_mol, 1.0);
    stats_.outcomes.push_back("rejected NCMC delete (protocol failed)");
    return false;
  }

  #ifdef STORMM_USE_CUDA
  phase_space_->download();
  #endif

  // NCMC deletion acceptance probability
  // P_accept = min(1, N * exp(-B - βW))
  double acc_prob = std::min(1.0, N_active_ * std::exp(-B_ - beta_ * protocol_work));

  // Metropolis acceptance
  stats_.n_deletes++;
  stats_.delete_works.push_back(protocol_work);
  stats_.delete_acceptance_probs.push_back(acc_prob);
  stats_.move_resids.push_back(active_mol->resid);

  if (rng_.uniformRandomNumber() < acc_prob) {
    // ACCEPT deletion
    active_mol->status = GCMCMoleculeStatus::GHOST;
    N_active_--;
    stats_.n_accepted++;
    stats_.n_accepted_deletes++;
    stats_.accepted_delete_works.push_back(protocol_work);
    stats_.N_history.push_back(N_active_);
    stats_.acc_rate_history.push_back(stats_.getAcceptanceRate());

    logMove("ACCEPTED NCMC deletion", active_mol->resid, protocol_work, acc_prob);
    stats_.outcomes.push_back("accepted NCMC delete");

    return true;
  } else {
    // REJECT - restore active state with velocity reversal
    *thermostat_ = thermostat_backup;
    adjustMoleculeLambdaAuto(this, *active_mol, 1.0);
    restoreVelocities(*active_mol, saved_vels, true);

    stats_.N_history.push_back(N_active_);
    stats_.acc_rate_history.push_back(stats_.getAcceptanceRate());

    logMove("REJECTED NCMC deletion", active_mol->resid, protocol_work, acc_prob);
    stats_.outcomes.push_back("rejected NCMC delete");

    return false;
  }
}

//-------------------------------------------------------------------------------------------------
bool NCMCSystemSampler::runGCMCCycle() {
  // Decide whether to attempt insertion or deletion
  // Use 50/50 probability or bias based on current occupancy

  if (N_active_ == 0 || (getGhostCount() > 0 && rng_.uniformRandomNumber() < 0.5)) {
    return attemptInsertion();
  } else {
    return attemptDeletion();
  }
}

//-------------------------------------------------------------------------------------------------
const NCMCProtocol& NCMCSystemSampler::getProtocol() const {
  return protocol_;
}

//-------------------------------------------------------------------------------------------------
void GCMCSystemSampler::runHybridSimulation(int total_md_steps,
                                            int move_frequency,
                                            double gcmc_probability) {

  if (thermostat_ == nullptr) {
    rtErr("Cannot run hybrid simulation without thermostat", "runHybridSimulation");
  }

  if (move_frequency <= 0) {
    rtErr("Move frequency must be positive", "runHybridSimulation");
  }

  if (gcmc_probability < 0.0 || gcmc_probability > 1.0) {
    rtErr("GCMC probability must be between 0 and 1", "runHybridSimulation");
  }

  // Check if system has any active molecules
  if (getActiveCount() == 0) {
    log_stream_ << "# Hybrid simulation starting with zero active molecules.\n";
    log_stream_ << "# MD propagation will be skipped until first successful GCMC insertion.\n";
    log_stream_.flush();
  }

  log_stream_ << "# Starting hybrid MD/MC simulation\n";
  log_stream_ << "#   Total MD steps: " << total_md_steps << "\n";
  log_stream_ << "#   Move frequency: every " << move_frequency << " MD steps\n";
  log_stream_ << "#   GCMC probability: " << gcmc_probability << "\n";
  log_stream_ << "#   MC probability: " << (1.0 - gcmc_probability) << "\n";
  log_stream_ << "#   Initial active molecules: " << getActiveCount() << "\n";
  log_stream_.flush();

  int n_move_attempts = 0;
  int n_gcmc_attempts = 0;
  int n_mc_attempts = 0;

  for (int md_step = 0; md_step < total_md_steps; md_step++) {

    // ============================================================
    // Continuous lambda-aware MD (no lambda changes)
    // ============================================================
    // Skip MD propagation if no active molecules present
    if (getActiveCount() > 0) {
      propagateSystem(1);  // Single MD step with current lambda values
    }

    // ============================================================
    // Periodic random move attempts
    // ============================================================
    if (md_step > 0 && md_step % move_frequency == 0) {
      n_move_attempts++;

      // Randomly select move type
      const double rand = rng_.uniformRandomNumber();

      if (rand < gcmc_probability) {
        // --------------------------------------------------------
        // GCMC Move (NCMC insertion or deletion)
        // --------------------------------------------------------
        // Note: runGCMCCycle() includes NCMC protocol which:
        // - Changes lambda from 0→1 (insertion) or 1→0 (deletion)
        // - Propagates system during lambda changes
        // - Accumulates work and applies Metropolis criterion

        n_gcmc_attempts++;
        const bool accepted = runGCMCCycle();

        if (accepted) {
          log_stream_ << "# MD step " << md_step << ": GCMC move ACCEPTED\n";
        } else {
          log_stream_ << "# MD step " << md_step << ": GCMC move REJECTED\n";
        }

      } else {
        // --------------------------------------------------------
        // MC Move (instant translation/rotation/torsion)
        // --------------------------------------------------------
        // Note: MC moves do NOT propagate system or change lambda
        // They just modify coordinates and apply Metropolis criterion

        n_mc_attempts++;
        const int n_accepted = attemptMCMovesOnAllMolecules();

        log_stream_ << "# MD step " << md_step << ": MC moves attempted, "
                    << n_accepted << " accepted\n";
      }

      log_stream_.flush();

      // Periodic statistics output
      if (n_move_attempts % 100 == 0) {
        log_stream_ << "# ===== Hybrid Simulation Statistics (move attempt "
                    << n_move_attempts << ") =====\n";
        log_stream_ << "#   MD steps completed: " << md_step << " / " << total_md_steps << "\n";
        log_stream_ << "#   GCMC attempts: " << n_gcmc_attempts << "\n";
        log_stream_ << "#   MC attempts: " << n_mc_attempts << "\n";
        log_stream_ << "#   Current active molecules: " << getActiveCount() << "\n";

        // GCMC statistics
        const GCMCStatistics& gcmc_stats = getStatistics();
        log_stream_ << "#   GCMC acceptance: "
                    << (gcmc_stats.n_moves > 0 ?
                        100.0 * gcmc_stats.n_accepted / gcmc_stats.n_moves : 0.0)
                    << "%\n";

        // MC statistics
        auto mc_stats = getMCStatistics();
        for (const auto& stat : mc_stats) {
          log_stream_ << "#   " << stat.first << " acceptance: "
                      << (stat.second.n_attempted > 0 ?
                          100.0 * stat.second.n_accepted / stat.second.n_attempted : 0.0)
                      << "%\n";
        }

        log_stream_.flush();
      }
    }

    // Periodic snapshot output
    if (md_step % 1000 == 0) {
      writeGhostSnapshot();
    }
  }

  // Final statistics
  log_stream_ << "# ===== Final Hybrid Simulation Statistics =====\n";
  log_stream_ << "#   Total MD steps: " << total_md_steps << "\n";
  log_stream_ << "#   Total move attempts: " << n_move_attempts << "\n";
  log_stream_ << "#   GCMC attempts: " << n_gcmc_attempts << "\n";
  log_stream_ << "#   MC attempts: " << n_mc_attempts << "\n";
  log_stream_ << "#   Final active molecules: " << getActiveCount() << "\n";
  log_stream_.flush();
}

} // namespace sampling
} // namespace stormm
