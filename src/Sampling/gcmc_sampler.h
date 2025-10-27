// -*-c++-*-
#ifndef STORMM_GCMC_SAMPLER_H
#define STORMM_GCMC_SAMPLER_H

#include <fstream>
#include <memory>
#include <string>
#include <vector>
#include "copyright.h"
#include "Accelerator/hybrid.h"
#include "Constants/symbol_values.h"
#include "MolecularMechanics/dynamics.h"
#include "Namelists/nml_dynamics.h"
#include "Potential/cellgrid.h"
#include "Potential/energy_abstracts.h"
#include "Potential/pmigrid.h"
#include "Potential/scorecard.h"
#include "Potential/static_exclusionmask.h"
#include "Random/random.h"
#include "Synthesis/atomgraph_synthesis.h"
#include "Synthesis/phasespace_synthesis.h"
#include "Synthesis/implicit_solvent_workspace.h"
#include "Topology/atomgraph.h"
#include "Topology/atomgraph_enumerators.h"
#include "Trajectory/phasespace.h"
#include "Trajectory/thermostat.h"
#include "gcmc_molecule.h"
#include "gcmc_sphere.h"
#include "mc_mover.h"
#include "ncmc_protocol.h"
#include "MolecularMechanics/mm_controls.h"
#include "Accelerator/core_kernel_manager.h"
#include "Potential/cacheresource.h"
#include "Synthesis/static_mask_synthesis.h"

namespace stormm {
namespace sampling {

/// \brief Enum for the three-stage adaptive B annealing protocol
enum class AnnealingStage {
  DISCOVERY,      ///< Stage 1: Find N_max with high fixed B
  COARSE,         ///< Stage 2: Reach target occupancy with coarse adaptive control
  FINE,           ///< Stage 3: Anneal to zero with fine adaptive control
  PRODUCTION      ///< Stage 4: Fixed B (if requested after annealing)
};

using card::Hybrid;
using card::CoreKlManager;
using card::HybridTargetLevel;
using energy::CacheResource;
using energy::CellGrid;
using energy::EvaluateForce;
using energy::NonbondedTheme;
using energy::PMIGrid;
using energy::ScoreCard;
using energy::StaticExclusionMask;
using energy::StaticExclusionMaskReader;
using mm::MolecularMechanicsControls;
using namelist::DynamicsControls;
using synthesis::StaticExclusionMaskSynthesis;
using random::Xoshiro256ppGenerator;
using symbols::boltzmann_constant_gafs;
using symbols::avogadro_number;
using synthesis::AtomGraphSynthesis;
using synthesis::PhaseSpaceSynthesis;
using topology::AtomGraph;
using topology::ChemicalDetailsKit;
using topology::ConstraintKit;
using topology::GhostMoleculeMetadata;
using topology::ImplicitSolventKit;
using topology::NonbondedKit;
using topology::ValenceKit;
using topology::VirtualSiteKit;
using trajectory::CoordinateSeries;
using trajectory::PhaseSpace;
using trajectory::PhaseSpaceWriter;
using trajectory::Thermostat;
using trajectory::ThermostatWriter;

/// \brief Base class for Grand Canonical Monte Carlo sampling
///
/// This class manages GCMC moves for molecules that can transition between ghost
/// (non-interacting) and active states. It handles the lambda scaling of interactions,
/// energy evaluation, and move statistics.
class GCMCSampler {
public:

  /// \brief Constructor for GCMC sampler
  ///
  /// \param topology        Pointer to AtomGraph (must include pre-allocated ghosts)
  /// \param ps              Pointer to PhaseSpace
  /// \param exclusions      Pointer to StaticExclusionMask (built once, includes all atoms)
  /// \param thermostat      Pointer to Thermostat for MD (optional, can be nullptr)
  /// \param temperature     Simulation temperature (K)
  /// \param ghost_metadata  Metadata about ghost molecules in the topology
  /// \param gb_model        Implicit solvent model to use (default NONE = disabled)
  /// \param resname         Residue name of GCMC molecules (default "HOH")
  /// \param ghost_file      File to write ghost molecule IDs
  /// \param log_file        Log file for move tracking
  GCMCSampler(AtomGraph* topology,
              PhaseSpace* ps,
              StaticExclusionMask* exclusions,
              Thermostat* thermostat,
              double temperature,
              const GhostMoleculeMetadata& ghost_metadata,
              topology::ImplicitSolventModel gb_model = topology::ImplicitSolventModel::NONE,
              const std::string& resname = "HOH",
              const std::string& ghost_file = "gcmc-ghosts.txt",
              const std::string& log_file = "gcmc.log");

  /// \brief Destructor
  virtual ~GCMCSampler();

  /// \brief Get the number of active molecules
  ///
  /// \return Number of molecules with status ACTIVE
  int getActiveCount() const;

  /// \brief Get the number of ghost molecules
  ///
  /// \return Number of molecules with status GHOST
  int getGhostCount() const;

  /// \brief Get the total number of tracked molecules
  ///
  /// \return Total number of GCMC-controlled molecules
  int getTotalMoleculeCount() const;

  /// \brief Get the current statistics
  ///
  /// \return Reference to the statistics object
  const GCMCStatistics& getStatistics() const;

  /// \brief Get atom indices for all active molecules
  ///
  /// \return Vector of atom indices for all ACTIVE molecules
  std::vector<int> getActiveAtomIndices() const;

  /// \brief Write current ghost molecule IDs to file
  void writeGhostSnapshot();

  /// \brief Write detailed move information to log
  ///
  /// \param move_type    Type of move ("ACCEPTED NCMC insertion", etc.)
  /// \param resid        Residue ID involved in move
  /// \param work         Protocol work (kcal/mol)
  /// \param accept_prob  Acceptance probability
  void logMove(const std::string& move_type, int resid, double work, double accept_prob);

  /// \brief Run MD propagation for n_steps
  ///
  /// Propagates the system using dynaStep() with the thermostat.
  /// Requires thermostat to be set (not nullptr).
  ///
  /// \param n_steps Number of MD steps to run
  void propagateSystem(int n_steps);

  /// \brief Register a Monte Carlo mover
  ///
  /// Adds a new MC mover to the collection of available moves.
  ///
  /// \param mover  Unique pointer to the mover to register
  void registerMCMover(std::unique_ptr<MCMover> mover);

  /// \brief Enable translation moves with specified maximum displacement
  ///
  /// \param max_displacement  Maximum displacement in Angstroms
  void enableTranslationMoves(double max_displacement);

  /// \brief Enable rotation moves with specified maximum angle
  ///
  /// \param max_angle  Maximum rotation angle in degrees
  void enableRotationMoves(double max_angle);

  /// \brief Enable torsion moves with specified maximum angle
  ///
  /// \param max_angle  Maximum torsion angle change in degrees
  void enableTorsionMoves(double max_angle);

  /// \brief Attempt a Monte Carlo move on a molecule
  ///
  /// Randomly selects one of the registered movers and attempts a move.
  ///
  /// \param mol  Molecule to apply the move to
  /// \return True if any move was accepted
  bool attemptMCMove(GCMCMolecule& mol);

  /// \brief Attempt MC moves on all active molecules
  ///
  /// \return Number of accepted moves
  int attemptMCMovesOnAllMolecules();

  /// \brief Get statistics for all MC movers
  ///
  /// \return Vector of move type names and their statistics
  std::vector<std::pair<std::string, MCMoveStatistics>> getMCStatistics() const;

  /// \brief Get the PhaseSpace pointer (needed by MC movers)
  ///
  /// \return Pointer to the phase space
  PhaseSpace* getPhaseSpace() const { return phase_space_; }
  PhaseSpace exportCurrentPhaseSpace(HybridTargetLevel tier = HybridTargetLevel::HOST) const;

  /// \brief Get the AtomGraph pointer (needed by MC movers)
  ///
  /// \return Pointer to the topology
  AtomGraph* getTopology() const { return topology_; }

  /// \brief Evaluate total potential energy with lambda scaling
  ///
  /// Uses custom lambda-scaled nonbonded evaluation for ghost molecules.
  /// Made public for MC mover access.
  ///
  /// \return Total potential energy (kcal/mol)
  virtual double evaluateTotalEnergy();

  /// \brief Invalidate energy cache
  ///
  /// Must be called after any coordinate-modifying operation (MD, MC moves, etc.)
  /// to ensure cached energies are not reused with different system state.
  /// Made public for MC mover access.
  void invalidateEnergyCache();

  /// \brief Calculate center of geometry for a molecule
  ///
  /// Uses heavy atoms only if available, otherwise all atoms.
  /// Made public for MC mover access.
  ///
  /// \param mol  Molecule to calculate COG for
  /// \return Center of geometry in Cartesian coordinates
  double3 calculateMoleculeCOG(const GCMCMolecule& mol) const;

  /// \brief Save current coordinates for a molecule
  ///
  /// Made public for MC mover access.
  ///
  /// \param mol  Molecule to save coordinates for
  /// \return Vector of atom positions
  std::vector<double3> saveCoordinates(const GCMCMolecule& mol) const;

  /// \brief Restore coordinates for a molecule
  ///
  /// Made public for MC mover access.
  ///
  /// \param mol         Molecule to restore
  /// \param saved_coords Saved coordinates to restore
  void restoreCoordinates(GCMCMolecule& mol, const std::vector<double3>& saved_coords);

  /// \brief Apply periodic boundary conditions to atom positions
  ///
  /// Made public for MC mover access.
  ///
  /// \param mol  Molecule to apply PBC to
  void applyPBC(const GCMCMolecule& mol);

  /// \brief Adjust lambda parameters for a specific molecule (GPU path)
  ///
  /// Updates lambda values directly on GPU, eliminating CPU→GPU uploads.
  /// Uses CUDA kernels to modify per-atom lambda arrays in device memory.
  /// For two-stage coupling: VDW first (0-0.75), then electrostatics (0.75-1.0).
  ///
  /// This is the optimized path that eliminates 600+ memory uploads per 100 cycles.
  /// Falls back to CPU path if CUDA not available or launcher_ is nullptr.
  /// Made public for helper function access.
  ///
  /// \param mol        Molecule to adjust
  /// \param new_lambda New lambda value [0, 1]
  void adjustMoleculeLambdaGPU(GCMCMolecule& mol, double new_lambda);

  /// \brief Get GPU workspace for MC move atom indices
  ///
  /// \return Reference to Hybrid array for atom indices (pre-allocate before use)
  Hybrid<int>& getMCAtomIndices() { return mc_atom_indices_; }

  /// \brief Get GPU workspace for MC move coordinate backup (X)
  ///
  /// \return Reference to Hybrid array for X coordinate backup
  Hybrid<double>& getMCSavedX() { return mc_saved_x_; }

  /// \brief Get GPU workspace for MC move coordinate backup (Y)
  ///
  /// \return Reference to Hybrid array for Y coordinate backup
  Hybrid<double>& getMCSavedY() { return mc_saved_y_; }

  /// \brief Get GPU workspace for MC move coordinate backup (Z)
  ///
  /// \return Reference to Hybrid array for Z coordinate backup
  Hybrid<double>& getMCSavedZ() { return mc_saved_z_; }

  /// \brief Get GPU workspace for rotation matrix
  ///
  /// \return Reference to Hybrid array for 3x3 rotation matrix (size=9)
  Hybrid<double>& getMCRotationMatrix() { return mc_rotation_matrix_; }

  /// \brief Get GPU workspace for torsion rotating atoms
  ///
  /// \return Reference to Hybrid array for rotating atom indices
  Hybrid<int>& getMCRotatingAtoms() { return mc_rotating_atoms_; }

protected:

  /// \brief Adjust lambda parameters for a specific molecule (CPU path)
  ///
  /// Updates both the molecule's lambda values and the AtomGraph charge array.
  /// For two-stage coupling: VDW first (0-0.75), then electrostatics (0.75-1.0).
  ///
  /// \param mol        Molecule to adjust
  /// \param new_lambda New lambda value [0, 1]
  void adjustMoleculeLambda(GCMCMolecule& mol, double new_lambda);

  /// \brief Ensure the coupled atom index list matches current lambda state
  ///
  /// Rebuilds the coupled index list if lambda values changed and returns the
  /// number of coupled atoms that exceed the active threshold.
  ///
  /// \param download_to_host  If true, download updated indices to host memory
  /// \return Number of coupled atoms
  int ensureCoupledAtomList(bool download_to_host = false);

  /// \brief Select a random ghost molecule for insertion
  ///
  /// \return Pointer to selected molecule, or nullptr if none available
  GCMCMolecule* selectRandomGhostMolecule();

  /// \brief Select a random active molecule for deletion
  ///
  /// \return Pointer to selected molecule, or nullptr if none available
  GCMCMolecule* selectRandomActiveMolecule();

  /// \brief Apply random rotation to a molecule about its center of geometry
  ///
  /// Generates a uniform random rotation using quaternions and applies it
  /// to all atoms in the molecule about its COG.
  ///
  /// \param mol  Molecule to rotate
  void applyRandomRotation(const GCMCMolecule& mol);

  /// \brief Save current velocities for a molecule
  ///
  /// \param mol  Molecule to save velocities for
  /// \return Vector of atom velocities
  std::vector<double3> saveVelocities(const GCMCMolecule& mol) const;

  /// \brief Restore velocities for a molecule (with reversal for detailed balance)
  ///
  /// \param mol          Molecule to restore
  /// \param saved_vels   Saved velocities to restore
  /// \param reverse      If true, reverse velocities for detailed balance
  void restoreVelocities(GCMCMolecule& mol, const std::vector<double3>& saved_vels,
                         bool reverse = true);

  /// \brief Generate Maxwell-Boltzmann velocity for an atom
  ///
  /// \param mass  Atom mass (amu)
  /// \return Velocity vector (Angstroms/fs)
  double3 generateMaxwellBoltzmannVelocity(double mass);

  /// \brief Apply periodic boundary conditions to ALL coupled molecules
  ///
  /// Wraps all molecules with lambda > 0.01 back into the primary unit cell.
  /// Should be called after MD propagation to prevent coordinate drift.
  void applyPBCToAllMolecules();

  // Member variables
  AtomGraph* topology_;                  ///< Pointer to topology (not owned)
  PhaseSpace* phase_space_;              ///< Pointer to phase space (not owned)
  StaticExclusionMask* exclusions_;      ///< Pointer to exclusion mask (not owned)
  Thermostat* thermostat_;               ///< Pointer to thermostat for MD (not owned, can be nullptr)

  double temperature_;                   ///< Temperature (K)
  double kT_;                            ///< Thermal energy (kcal/mol)
  double beta_;                          ///< Inverse thermal energy (mol/kcal)
  std::string resname_;                  ///< Residue name being tracked

  std::vector<GCMCMolecule> molecules_;  ///< All GCMC-controlled molecules
  int N_active_;                         ///< Current number of active molecules

  ScoreCard scorecard_;                  ///< Energy tracking
  GCMCStatistics stats_;                 ///< Move statistics
  Xoshiro256ppGenerator rng_;            ///< Random number generator

  /// \brief Collection of registered Monte Carlo movers
  std::vector<std::unique_ptr<MCMover>> mc_movers_;

  /// \brief PME infrastructure for periodic electrostatics
  ///
  /// For periodic systems, these objects enable Particle Mesh Ewald (PME) evaluation:
  /// - topology_synthesis_: Wraps single topology for multi-system PME interface
  /// - ps_synthesis_: Wraps single phase space for multi-system PME interface
  /// - cell_grid_: Spatial decomposition for mesh mapping
  /// - pme_grid_: PME reciprocal space grid
  ///
  /// All nullptr for non-periodic systems (cutoff electrostatics only)
  AtomGraphSynthesis* topology_synthesis_;              ///< Topology synthesis (owned)
  PhaseSpaceSynthesis* ps_synthesis_;                   ///< Phase space synthesis (owned)
  // FIX: Use llint for accumulator (neighbor list images require integers in PBC)
  CellGrid<double, llint, double, double4>* cell_grid_;  ///< Spatial decomposition (owned)
  PMIGrid* pme_grid_;                                   ///< PME grid (owned)
  double ewald_coeff_;                                  ///< Ewald coefficient for PME direct space (0.0 for non-periodic)

  /// \brief Infrastructure for GPU-accelerated lambda dynamics
  ///
  /// These objects enable the new launchLambdaDynamicsStep() API:
  /// - se_synthesis_: Exclusion mask synthesis wrapping exclusions_
  /// - launcher_: GPU kernel launch manager
  /// - mmctrl_: Molecular mechanics controls (timestep, integration mode)
  ///
  /// NOTE: CacheResource objects are now created locally in evaluateTotalEnergy()
  /// and propagateSystem() to avoid CUDA pinned memory fragmentation issues.
  /// This follows the same pattern as regular STORMM dynamics.
  ///
  /// All created in constructor if CUDA is available, nullptr otherwise
  StaticExclusionMaskSynthesis* se_synthesis_;  ///< Exclusion mask synthesis (owned)
  CoreKlManager* launcher_;                     ///< GPU kernel launcher (owned)
  MolecularMechanicsControls* mmctrl_;          ///< MD integration controls (owned)

  /// \brief Energy cache to avoid redundant evaluations
  ///
  /// GCMC moves alternate between lambda=0 and lambda=1 for molecules.
  /// The energy at the END of move N equals the energy at the START of move N+1.
  /// Caching eliminates 1 of 2 energy evaluations per move (50% for instant GCMC).
  ///
  /// IMPORTANT: Cache is invalidated by ANY coordinate-modifying operation
  /// (MD propagation, MC moves, etc.) to ensure consistency.
  bool energy_cached_;                   ///< Is cached energy valid?
  double cached_energy_;                 ///< Cached energy value
  size_t cached_lambda_hash_;            ///< Hash of all molecule lambda values

  std::ofstream ghost_stream_;           ///< Ghost ID output file
  std::ofstream log_stream_;             ///< Detailed log file

  /// \brief Immutable cache of LJ parameters
  ///
  /// These are initialized once from the topology at construction and never modified.
  /// This prevents topology corruption from affecting GCMC energy evaluations.
  /// Indexed by LJ type: cached_lj_sigma_[lj_type], cached_lj_epsilon_[lj_type]
  std::vector<double> cached_lj_sigma_;    ///< Cached sigma values by LJ type
  std::vector<double> cached_lj_epsilon_;  ///< Cached epsilon values by LJ type

  /// \brief Original cache values to detect corruption
  std::vector<double> original_cached_sigma_;    ///< Original sigma values for corruption detection
  std::vector<double> original_cached_epsilon_;  ///< Original epsilon values for corruption detection

  /// \brief GPU-resident arrays for lambda-scaled energy evaluation
  ///
  /// These Hybrid arrays maintain synchronized CPU/GPU copies of per-atom lambda
  /// parameters and LJ parameters for efficient GPU energy evaluation.
  Hybrid<double> lambda_vdw_;        ///< Per-atom VDW lambda values (GPU/CPU)
  Hybrid<double> lambda_ele_;        ///< Per-atom electrostatic lambda values (GPU/CPU)
  Hybrid<double> atom_sigma_;        ///< Per-atom LJ sigma values (GPU/CPU)
  Hybrid<double> atom_epsilon_;      ///< Per-atom LJ epsilon values (GPU/CPU)
  Hybrid<int> coupled_indices_;      ///< Indices of coupled atoms (GPU/CPU)
  Hybrid<double> energy_output_elec_;  ///< Per-coupled-atom elec energy output (GPU/CPU)
  Hybrid<double> energy_output_vdw_;   ///< Per-coupled-atom VDW energy output (GPU/CPU)
  Hybrid<double> total_elec_;          ///< Scalar total elec energy (GPU/CPU, size=1)
  Hybrid<double> total_vdw_;           ///< Scalar total VDW energy (GPU/CPU, size=1)

  /// \brief GPU-side work accumulation for NCMC protocol
  ///
  /// These arrays enable work accumulation entirely on GPU, eliminating
  /// 100 CPU↔GPU transfers per NCMC move (2 per perturbation × 50 steps).
  Hybrid<double> work_accumulator_;    ///< Total NCMC work (GPU/CPU, size=1)
  Hybrid<double> energy_before_elec_;  ///< Energy before lambda change - elec (GPU/CPU, size=1)
  Hybrid<double> energy_before_vdw_;   ///< Energy before lambda change - vdw (GPU/CPU, size=1)
  Hybrid<double> energy_after_elec_;   ///< Energy after lambda change - elec (GPU/CPU, size=1)
  Hybrid<double> energy_after_vdw_;    ///< Energy after lambda change - vdw (GPU/CPU, size=1)

  /// \brief GPU-side lambda scheduling for NCMC protocol
  ///
  /// These arrays enable lambda updates entirely on GPU, eliminating
  /// 100 CPU↔GPU lambda uploads per NCMC move.
  Hybrid<double> lambda_schedule_;     ///< NCMC lambda schedule (GPU/CPU, size=n_pert_steps+1)
  Hybrid<int> molecule_atom_indices_;  ///< Atom indices for target molecule (GPU/CPU)
  Hybrid<int> molecule_atom_count_;    ///< Number of atoms in target molecule (GPU/CPU, size=1)

  /// \brief Generalized Born implicit solvent support
  ///
  /// These members enable GB implicit solvent calculations for GCMC simulations.
  /// The workspace is created only if a GB model is specified (not NONE).
  topology::ImplicitSolventModel gb_model_;  ///< GB model to use (NONE = disabled)
  synthesis::ImplicitSolventWorkspace* gb_workspace_;  ///< GB workspace (nullptr if GB disabled)

  /// \brief GPU workspace for Monte Carlo moves (eliminates CPU↔GPU transfers)
  ///
  /// These Hybrid arrays enable fully GPU-accelerated MC moves with no coordinate transfers.
  /// Pre-allocated to maximum molecule size for efficiency.
  /// Flow: backup coords (GPU) → apply move (GPU) → evaluate energy (GPU) → restore if rejected (GPU)
  Hybrid<int> mc_atom_indices_;      ///< Atom indices for current MC move (GPU/CPU)
  Hybrid<double> mc_saved_x_;        ///< Backup X coordinates (GPU/CPU)
  Hybrid<double> mc_saved_y_;        ///< Backup Y coordinates (GPU/CPU)
  Hybrid<double> mc_saved_z_;        ///< Backup Z coordinates (GPU/CPU)
  Hybrid<double> mc_rotation_matrix_; ///< 3x3 rotation matrix for rotation/torsion moves (GPU/CPU, size=9)
  Hybrid<int> mc_rotating_atoms_;    ///< Atom indices for torsion rotation (GPU/CPU)

  /// \brief GPU-side cache for molecule atom indices (eliminates repeated uploads)
  ///
  /// This Hybrid array caches atom indices for a molecule on GPU to enable direct
  /// GPU-side lambda modifications without CPU→GPU transfers. Reused across all
  /// lambda modification calls during GCMC cycles.
  /// Pre-allocated to maximum molecule size for efficiency.
  Hybrid<int> gpu_molecule_indices_; ///< GPU-cached atom indices for lambda operations (GPU/CPU)

  /// \brief Flag to track if lambda arrays are up-to-date on GPU
  ///
  /// Set to true after adjustMoleculeLambdaGPU() modifies lambda values on GPU.
  /// Set to false after evaluateTotalEnergy() rebuilds coupled indices.
  /// When true, evaluateTotalEnergy() skips lambda array uploads and rebuilds coupled indices on GPU.
  bool gpu_lambda_arrays_dirty_; ///< Flag indicating GPU lambda arrays need coupled indices rebuild
  bool coupled_indices_valid_;   ///< Track whether coupled_indices_ matches current lambda values
  int coupled_atom_count_;       ///< Cached coupled atom count from latest rebuild
};

/// \brief GCMC sampler with spherical sampling region
///
/// Extends the base GCMCSampler to restrict insertion/deletion moves to a
/// spherical region of interest.
class GCMCSphereSampler : public GCMCSampler {
public:

  /// \brief Constructor with sphere definition
  ///
  /// \param topology         Pointer to AtomGraph
  /// \param ps               Pointer to PhaseSpace
  /// \param exclusions       Pointer to StaticExclusionMask
  /// \param temperature      Temperature (K)
  /// \param ghost_metadata   Metadata about ghost molecules in the topology
  /// \param ref_atoms        Atom indices for sphere center
  /// \param sphere_radius    Sphere radius (Angstroms)
  /// \param mu_ex            Excess chemical potential (kcal/mol)
  /// \param standard_volume  Standard volume (Angstrom^3, default 30.0)
  /// \param adams            Adams B parameter (optional, overrides mu_ex)
  /// \param adams_shift      Shift for Adams parameter
  /// \param max_N            Maximum number of molecules in sphere
  /// \param gb_model         Implicit solvent model to use (default NONE = disabled)
  /// \param resname          Residue name to track
  /// \param ghost_file       Ghost ID output file
  /// \param log_file         Detailed log file
  GCMCSphereSampler(AtomGraph* topology,
                    PhaseSpace* ps,
                    StaticExclusionMask* exclusions,
                    Thermostat* thermostat,
                    double temperature,
                    const GhostMoleculeMetadata& ghost_metadata,
                    const std::vector<int>& ref_atoms,
                    double sphere_radius,
                    double mu_ex,
                    double standard_volume = 30.0,
                    double adams = std::numeric_limits<double>::quiet_NaN(),
                    double adams_shift = 0.0,
                    int max_N = 100,
                    topology::ImplicitSolventModel gb_model = topology::ImplicitSolventModel::NONE,
                    const std::string& resname = "HOH",
                    const std::string& ghost_file = "gcmc-ghosts.txt",
                    const std::string& log_file = "gcmc.log");

  /// \brief Update sphere center based on reference atoms
  void updateSphereCenter();

  /// \brief Classify molecules by sphere membership
  ///
  /// Updates molecule status based on position relative to sphere.
  void classifyMolecules();

  /// \brief Check if a molecule is inside the sphere
  ///
  /// \param mol  Molecule to check
  /// \return True if molecule COG is within sphere
  bool isMoleculeInSphere(const GCMCMolecule& mol) const;

  /// \brief Select a random insertion site within the sphere
  ///
  /// \return Random position within the sphere
  double3 selectInsertionSite();

  /// \brief Attempt a standard GCMC insertion
  ///
  /// \return True if insertion was accepted
  virtual bool attemptInsertion();

  /// \brief Attempt a standard GCMC deletion
  ///
  /// \return True if deletion was accepted
  virtual bool attemptDeletion();

  /// \brief Run a complete GCMC cycle (insertion or deletion)
  ///
  /// \return True if the move was accepted
  bool runGCMCCycle();

protected:
  GCMCSphere sphere_;                    ///< Sampling sphere
  double B_;                             ///< Adams B parameter
  double mu_ex_;                         ///< Excess chemical potential
  double standard_volume_;               ///< Standard volume (A^3)
  int max_N_;                            ///< Maximum molecules in sphere
};

/// \brief GCMC sampler for entire simulation box
///
/// Extends the base GCMCSampler to perform insertion/deletion moves anywhere in the
/// simulation box without spatial restrictions.
class GCMCSystemSampler : public GCMCSampler {
public:

  /// \brief Constructor for system-wide GCMC sampler
  ///
  /// \param topology         Pointer to AtomGraph
  /// \param ps               Pointer to PhaseSpace
  /// \param exclusions       Pointer to StaticExclusionMask
  /// \param temperature      Temperature (K)
  /// \param ghost_metadata   Metadata about ghost molecules in the topology
  /// \param mu_ex            Excess chemical potential (kcal/mol)
  /// \param standard_volume  Standard volume (Angstrom^3, default 30.0)
  /// \param adams            Adams B parameter (optional, overrides mu_ex)
  /// \param adams_shift      Shift for Adams parameter
  /// \param gb_model         Implicit solvent model to use (default NONE = disabled)
  /// \param resname          Residue name to track
  /// \param ghost_file       Ghost ID output file
  /// \param log_file         Detailed log file
  GCMCSystemSampler(AtomGraph* topology,
                    PhaseSpace* ps,
                    StaticExclusionMask* exclusions,
                    Thermostat* thermostat,
                    double temperature,
                    const GhostMoleculeMetadata& ghost_metadata,
                    double mu_ex,
                    double standard_volume = 30.0,
                    double adams = std::numeric_limits<double>::quiet_NaN(),
                    double adams_shift = 0.0,
                    topology::ImplicitSolventModel gb_model = topology::ImplicitSolventModel::NONE,
                    const std::string& resname = "HOH",
                    const std::string& ghost_file = "gcmc-ghosts.txt",
                    const std::string& log_file = "gcmc.log");

  /// \brief Get the simulation box volume
  ///
  /// \return Box volume in Angstrom^3
  double getBoxVolume() const;

  /// \brief Select a random insertion site within the box
  ///
  /// \return Random position within the simulation box
  double3 selectInsertionSite();

  /// \brief Attempt a standard GCMC insertion anywhere in the box
  ///
  /// \return True if insertion was accepted
  virtual bool attemptInsertion();

  /// \brief Attempt a standard GCMC deletion
  ///
  /// \return True if deletion was accepted
  virtual bool attemptDeletion();

  /// \brief Run a complete GCMC cycle (insertion or deletion)
  ///
  /// \return True if the move was accepted
  bool runGCMCCycle();

  /// \brief Enable adaptive B protocol
  ///
  /// \param stage1_moves         Number of moves for discovery stage
  /// \param stage2_moves         Number of moves for coarse equilibration
  /// \param stage3_moves         Number of moves for fine annealing
  /// \param b_discovery          High B value for discovery stage
  /// \param target_occupancy     Target fraction of N_max for stage 2
  /// \param coarse_rate          Learning rate for stage 2
  /// \param fine_rate            Learning rate for stage 3
  /// \param b_min                Minimum B clamp value
  /// \param b_max                Maximum B clamp value
  void enableAdaptiveB(int stage1_moves, int stage2_moves, int stage3_moves,
                       double b_discovery, double target_occupancy,
                       double coarse_rate, double fine_rate,
                       double b_min, double b_max);

  /// \brief Compute adaptive B value based on current stage and molecule count
  ///
  /// \param move_number  Current move number in the simulation
  /// \return Adjusted B value for the current stage
  double computeAdaptiveB(int move_number);

  /// \brief Count currently active fragments
  ///
  /// \return Number of molecules with lambda > 0.5
  int countActiveFragments() const;

  /// \brief Update the current annealing stage based on move number
  ///
  /// \param move_number  Current move number in the simulation
  void updateStageProgress(int move_number);

  /// \brief Get current B value (either fixed or adaptive)
  ///
  /// \return Current B value
  double getCurrentB() const { return adaptive_b_enabled_ ? current_adaptive_b_ : B_; }

  /// \brief Check if adaptive B is enabled
  ///
  /// \return True if adaptive B protocol is active
  bool isAdaptiveBEnabled() const { return adaptive_b_enabled_; }

  /// \brief Get current annealing stage
  ///
  /// \return Current stage of the annealing protocol
  AnnealingStage getCurrentStage() const { return current_stage_; }

  /// \brief Get maximum fragment count discovered
  ///
  /// \return Maximum number of fragments observed in stage 1
  int getMaxFragments() const { return n_max_fragments_; }

  /// \brief Run hybrid MD/MC simulation with continuous lambda-aware MD
  ///
  /// Runs MD with lambda-scaled forces (no lambda changes during MD).
  /// Periodically attempts randomly selected moves:
  /// - GCMC insertion/deletion (via NCMC, includes MD propagation)
  /// - MC translation (instant coordinate change)
  /// - MC rotation (instant coordinate change)
  /// - MC torsion (instant coordinate change)
  ///
  /// \param total_md_steps     Total number of MD steps to run
  /// \param move_frequency     Attempt move every N MD steps
  /// \param gcmc_probability   Probability of GCMC vs MC move (0-1)
  void runHybridSimulation(int total_md_steps,
                           int move_frequency = 100,
                           double gcmc_probability = 0.5);

protected:
  double B_;                             ///< Adams B parameter
  double mu_ex_;                         ///< Excess chemical potential
  double standard_volume_;               ///< Standard volume (A^3)
  double box_volume_;                    ///< Current simulation box volume

  // Adaptive B protocol members
  bool adaptive_b_enabled_;              ///< Enable adaptive B protocol
  AnnealingStage current_stage_;         ///< Current annealing stage
  int stage1_moves_;                     ///< Number of moves in discovery stage
  int stage2_moves_;                     ///< Number of moves in coarse equilibration
  int stage3_moves_;                     ///< Number of moves in fine annealing
  double b_discovery_;                   ///< High B value for discovery
  double target_occupancy_;              ///< Target fraction of N_max (e.g., 0.5)
  double coarse_learning_rate_;          ///< Learning rate for stage 2
  double fine_learning_rate_;            ///< Learning rate for stage 3
  double b_min_;                         ///< Minimum B clamp value
  double b_max_;                         ///< Maximum B clamp value
  int n_max_fragments_;                  ///< Maximum fragments observed in stage 1
  double current_adaptive_b_;            ///< Current adaptive B value
  int move_counter_;                     ///< Track moves for logging
};

/// \brief NCMC-enhanced system-wide GCMC sampler
///
/// Uses Nonequilibrium Candidate Monte Carlo (NCMC) protocols to improve
/// acceptance rates for insertion and deletion moves in the entire simulation box.
class NCMCSystemSampler : public GCMCSystemSampler {
public:

  /// \brief Constructor for NCMC-enhanced system-wide GCMC sampler
  ///
  /// \param topology              AtomGraph pointer
  /// \param ps                    PhaseSpace pointer
  /// \param exclusions            StaticExclusionMask pointer
  /// \param thermostat            Thermostat pointer (owns integration state)
  /// \param temperature           Temperature (K)
  /// \param ghost_metadata        Metadata about ghost molecules in the topology
  /// \param n_pert_steps          Number of lambda perturbation steps
  /// \param n_prop_steps_per_pert MD steps between lambda changes
  /// \param timestep              Integration timestep (fs)
  /// \param lambdas               Custom lambda schedule (optional)
  /// \param record_traj           Record move trajectories
  /// \param mu_ex                 Excess chemical potential (kcal/mol)
  /// \param standard_volume       Standard volume (Angstrom^3)
  /// \param adams                 Adams B parameter (optional)
  /// \param adams_shift           Shift for Adams parameter
  /// \param gb_model              Implicit solvent model to use (default NONE = disabled)
  /// \param resname               Residue name to track
  /// \param ghost_file            Ghost ID output file
  /// \param log_file              Detailed log file
  NCMCSystemSampler(AtomGraph* topology,
                    PhaseSpace* ps,
                    StaticExclusionMask* exclusions,
                    Thermostat* thermostat,
                    double temperature,
                    const GhostMoleculeMetadata& ghost_metadata,
                    int n_pert_steps = 1,
                    int n_prop_steps_per_pert = 1,
                    double timestep = 2.0,
                    const std::vector<double>& lambdas = {},
                    bool record_traj = false,
                    double mu_ex = 0.0,
                    double standard_volume = 30.0,
                    double adams = std::numeric_limits<double>::quiet_NaN(),
                    double adams_shift = 0.0,
                    topology::ImplicitSolventModel gb_model = topology::ImplicitSolventModel::NONE,
                    const std::string& resname = "HOH",
                    const std::string& ghost_file = "gcmc-ghosts.txt",
                    const std::string& log_file = "gcmc.log");

  /// \brief Attempt an NCMC insertion move anywhere in the box
  ///
  /// \return True if insertion was accepted
  bool attemptInsertion() override;

  /// \brief Attempt an NCMC deletion move
  ///
  /// \return True if deletion was accepted
  bool attemptDeletion() override;

  /// \brief Propagate system with MD for n_steps
  ///
  /// Uses dynaStep() free function with thermostat.
  ///
  /// \param n_steps Number of MD steps to take
  void propagateSystem(int n_steps);

  /// \brief Run a complete GCMC cycle
  ///
  /// Attempts one insertion or deletion move.
  ///
  /// \return True if the move was accepted
  bool runGCMCCycle();

  /// \brief Get the NCMC protocol
  ///
  /// \return Reference to the protocol object
  const NCMCProtocol& getProtocol() const;

private:
  NCMCProtocol protocol_;                ///< NCMC switching protocol
  DynamicsControls dyn_controls_;        ///< Dynamics control parameters
  bool record_traj_;                     ///< Record trajectories of moves

  /// \brief Perform NCMC protocol work calculation
  ///
  /// \param mol           Molecule being perturbed
  /// \param forward       True for insertion (0→1), false for deletion (1→0)
  /// \param propagate     If true, run MD between lambda changes
  /// \return Protocol work in kcal/mol
  double performNCMCProtocol(GCMCMolecule& mol, bool forward, bool propagate = true);
};

/// \brief NCMC-enhanced GCMC sampler
///
/// Uses Nonequilibrium Candidate Monte Carlo (NCMC) protocols to improve
/// acceptance rates for insertion and deletion moves.
class NCMCSampler : public GCMCSphereSampler {
public:

  /// \brief Constructor for NCMC-enhanced GCMC sampler
  ///
  /// \param topology              AtomGraph pointer
  /// \param ps                    PhaseSpace pointer
  /// \param exclusions            StaticExclusionMask pointer
  /// \param thermostat            Thermostat pointer (owns integration state)
  /// \param temperature           Temperature (K)
  /// \param ghost_metadata        Metadata about ghost molecules in the topology
  /// \param n_pert_steps          Number of lambda perturbation steps
  /// \param n_prop_steps_per_pert MD steps between lambda changes
  /// \param timestep              Integration timestep (fs)
  /// \param lambdas               Custom lambda schedule (optional)
  /// \param record_traj           Record move trajectories
  /// \param ref_atoms             Atom indices for sphere center
  /// \param sphere_radius         Sphere radius (Angstroms)
  /// \param mu_ex                 Excess chemical potential (kcal/mol)
  /// \param standard_volume       Standard volume (Angstrom^3)
  /// \param adams                 Adams B parameter (optional)
  /// \param adams_shift           Shift for Adams parameter
  /// \param max_N                 Maximum molecules in sphere
  /// \param gb_model              Implicit solvent model to use (default NONE = disabled)
  /// \param resname               Residue name to track
  /// \param ghost_file            Ghost ID output file
  /// \param log_file              Detailed log file
  NCMCSampler(AtomGraph* topology,
              PhaseSpace* ps,
              StaticExclusionMask* exclusions,
              Thermostat* thermostat,
              double temperature,
              const GhostMoleculeMetadata& ghost_metadata,
              int n_pert_steps = 1,
              int n_prop_steps_per_pert = 1,
              double timestep = 2.0,
              const std::vector<double>& lambdas = {},
              bool record_traj = false,
              const std::vector<int>& ref_atoms = {},
              double sphere_radius = 5.0,
              double mu_ex = 0.0,
              double standard_volume = 30.0,
              double adams = std::numeric_limits<double>::quiet_NaN(),
              double adams_shift = 0.0,
              int max_N = 100,
              topology::ImplicitSolventModel gb_model = topology::ImplicitSolventModel::NONE,
              const std::string& resname = "HOH",
              const std::string& ghost_file = "gcmc-ghosts.txt",
              const std::string& log_file = "gcmc.log");

  /// \brief Attempt an NCMC insertion move
  ///
  /// \return True if insertion was accepted
  bool attemptInsertion() override;

  /// \brief Attempt an NCMC deletion move
  ///
  /// \return True if deletion was accepted
  bool attemptDeletion() override;

  /// \brief Propagate system with MD for n_steps
  ///
  /// Uses dynaStep() free function with thermostat.
  ///
  /// \param n_steps Number of MD steps to take
  void propagateSystem(int n_steps);

  /// \brief Run a complete GCMC cycle
  ///
  /// Attempts one insertion or deletion move based on current state.
  ///
  /// \return True if the move was accepted
  bool runGCMCCycle();

  /// \brief Get the NCMC protocol
  ///
  /// \return Reference to the protocol object
  const NCMCProtocol& getProtocol() const;

private:
  NCMCProtocol protocol_;                ///< NCMC switching protocol
  DynamicsControls dyn_controls_;        ///< Dynamics control parameters
  bool record_traj_;                     ///< Record trajectories of moves


  /// \brief Perform NCMC protocol work calculation
  ///
  /// \param mol           Molecule being perturbed
  /// \param forward       True for insertion (0→1), false for deletion (1→0)
  /// \param propagate     If true, run MD between lambda changes
  /// \return Protocol work in kcal/mol
  double performNCMCProtocol(GCMCMolecule& mol, bool forward, bool propagate = true);
};

} // namespace sampling
} // namespace stormm

#endif
