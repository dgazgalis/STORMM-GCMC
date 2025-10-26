#include "copyright.h"
#include "Constants/hpc_bounds.h"
#include <array>
#include <iostream>
#ifdef STORMM_USE_HPC
#include <cuda_runtime.h>
#endif
#include "Numerics/split_fixed_precision.h"
#include "Synthesis/synthesis_enumerators.h"
#include "Trajectory/trajectory_enumerators.h"
#include "mm_controls.h"

#ifndef STORMM_MMCTRL_DEBUG
#define STORMM_MMCTRL_DEBUG 0
#endif

namespace stormm {
namespace mm {

using card::HybridFormat;
using card::HybridKind;
using card::HybridTargetLevel;
using stmath::ReductionGoal;
using namelist::maximum_nt_warp_multiplicity;
using numerics::AccumulationMethod;
using topology::UnitCellType;
using trajectory::IntegrationStage;
  
//-------------------------------------------------------------------------------------------------
MolecularMechanicsControls::MolecularMechanicsControls(const double initial_step_in,
                                                       const int sd_cycles_in,
                                                       const int max_cycles_in,
                                                       const int nt_warp_multiplicity_in,
                                                       const double electrostatic_cutoff_in,
                                                       const double van_der_waals_cutoff_in) :
    step_number{0}, sd_cycles{sd_cycles_in}, max_cycles{max_cycles_in},
    initial_step{initial_step_in}, nt_warp_multiplicity{nt_warp_multiplicity_in},
    electrostatic_cutoff{electrostatic_cutoff_in}, van_der_waals_cutoff{van_der_waals_cutoff_in},
    // FIX: All pointer Hybrids must use DECOUPLED to match int_data format
    vwu_progress{HybridKind::POINTER, "mm_vwu_counters", HybridFormat::DECOUPLED},
    velocity_update_progress{HybridKind::POINTER, "mm_vupt_counters", HybridFormat::DECOUPLED},
    velocity_constraint_progress{HybridKind::POINTER, "mm_vcns_counters", HybridFormat::DECOUPLED},
    position_update_progress{HybridKind::POINTER, "mm_pupt_counters", HybridFormat::DECOUPLED},
    geometry_constraint_progress{HybridKind::POINTER, "mm_gcns_counters", HybridFormat::DECOUPLED},
    nbwu_progress{HybridKind::POINTER, "mm_nbwu_counters", HybridFormat::DECOUPLED},
    pmewu_progress{HybridKind::POINTER, "mm_pmewu_counters", HybridFormat::DECOUPLED},
    gbrwu_progress{HybridKind::POINTER, "mm_gbrwu_counters", HybridFormat::DECOUPLED},
    gbdwu_progress{HybridKind::POINTER, "mm_gbdwu_counters", HybridFormat::DECOUPLED},
    gather_wu_progress{HybridKind::POINTER, "mm_gtwu_counters", HybridFormat::DECOUPLED},
    scatter_wu_progress{HybridKind::POINTER, "mm_scwu_counters", HybridFormat::DECOUPLED},
    all_reduce_wu_progress{HybridKind::POINTER, "mm_rdwu_counters", HybridFormat::DECOUPLED},
    int_data{work_unit_payload_words + work_unit_guard_words, "work_unit_prog_data",
             HybridFormat::DECOUPLED},
    work_unit_guard_offset{work_unit_payload_words}
{
#if STORMM_USE_HPC && STORMM_MMCTRL_DEBUG
  std::cerr << "DEBUG: MMControls ctor allocate work_unit_prog_data host="
            << static_cast<const void*>(int_data.data()) << std::endl;
#endif
  vwu_progress.setPointer(&int_data,                                  0, 2 * warp_size_int);
  velocity_update_progress.setPointer(&int_data,      2 * warp_size_int, 2 * warp_size_int);
  velocity_constraint_progress.setPointer(&int_data,  4 * warp_size_int, 2 * warp_size_int);
  position_update_progress.setPointer(&int_data,      6 * warp_size_int, 2 * warp_size_int);
  geometry_constraint_progress.setPointer(&int_data,  8 * warp_size_int, 2 * warp_size_int);
  nbwu_progress.setPointer(&int_data,                10 * warp_size_int, 2 * warp_size_int);
  pmewu_progress.setPointer(&int_data,               12 * warp_size_int, 2 * warp_size_int);
  gbrwu_progress.setPointer(&int_data,               14 * warp_size_int, 2 * warp_size_int);
  gbdwu_progress.setPointer(&int_data,               16 * warp_size_int, 2 * warp_size_int);
  gather_wu_progress.setPointer(&int_data,           18 * warp_size_int, 2 * warp_size_int);
  scatter_wu_progress.setPointer(&int_data,          20 * warp_size_int, 2 * warp_size_int);
  all_reduce_wu_progress.setPointer(&int_data,       22 * warp_size_int, 2 * warp_size_int);
  initializeWorkUnitGuards();
}

//-------------------------------------------------------------------------------------------------
MolecularMechanicsControls::MolecularMechanicsControls(const DynamicsControls &user_input) :
    MolecularMechanicsControls(default_minimize_dx0, default_minimize_ncyc,
                               user_input.getStepCount(), user_input.getNTWarpMultiplicity(),
                               user_input.getElectrostaticCutoff(),
                               user_input.getVanDerWaalsCutoff())
{}

//-------------------------------------------------------------------------------------------------
MolecularMechanicsControls::MolecularMechanicsControls(const MinimizeControls &user_input) :
    MolecularMechanicsControls(user_input.getInitialStep(), user_input.getSteepestDescentCycles(),
                               user_input.getTotalCycles(), user_input.getElectrostaticCutoff(),
                               user_input.getLennardJonesCutoff())
{}

//-------------------------------------------------------------------------------------------------
MolecularMechanicsControls::
MolecularMechanicsControls(const MolecularMechanicsControls &original) :
  step_number{original.step_number},
  sd_cycles{original.sd_cycles},
  max_cycles{original.max_cycles},
  initial_step{original.initial_step},
  nt_warp_multiplicity{original.nt_warp_multiplicity},
  electrostatic_cutoff{original.electrostatic_cutoff},
  van_der_waals_cutoff{original.van_der_waals_cutoff},
  vwu_progress{original.vwu_progress},
  velocity_update_progress{original.velocity_update_progress},
  velocity_constraint_progress{original.velocity_constraint_progress},
  position_update_progress{original.position_update_progress},
  geometry_constraint_progress{original.geometry_constraint_progress},
  nbwu_progress{original.nbwu_progress},
  pmewu_progress{original.pmewu_progress},
  gbrwu_progress{original.gbrwu_progress},
  gbdwu_progress{original.gbdwu_progress},
  gather_wu_progress{original.gather_wu_progress},
  scatter_wu_progress{original.scatter_wu_progress},
  all_reduce_wu_progress{original.all_reduce_wu_progress},
  int_data{original.int_data},
  work_unit_guard_offset{original.work_unit_guard_offset}
{
#if STORMM_USE_HPC && STORMM_MMCTRL_DEBUG
  std::cerr << "DEBUG: MMControls copy ctor work_unit_prog_data host="
            << static_cast<const void*>(int_data.data()) << std::endl;
#endif
  rebasePointers();
  initializeWorkUnitGuards();
}

//-------------------------------------------------------------------------------------------------
MolecularMechanicsControls&
MolecularMechanicsControls::operator=(const MolecularMechanicsControls &other) {

  // Guard against self assignment
  if (this == &other) {
    return *this;
  }
  step_number = other.step_number;
  sd_cycles = other.sd_cycles;
  max_cycles = other.max_cycles;
  initial_step = other.initial_step;
  nt_warp_multiplicity = other.nt_warp_multiplicity;
  electrostatic_cutoff = other.electrostatic_cutoff;
  van_der_waals_cutoff = other.van_der_waals_cutoff;
  vwu_progress = other.vwu_progress;
  velocity_update_progress = other.velocity_update_progress;
  velocity_constraint_progress = other.velocity_constraint_progress;
  position_update_progress = other.position_update_progress;
  geometry_constraint_progress = other.geometry_constraint_progress;
  nbwu_progress = other.nbwu_progress;
  pmewu_progress = other.pmewu_progress;
  gbrwu_progress = other.gbrwu_progress;
  gbdwu_progress = other.gbdwu_progress;
  gather_wu_progress = other.gather_wu_progress;
  scatter_wu_progress = other.scatter_wu_progress;
  all_reduce_wu_progress = other.all_reduce_wu_progress;
  int_data = other.int_data;
  work_unit_guard_offset = other.work_unit_guard_offset;

  // Repair pointers and return the result
  rebasePointers();
  initializeWorkUnitGuards();
  return *this;
}

//-------------------------------------------------------------------------------------------------
MolecularMechanicsControls::MolecularMechanicsControls(MolecularMechanicsControls &&original) :
  step_number{original.step_number},
  sd_cycles{original.sd_cycles},
  max_cycles{original.max_cycles},
  initial_step{original.initial_step},
  nt_warp_multiplicity{original.nt_warp_multiplicity},
  electrostatic_cutoff{original.electrostatic_cutoff},
  van_der_waals_cutoff{original.van_der_waals_cutoff},
  vwu_progress{std::move(original.vwu_progress)},
  velocity_update_progress{std::move(original.velocity_update_progress)},
  velocity_constraint_progress{std::move(original.velocity_constraint_progress)},
  position_update_progress{std::move(original.position_update_progress)},
  geometry_constraint_progress{std::move(original.geometry_constraint_progress)},
  nbwu_progress{std::move(original.nbwu_progress)},
  pmewu_progress{std::move(original.pmewu_progress)},
  gbrwu_progress{std::move(original.gbrwu_progress)},
  gbdwu_progress{std::move(original.gbdwu_progress)},
  gather_wu_progress{std::move(original.gather_wu_progress)},
  scatter_wu_progress{std::move(original.scatter_wu_progress)},
  all_reduce_wu_progress{std::move(original.all_reduce_wu_progress)},
  int_data{std::move(original.int_data)},
  work_unit_guard_offset{original.work_unit_guard_offset}
{
#if STORMM_USE_HPC && STORMM_MMCTRL_DEBUG
  std::cerr << "DEBUG: MMControls move ctor work_unit_prog_data host="
            << static_cast<const void*>(int_data.data()) << std::endl;
#endif
  initializeWorkUnitGuards();
}

//-------------------------------------------------------------------------------------------------
MolecularMechanicsControls::~MolecularMechanicsControls() {
  verifyWorkUnitGuards("~MolecularMechanicsControls");
#if STORMM_MMCTRL_DEBUG
  std::cout << "DEBUG: destroying MolecularMechanicsControls (step=" << step_number
            << ") releasing work_unit_prog_data" << std::endl;
#endif
}

//-------------------------------------------------------------------------------------------------
MolecularMechanicsControls&
MolecularMechanicsControls::operator=(MolecularMechanicsControls &&other) {

  // Guard against self assignment
  if (this == &other) {
    return *this;
  }
  step_number = other.step_number;
  sd_cycles = other.sd_cycles;
  max_cycles = other.max_cycles;
  initial_step = other.initial_step;
  nt_warp_multiplicity = other.nt_warp_multiplicity;
  electrostatic_cutoff = other.electrostatic_cutoff;
  van_der_waals_cutoff = other.van_der_waals_cutoff;
  vwu_progress = std::move(other.vwu_progress);
  velocity_update_progress = std::move(other.velocity_update_progress);
  velocity_constraint_progress = std::move(other.velocity_constraint_progress);
  position_update_progress = std::move(other.position_update_progress);
  geometry_constraint_progress = std::move(other.geometry_constraint_progress);
  nbwu_progress = std::move(other.nbwu_progress);
  pmewu_progress = std::move(other.pmewu_progress);
  gbrwu_progress = std::move(other.gbrwu_progress);
  gbdwu_progress = std::move(other.gbdwu_progress);
  gather_wu_progress = std::move(other.gather_wu_progress);
  scatter_wu_progress = std::move(other.scatter_wu_progress);
  all_reduce_wu_progress = std::move(other.all_reduce_wu_progress);
  int_data = std::move(other.int_data);
  work_unit_guard_offset = other.work_unit_guard_offset;
  initializeWorkUnitGuards();
  return *this;
}

//-------------------------------------------------------------------------------------------------
int MolecularMechanicsControls::getStepNumber() const {
  return step_number;
}

//-------------------------------------------------------------------------------------------------
int MolecularMechanicsControls::getSteepestDescentCycles() const {
  return sd_cycles;
}

//-------------------------------------------------------------------------------------------------
int MolecularMechanicsControls::getTotalCycles() const {
  return max_cycles;
}

//-------------------------------------------------------------------------------------------------
double MolecularMechanicsControls::getInitialMinimizationStep() const {
  return initial_step;
}

//-------------------------------------------------------------------------------------------------
double MolecularMechanicsControls::getElectrostaticCutoff() const {
  return electrostatic_cutoff;
}

//-------------------------------------------------------------------------------------------------
double MolecularMechanicsControls::getVanDerWaalsCutoff() const {
  return van_der_waals_cutoff;
}

//-------------------------------------------------------------------------------------------------
int MolecularMechanicsControls::getValenceWorkUnitProgress(const int counter_index,
                                                           const HybridTargetLevel tier) const {
  switch (tier) {
  case HybridTargetLevel::HOST:
    return vwu_progress.readHost(counter_index);    
    break;
#ifdef STORMM_USE_HPC
  case HybridTargetLevel::DEVICE:
    return vwu_progress.readDevice(counter_index);
    break;
#endif
  }
  __builtin_unreachable();
}

//-------------------------------------------------------------------------------------------------
int MolecularMechanicsControls::getNonbondedWorkUnitProgress(const int counter_index,
                                                             const HybridTargetLevel tier) const {
  switch (tier) {
  case HybridTargetLevel::HOST:
    return nbwu_progress.readHost(counter_index);    
    break;
#ifdef STORMM_USE_HPC
  case HybridTargetLevel::DEVICE:
    return nbwu_progress.readDevice(counter_index);
    break;
#endif
  }
  __builtin_unreachable();
}

//-------------------------------------------------------------------------------------------------
int MolecularMechanicsControls::getPmeWorkUnitProgress(const int counter_index,
                                                       const HybridTargetLevel tier) const {
  switch (tier) {
  case HybridTargetLevel::HOST:
    return pmewu_progress.readHost(counter_index);    
    break;
#ifdef STORMM_USE_HPC
  case HybridTargetLevel::DEVICE:
    return pmewu_progress.readDevice(counter_index);
    break;
#endif
  }
  __builtin_unreachable();
}

//-------------------------------------------------------------------------------------------------
int MolecularMechanicsControls::getReductionWorkUnitProgress(const int counter_index,
                                                             const ReductionStage process,
                                                             const HybridTargetLevel tier) const {
  switch (tier) {
  case HybridTargetLevel::HOST:
    switch (process) {
    case ReductionStage::GATHER:
      return gather_wu_progress.readHost(counter_index);
    case ReductionStage::SCATTER:
    case ReductionStage::RESCALE:
      return scatter_wu_progress.readHost(counter_index);
    case ReductionStage::ALL_REDUCE:
      return all_reduce_wu_progress.readHost(counter_index);
    }
    break;
#ifdef STORMM_USE_HPC
  case HybridTargetLevel::DEVICE:
    switch (process) {
    case ReductionStage::GATHER:
      return gather_wu_progress.readDevice(counter_index);
    case ReductionStage::SCATTER:
    case ReductionStage::RESCALE:
      return scatter_wu_progress.readDevice(counter_index);
    case ReductionStage::ALL_REDUCE:
      return all_reduce_wu_progress.readDevice(counter_index);
    }
    break;
#endif
  }
  __builtin_unreachable();
}

//-------------------------------------------------------------------------------------------------
MMControlKit<double> MolecularMechanicsControls::dpData(const HybridTargetLevel tier) {
  return MMControlKit<double>(step_number, sd_cycles, max_cycles, initial_step,
                              nt_warp_multiplicity, electrostatic_cutoff, van_der_waals_cutoff,
                              vwu_progress.data(tier), velocity_update_progress.data(tier),
                              velocity_constraint_progress.data(tier),
                              position_update_progress.data(tier),
                              geometry_constraint_progress.data(tier), nbwu_progress.data(tier),
                              pmewu_progress.data(tier), gbrwu_progress.data(tier),
                              gbdwu_progress.data(tier), gather_wu_progress.data(tier),
                              scatter_wu_progress.data(tier), all_reduce_wu_progress.data(tier));
}

//-------------------------------------------------------------------------------------------------
MMControlKit<float> MolecularMechanicsControls::spData(const HybridTargetLevel tier) {
  return MMControlKit<float>(step_number, sd_cycles, max_cycles, initial_step,
                             nt_warp_multiplicity, electrostatic_cutoff, van_der_waals_cutoff,
                             vwu_progress.data(tier), velocity_update_progress.data(tier),
                             velocity_constraint_progress.data(tier),
                             position_update_progress.data(tier),
                             geometry_constraint_progress.data(tier), nbwu_progress.data(tier),
                             pmewu_progress.data(tier), gbrwu_progress.data(tier),
                             gbdwu_progress.data(tier), gather_wu_progress.data(tier),
                             scatter_wu_progress.data(tier), all_reduce_wu_progress.data(tier));
}

//-------------------------------------------------------------------------------------------------
void MolecularMechanicsControls::primeWorkUnitCounters(const CoreKlManager &launcher,
                                                       const EvaluateForce eval_frc,
                                                       const EvaluateEnergy eval_nrg,
                                                       const ClashResponse softcore,
                                                       const VwuGoal purpose,
                                                       const PrecisionModel valence_prec,
                                                       const PrecisionModel nonbond_prec,
                                                       const QMapMethod qspread_approach,
                                                       const PrecisionModel acc_prec,
                                                       const size_t image_coord_type,
                                                       const int qspread_order,
                                                       const NeighborListKind nbgr_config,
                                                       const TinyBoxPresence has_tiny_box,
                                                       const AtomGraphSynthesis &poly_ag) {
  const GpuDetails wgpu = launcher.getGpu();
  const ImplicitSolventModel igb = poly_ag.getImplicitSolventModel();
  const int arch_major = wgpu.getArchMajor();
  const int arch_minor = wgpu.getArchMinor();
  const int smp_count = wgpu.getSMPCount();

  // The numbers of blocks that will be launched in each grid are critical for priming the work
  // unit progress counters.  As such, they (the grid launch size) should be consistent across
  // different variants of each kernel for a given precision level, even if the exact numbers of
  // threads per block have to vary based on what each block of that kernel variant is required to
  // do.  Here, it should suffice to query the launch parameters of just one of the blocks.
  const int2 vwu_lp = launcher.getValenceKernelDims(valence_prec, eval_frc, eval_nrg,
                                                    AccumulationMethod::SPLIT, purpose, softcore);
  const int2 vadv_lp = launcher.getIntegrationKernelDims(valence_prec, AccumulationMethod::SPLIT,
                                                         IntegrationStage::VELOCITY_ADVANCE);
  const int2 vcns_lp = launcher.getIntegrationKernelDims(valence_prec, AccumulationMethod::SPLIT,
                                                         IntegrationStage::VELOCITY_CONSTRAINT);
  const int2 padv_lp = launcher.getIntegrationKernelDims(valence_prec, AccumulationMethod::SPLIT,
                                                         IntegrationStage::POSITION_ADVANCE);
  const int2 gcns_lp = launcher.getIntegrationKernelDims(valence_prec, AccumulationMethod::SPLIT,
                                                         IntegrationStage::GEOMETRY_CONSTRAINT);
  switch (poly_ag.getUnitCellType()) {
  case UnitCellType::NONE:
    {
      const int2 gbrwu_lp = launcher.getBornRadiiKernelDims(nonbond_prec,
                                                            poly_ag.getNonbondedWorkType(),
                                                            AccumulationMethod::SPLIT, igb);
      const int2 gbdwu_lp = launcher.getBornDerivativeKernelDims(nonbond_prec,
                                                                 poly_ag.getNonbondedWorkType(),
                                                                 AccumulationMethod::SPLIT, igb);
      const int2 nbwu_lp = launcher.getNonbondedKernelDims(nonbond_prec,
                                                           poly_ag.getNonbondedWorkType(),
                                                           eval_frc, eval_nrg,
                                                           AccumulationMethod::SPLIT, igb,
                                                           softcore);
      for (int i = 0; i < twice_warp_size_int; i++) {
        nbwu_progress.putHost(nbwu_lp.x, i);
        gbrwu_progress.putHost(gbrwu_lp.x, i);
        gbdwu_progress.putHost(gbdwu_lp.x, i);
      }
    }
    break;
  case UnitCellType::ORTHORHOMBIC:
  case UnitCellType::TRICLINIC:
    {
      // The launch grid for the density mapping kernel will be identical for short- and long-form
      // fixed precision accumulations.
      const int2 pmewu_lp = launcher.getDensityMappingKernelDims(qspread_approach, nonbond_prec,
                                                                 acc_prec, true, image_coord_type,
                                                                 qspread_order);
      const PrecisionModel cellgrid_prec = (image_coord_type == double_type_index ||
                                            image_coord_type == llint_type_index) ?
                                           PrecisionModel::DOUBLE : PrecisionModel::SINGLE;
      const int2 nbwu_lp = launcher.getPMEPairsKernelDims(cellgrid_prec, nonbond_prec, nbgr_config,
                                                          has_tiny_box, eval_frc, eval_nrg,
                                                          softcore);
      for (int i = 0; i < twice_warp_size_int; i++) {
        pmewu_progress.putHost(pmewu_lp.x, i);
        nbwu_progress.putHost(nbwu_lp.x * (nbwu_lp.y / warp_size_int), i);
      }
    }
    break;
  }
  const int2 rdwu_lp = launcher.getReductionKernelDims(valence_prec,
                                                       ReductionGoal::CONJUGATE_GRADIENT,
                                                       ReductionStage::ALL_REDUCE);
  const int vwu_block_count  = vwu_lp.x;
  const int vadv_block_count = vadv_lp.x;
  const int vcns_block_count = vcns_lp.x;
  const int padv_block_count = padv_lp.x;
  const int gcns_block_count = gcns_lp.x;
  const int gtwu_block_count = rdwu_lp.x;
  const int scwu_block_count = rdwu_lp.x;
  const int rdwu_block_count = rdwu_lp.x;
  for (int i = 0; i < twice_warp_size_int; i++) {
    vwu_progress.putHost(vwu_block_count, i);
    velocity_update_progress.putHost(vadv_block_count, i);
    velocity_constraint_progress.putHost(vcns_block_count, i);
    position_update_progress.putHost(padv_block_count, i);
    geometry_constraint_progress.putHost(gcns_block_count, i);
    gather_wu_progress.putHost(gtwu_block_count, i);
    scatter_wu_progress.putHost(scwu_block_count, i);
    all_reduce_wu_progress.putHost(rdwu_block_count, i);
  }
  initializeWorkUnitGuards();
#ifdef STORMM_USE_HPC
  upload();
#endif
}

//-------------------------------------------------------------------------------------------------
void MolecularMechanicsControls::primeWorkUnitCounters(const CoreKlManager &launcher,
                                                       const EvaluateForce eval_frc,
                                                       const EvaluateEnergy eval_nrg,
                                                       const ClashResponse softcore,
                                                       const VwuGoal purpose,
                                                       const PrecisionModel valence_prec,
                                                       const PrecisionModel nonbond_prec,
                                                       const AtomGraphSynthesis &poly_ag) {
  primeWorkUnitCounters(launcher, eval_frc, eval_nrg, softcore, purpose, valence_prec,
                        nonbond_prec, QMapMethod::GENERAL_PURPOSE, valence_prec, int_type_index,
                        4, NeighborListKind::MONO, TinyBoxPresence::NO, poly_ag);
}

//-------------------------------------------------------------------------------------------------
void MolecularMechanicsControls::primeWorkUnitCounters(const CoreKlManager &launcher,
                                                       const EvaluateForce eval_frc,
                                                       const EvaluateEnergy eval_nrg,
                                                       const VwuGoal purpose,
                                                       const PrecisionModel valence_prec,
                                                       const PrecisionModel nonbond_prec,
                                                       const AtomGraphSynthesis &poly_ag) {
  primeWorkUnitCounters(launcher, eval_frc, eval_nrg, ClashResponse::NONE, purpose, valence_prec,
                        nonbond_prec, QMapMethod::GENERAL_PURPOSE, valence_prec, int_type_index,
                        4, NeighborListKind::MONO, TinyBoxPresence::NO, poly_ag);
}

//-------------------------------------------------------------------------------------------------
void MolecularMechanicsControls::primeWorkUnitCounters(const CoreKlManager &launcher,
                                                       const EvaluateForce eval_frc,
                                                       const EvaluateEnergy eval_nrg,
                                                       const VwuGoal purpose,
                                                       const PrecisionModel general_prec,
                                                       const AtomGraphSynthesis &poly_ag) {
  primeWorkUnitCounters(launcher, eval_frc, eval_nrg, ClashResponse::NONE, purpose, general_prec,
                        general_prec, QMapMethod::GENERAL_PURPOSE, general_prec, int_type_index,
                        4, NeighborListKind::MONO, TinyBoxPresence::NO, poly_ag);
}

//-------------------------------------------------------------------------------------------------
void MolecularMechanicsControls::setNTWarpMultiplicity(const int mult_in) {
  nt_warp_multiplicity = mult_in;
}

//-------------------------------------------------------------------------------------------------
void MolecularMechanicsControls::incrementStep() {
  step_number += 1;
}
  
#ifdef STORMM_USE_HPC
//-------------------------------------------------------------------------------------------------
void MolecularMechanicsControls::upload() {
  int_data.upload();
}

//-------------------------------------------------------------------------------------------------
void MolecularMechanicsControls::download() {
  int_data.download();
}
#endif

//-------------------------------------------------------------------------------------------------
void MolecularMechanicsControls::initializeWorkUnitGuards() {
  const size_t guard_start = work_unit_guard_offset;
  const size_t guard_end = guard_start + static_cast<size_t>(work_unit_guard_words);
  if (int_data.size() < guard_end) {
    return;
  }
  for (size_t idx = guard_start; idx < guard_end; idx++) {
    int_data.putHost(work_unit_guard_pattern, idx);
  }
#ifdef STORMM_USE_HPC
  int_data.upload(guard_start, work_unit_guard_words);
#endif
}

//-------------------------------------------------------------------------------------------------
void MolecularMechanicsControls::verifyWorkUnitGuards(const char* context) {
  const size_t guard_start = work_unit_guard_offset;
  const size_t guard_end = guard_start + static_cast<size_t>(work_unit_guard_words);
  if (int_data.size() < guard_end) {
    return;
  }
  std::array<int, work_unit_guard_words> guard_values{};
  bool copied_from_device = false;
#ifdef STORMM_USE_HPC
  const int* dev_ptr = int_data.data(HybridTargetLevel::DEVICE);
  if (dev_ptr != nullptr) {
    const int* dev_guard_ptr = dev_ptr + guard_start;
    const cudaError_t copy_status =
        cudaMemcpy(guard_values.data(), dev_guard_ptr,
                   work_unit_guard_words * sizeof(int), cudaMemcpyDeviceToHost);
    if (copy_status == cudaSuccess) {
      copied_from_device = true;
    }
    else {
#if STORMM_MMCTRL_DEBUG
      std::cerr << "DEBUG: work_unit_prog_data guard cudaMemcpy failed during "
                << (context != nullptr ? context : "unknown event")
                << " : " << cudaGetErrorString(copy_status) << " ("
                << static_cast<int>(copy_status) << ")" << std::endl;
#endif
    }
  }
#endif
  if (!copied_from_device) {
    const int* host_ptr = int_data.data();
    if (host_ptr == nullptr) {
      return;
    }
    for (size_t idx = 0; idx < static_cast<size_t>(work_unit_guard_words); idx++) {
      guard_values[idx] = host_ptr[guard_start + idx];
    }
  }
  bool corrupted = false;
  for (size_t idx = 0; idx < static_cast<size_t>(work_unit_guard_words); idx++) {
    if (guard_values[idx] != work_unit_guard_pattern) {
      corrupted = true;
      break;
    }
  }
  if (corrupted) {
#if STORMM_MMCTRL_DEBUG
    std::cerr << "DEBUG: work_unit_prog_data guard corruption detected during "
              << (context != nullptr ? context : "unknown event")
              << " guard_start=" << guard_start
              << " guard_words=" << work_unit_guard_words << std::endl;
    for (size_t idx = 0; idx < static_cast<size_t>(work_unit_guard_words); idx++) {
      std::cerr << "  guard[" << idx << "]=" << guard_values[idx] << std::endl;
    }
#endif
  }
}

//-------------------------------------------------------------------------------------------------
void MolecularMechanicsControls::rebasePointers() {
  vwu_progress.swapTarget(&int_data);
  velocity_update_progress.swapTarget(&int_data);
  velocity_constraint_progress.swapTarget(&int_data);
  position_update_progress.swapTarget(&int_data);
  geometry_constraint_progress.swapTarget(&int_data);
  nbwu_progress.swapTarget(&int_data);
  pmewu_progress.swapTarget(&int_data);
  gbrwu_progress.swapTarget(&int_data);
  gbdwu_progress.swapTarget(&int_data);
  gather_wu_progress.swapTarget(&int_data);
  scatter_wu_progress.swapTarget(&int_data);
  all_reduce_wu_progress.swapTarget(&int_data);
}

} // namespace mm
} // namespace stormm
