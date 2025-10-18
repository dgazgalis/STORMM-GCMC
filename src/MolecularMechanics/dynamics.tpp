// -*-c++-*-
#include "copyright.h"

namespace stormm {
namespace mm {
  
//-------------------------------------------------------------------------------------------------
template <typename Tcoord, typename Tcalc, typename Tcalc2, typename Tcalc4>
void dynaStep(const Tcoord* xcrd, const Tcoord* ycrd, const Tcoord* zcrd, const Tcoord* xvel,
              const Tcoord* yvel, const Tcoord* zvel, Tcoord* xfrc, Tcoord* yfrc, Tcoord* zfrc,
              Tcoord* xalt, Tcoord* yalt, Tcoord* zalt, Tcoord* vxalt, Tcoord* vyalt,
              Tcoord* vzalt, Tcoord* fxalt, Tcoord* fyalt, Tcoord* fzalt, ScoreCard *sc,
              const ThermostatWriter<Tcalc> &tstw, const ValenceKit<Tcalc> &vk,
              const NonbondedKit<Tcalc> &nbk, const ImplicitSolventKit<Tcalc> &isk,
              const NeckGeneralizedBornKit<Tcalc> &neck_gbk, Tcoord* effective_gb_radii,
              Tcoord* psi, Tcoord* sumdeijda, const RestraintKit<Tcalc, Tcalc2, Tcalc4> &rar,
              const VirtualSiteKit<Tcalc> &vsk, const ChemicalDetailsKit &cdk,
              const ConstraintKit<Tcalc> &cnk, const StaticExclusionMaskReader &ser,
              const DynamicsControls &dyncon, const int system_index,
              const Tcalc gpos_scale_factor, const Tcalc vel_scale_factor,
              const Tcalc frc_scale_factor) {

  // Evaluate the force and energy for a system in vacuum with isolated boundary conditions
  evalRestrainedMMGB<Tcoord, Tcoord,
                     Tcalc, Tcalc2, Tcalc4>(xcrd, ycrd, zcrd, nullptr, nullptr, UnitCellType::NONE,
                                            xfrc, yfrc, zfrc, sc, vk, nbk, ser, isk, neck_gbk,
                                            effective_gb_radii, psi, sumdeijda, rar,
                                            EvaluateForce::YES, system_index, tstw.step);
  transmitVirtualSiteForces<Tcoord, Tcoord, Tcalc>(xcrd, ycrd, zcrd, xfrc, yfrc, zfrc, nullptr,
                                                   nullptr, UnitCellType::NONE, vsk);

  // Find the mass array that is amenable to the template.  This is not an ideal thing to do, but
  // it will cast the float* array to Tcalc* if Tcalc is float and the double* array to Tcalc* if
  // Tcalc is double.  This requirement is the legacy of ChemicalDetailsKit originally not
  // containing the array of masses and therefore having no need to be templated in its own right.
  const bool tcalc_is_double = (std::type_index(typeid(Tcalc)).hash_code() == double_type_index);
  const Tcalc* mass_ptr = (tcalc_is_double) ? reinterpret_cast<const Tcalc*>(cdk.masses) :
                                              reinterpret_cast<const Tcalc*>(cdk.sp_masses);

  // Update the velocities by the first half step with the new forces.
  velocityVerletVelocityUpdate<Tcoord, Tcalc>(xvel, yvel, zvel, xfrc, yfrc, zfrc, cdk.natom,
                                              mass_ptr, vxalt, vyalt, vzalt, tstw, nullptr,
                                              nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                              nullptr, nullptr, 0, vel_scale_factor,
                                              frc_scale_factor);

  // Constrain velocities
  if (tstw.cnst_geom) {
    rattleVelocities<Tcoord, Tcalc>(vxalt, vyalt, vzalt, xcrd, ycrd, zcrd, cnk, tstw.dt,
                                    dyncon.getRattleTolerance(), dyncon.getRattleIterations(),
                                    dyncon.getCpuRattleMethod(), gpos_scale_factor,
                                    vel_scale_factor);
  }

  // Commit the energy, all components (energy computations are obligatory in CPU functions).  The
  // diagnostics from the initial state will always be stored.
  if (dyncon.getDiagnosticPrintFrequency() > 0 &&
      tstw.step % dyncon.getDiagnosticPrintFrequency() == 0) {
    evalKineticEnergy<Tcoord, Tcalc>(vxalt, vyalt, vzalt, sc, cdk, system_index,
                                     static_cast<Tcalc>(1.0) / vel_scale_factor);
    computeTemperature(sc, cdk, tstw.cnst_geom, system_index);
    sc->commit(StateVariable::ALL_STATES, system_index);
    sc->incrementSampleCount();
    sc->setLastTimeStep(tstw.step);
  }

  // Move particles, placing their new positions in the {x,y,z}alt arrays.
  velocityVerletCoordinateUpdate<Tcoord, Tcalc>(xcrd, ycrd, zcrd, xfrc, yfrc, zfrc, cdk.natom,
                                                mass_ptr, xalt, yalt, zalt, vxalt, vyalt, vzalt,
                                                tstw, nullptr, nullptr, nullptr, nullptr, nullptr,
                                                nullptr, nullptr, nullptr, nullptr, nullptr,
                                                nullptr, nullptr, 0, gpos_scale_factor,
                                                vel_scale_factor, frc_scale_factor);

  // Apply positional constraints
  if (tstw.cnst_geom) {
    shakePositions<Tcoord, Tcalc>(xalt, yalt, zalt, vxalt, vyalt, vzalt, xcrd, ycrd, zcrd, cnk,
                                  tstw.dt, dyncon.getRattleTolerance(),
                                  dyncon.getRattleIterations(), dyncon.getCpuRattleMethod(),
                                  gpos_scale_factor, vel_scale_factor);
  }

  // Replace virtual sites
  placeVirtualSites<Tcoord, Tcalc>(xalt, yalt, zalt, nullptr, nullptr, UnitCellType::NONE, vsk,
                                   gpos_scale_factor);

  // Zero forces in the alternate time point, in preparation for the next step.  Auxiliary arrays
  // involved in Generalized Born calculations (psi, effective_gb_radii, sumdeijda) will be
  // initialized in their respective CPU routines.
  const Tcoord zero = 0.0;
  for (int i = 0; i < cdk.natom; i++) {
    fxalt[i] = zero;
    fyalt[i] = zero;
    fzalt[i] = zero;
  }
}

//-------------------------------------------------------------------------------------------------
template <typename Tcoord, typename Tcalc, typename Tcalc2, typename Tcalc4>
void lambdaDynaStep(const Tcoord* xcrd, const Tcoord* ycrd, const Tcoord* zcrd, const Tcoord* xvel,
                    const Tcoord* yvel, const Tcoord* zvel, Tcoord* xfrc, Tcoord* yfrc, Tcoord* zfrc,
                    Tcoord* xalt, Tcoord* yalt, Tcoord* zalt, Tcoord* vxalt, Tcoord* vyalt,
                    Tcoord* vzalt, Tcoord* fxalt, Tcoord* fyalt, Tcoord* fzalt, ScoreCard *sc,
                    const ThermostatWriter<Tcalc> &tstw, const ValenceKit<Tcalc> &vk,
                    const LambdaNonbondedKit<Tcalc> &lambda_nbk, const ImplicitSolventKit<Tcalc> &isk,
                    const NeckGeneralizedBornKit<Tcalc> &neck_gbk, Tcoord* effective_gb_radii,
                    Tcoord* psi, Tcoord* sumdeijda, const RestraintKit<Tcalc, Tcalc2, Tcalc4> &rar,
                    const VirtualSiteKit<Tcalc> &vsk, const ChemicalDetailsKit &cdk,
                    const ConstraintKit<Tcalc> &cnk, const StaticExclusionMaskReader &ser,
                    const DynamicsControls &dyncon, const int system_index,
                    const Tcalc gpos_scale_factor, const Tcalc vel_scale_factor,
                    const Tcalc frc_scale_factor, const Tcalc vdw_coupling_threshold,
                    const Tcalc softcore_alpha) {

  // Evaluate the lambda-aware force and energy for a system in vacuum with isolated boundary conditions
  evalLambdaRestrainedMMGB<Tcoord, Tcoord,
                           Tcalc, Tcalc2, Tcalc4>(xcrd, ycrd, zcrd, nullptr, nullptr,
                                                  UnitCellType::NONE, xfrc, yfrc, zfrc, sc, vk,
                                                  lambda_nbk, ser, isk, neck_gbk, effective_gb_radii,
                                                  psi, sumdeijda, rar, EvaluateForce::YES,
                                                  system_index, tstw.step,
                                                  static_cast<Tcalc>(1.0) / gpos_scale_factor,
                                                  frc_scale_factor, static_cast<Tcalc>(0.0),
                                                  static_cast<Tcalc>(0.0), vdw_coupling_threshold,
                                                  softcore_alpha);
  transmitVirtualSiteForces<Tcoord, Tcoord, Tcalc>(xcrd, ycrd, zcrd, xfrc, yfrc, zfrc, nullptr,
                                                   nullptr, UnitCellType::NONE, vsk);

  // Find the mass array that is amenable to the template.  This is not an ideal thing to do, but
  // it will cast the float* array to Tcalc* if Tcalc is float and the double* array to Tcalc* if
  // Tcalc is double.  This requirement is the legacy of ChemicalDetailsKit originally not
  // containing the array of masses and therefore having no need to be templated in its own right.
  const bool tcalc_is_double = (std::type_index(typeid(Tcalc)).hash_code() == double_type_index);
  const Tcalc* mass_ptr = (tcalc_is_double) ? reinterpret_cast<const Tcalc*>(cdk.masses) :
                                              reinterpret_cast<const Tcalc*>(cdk.sp_masses);

  // Update the velocities by the first half step with the new forces.
  velocityVerletVelocityUpdate<Tcoord, Tcalc>(xvel, yvel, zvel, xfrc, yfrc, zfrc, cdk.natom,
                                              mass_ptr, vxalt, vyalt, vzalt, tstw, nullptr,
                                              nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                              nullptr, nullptr, 0, vel_scale_factor,
                                              frc_scale_factor);

  // Constrain velocities
  if (tstw.cnst_geom) {
    rattleVelocities<Tcoord, Tcalc>(vxalt, vyalt, vzalt, xcrd, ycrd, zcrd, cnk, tstw.dt,
                                    dyncon.getRattleTolerance(), dyncon.getRattleIterations(),
                                    dyncon.getCpuRattleMethod(), gpos_scale_factor,
                                    vel_scale_factor);
  }

  // Commit the energy, all components (energy computations are obligatory in CPU functions).  The
  // diagnostics from the initial state will always be stored.
  if (dyncon.getDiagnosticPrintFrequency() > 0 &&
      tstw.step % dyncon.getDiagnosticPrintFrequency() == 0) {
    evalKineticEnergy<Tcoord, Tcalc>(vxalt, vyalt, vzalt, sc, cdk, system_index,
                                     static_cast<Tcalc>(1.0) / vel_scale_factor);
    computeTemperature(sc, cdk, tstw.cnst_geom, system_index);
    sc->commit(StateVariable::ALL_STATES, system_index);
    sc->incrementSampleCount();
    sc->setLastTimeStep(tstw.step);
  }

  // Move particles, placing their new positions in the {x,y,z}alt arrays.
  velocityVerletCoordinateUpdate<Tcoord, Tcalc>(xcrd, ycrd, zcrd, xfrc, yfrc, zfrc, cdk.natom,
                                                mass_ptr, xalt, yalt, zalt, vxalt, vyalt, vzalt,
                                                tstw, nullptr, nullptr, nullptr, nullptr, nullptr,
                                                nullptr, nullptr, nullptr, nullptr, nullptr,
                                                nullptr, nullptr, 0, gpos_scale_factor,
                                                vel_scale_factor, frc_scale_factor);

  // Apply positional constraints
  if (tstw.cnst_geom) {
    shakePositions<Tcoord, Tcalc>(xalt, yalt, zalt, vxalt, vyalt, vzalt, xcrd, ycrd, zcrd, cnk,
                                  tstw.dt, dyncon.getRattleTolerance(),
                                  dyncon.getRattleIterations(), dyncon.getCpuRattleMethod(),
                                  gpos_scale_factor, vel_scale_factor);
  }

  // Replace virtual sites
  placeVirtualSites<Tcoord, Tcalc>(xalt, yalt, zalt, nullptr, nullptr, UnitCellType::NONE, vsk,
                                   gpos_scale_factor);

  // Zero forces in the alternate time point, in preparation for the next step.  Auxiliary arrays
  // involved in Generalized Born calculations (psi, effective_gb_radii, sumdeijda) will be
  // initialized in their respective CPU routines.
  const Tcoord zero = 0.0;
  for (int i = 0; i < cdk.natom; i++) {
    fxalt[i] = zero;
    fyalt[i] = zero;
    fzalt[i] = zero;
  }
}

//-------------------------------------------------------------------------------------------------
template <typename Tcoord, typename Tacc, typename Tcoord4,
          typename Tcalc, typename Tcalc2, typename Tcalc4>
void dynaStep(PsSynthesisWriter *poly_psw, CellGridWriter<void, void, void, void> *cgw_v,
              ScoreCard *sc, ThermostatWriter<Tcalc> *tstw,
              const SyValenceKit<Tcalc> &poly_vk, const SyNonbondedKit<Tcalc, Tcalc2> &poly_nbk,
              const SyRestraintKit<Tcalc, Tcalc2, Tcalc4> &poly_rk,
              const SyAtomUpdateKit<Tcalc, Tcalc2, Tcalc4> &poly_auk,
              const LocalExclusionMaskReader &lemr, const Tcalc cutoff, const Tcalc qqew_coeff,
              const VdwSumMethod vdw_sum, const int ntpr) {

  // Create read-only forms of the coordinate synthesis and thermostat abstracts
  const PsSynthesisReader poly_psr(poly_psw);
  const ThermostatReader<Tcalc> tstr(tstw);

  // Produce a mutable energy tracking abstract
  ScoreCardWriter scw = sc->data();
  
  // Compute the non-bonded particle-particle interactions
  evaluateParticleParticleEnergy<Tcoord, Tacc,
                                 Tcalc, Tcalc2, Tcoord4>(cgw_v, &scw, poly_psr, poly_nbk, lemr,
                                                         cutoff, qqew_coeff, vdw_sum,
                                                         EvaluateForce::YES, NonbondedTheme::ALL);

  // Contribute the non-bonded forces to the synthesis accumulators
  const CellGridReader<Tcoord, Tacc, Tcalc, Tcoord4> cgr = restoreType<Tcoord, Tacc,
                                                                       Tcalc, Tcoord4>(cgw_v);
  contributeCellGridForces<Tcoord, Tacc, Tcalc, Tcoord4>(poly_psw, cgr);

  // Compute valence interactions
  evalValeRestMM<Tcalc, Tcalc2, Tcalc4>(poly_psw, sc, poly_vk, poly_rk, poly_auk,
                                        EvaluateForce::YES, VwuTask::ALL_TASKS, tstw->step);
  
   // Transmit virtual site forces to frame atoms, if applicable.
  transmitVirtualSiteForces<Tcalc, Tcalc2, Tcalc4>(poly_psw, poly_vk, poly_auk);
  
  // Velocity Verlet update I
  velocityVerletVelocityUpdate<Tcalc, Tcalc2, Tcalc4>(poly_psw, poly_auk, tstr);

  // Apply velocity constraints
  if (tstr.cnst_geom) {
    rattleVelocities<Tcalc, Tcalc2, Tcalc4>(poly_psw, poly_vk, poly_auk, tstr.dt, tstr.rattle_tol,
                                            tstr.rattle_iter);
    settleVelocities<Tcalc, Tcalc2, Tcalc4>(poly_psw, poly_vk, poly_auk);
  }
  if (ntpr > 0 && tstw->step % ntpr == 0) {
    evalKineticEnergy<Tcalc, Tcalc2, Tcalc4, Tcalc>(poly_psr, sc, poly_auk);
    computeTemperature<Tcalc, Tcalc2, Tcalc4, Tcalc>(poly_psr, sc, poly_auk, tstr, true);
    sc->commit(StateVariable::ALL_STATES);
    sc->incrementSampleCount();
    sc->setLastTimeStep(tstw->step);
  }
  
  // Velocity Verlet update II
  velocityVerletCoordinateUpdate<Tcalc, Tcalc2, Tcalc4>(poly_psw, poly_auk, tstr);
  
  // Apply position constraints
  if (tstr.cnst_geom) {
    shakePositions<Tcalc>(poly_psw, poly_vk, poly_auk, tstr.dt, tstr.rattle_tol,
                          tstr.rattle_iter);
    settlePositions<Tcalc, Tcalc2, Tcalc4>(poly_psw, poly_vk, poly_auk, tstr.dt);
  }

  // Replace virtual sites
  placeVirtualSites<Tcalc, Tcalc2, Tcalc4>(poly_psw, poly_vk, poly_auk);
  
  // Migrate particles within the cell grid
  CellGridWriter<Tcoord, Tacc, Tcalc, Tcoord4> cgw = restoreType<Tcoord, Tacc,
                                                                 Tcoord, Tcoord4>(cgw_v);
  migrate(&cgw, poly_psr);
}

//-------------------------------------------------------------------------------------------------
template <typename Tcoord, typename Tacc, typename Tcalc, typename Tcoord4>
void dynamics(PhaseSpaceSynthesis *poly_ps, CellGrid<Tcoord, Tacc, Tcalc, Tcoord4> *cg,
              ScoreCard *sc, Thermostat *heat_bath, const AtomGraphSynthesis &poly_ag,
              const LocalExclusionMask &lem, const DynamicsControls &dyncon,
              const PrecisionControls &preccon, const PPPMControls &pmecon) {

  // Produce abstracts of the relevant abstracts, at each point in the coordinate time cycle
  const CoordinateCycle poly_ps_next_stage = getNextCyclePosition(poly_ps->getCyclePosition());
  PsSynthesisWriter poly_psw = poly_ps->data();
  PsSynthesisWriter poly_psw_alt = poly_ps->data(poly_ps_next_stage);
  ScoreCardWriter scw = sc->data();
  const LocalExclusionMaskReader lemr = lem.data();
  const CoordinateCycle cg_next_stage = getNextCyclePosition(cg->getCyclePosition());
  CellGridWriter<void, void, void, void> cgv = cg->templateFreeData();
  CellGridWriter<void, void, void, void> cgv_alt = cg->templateFreeData(cg_next_stage);
  MotionSweeper mos(poly_ps);
  const int traj_freq = dyncon.getTrajectoryPrintFrequency();
  const int diag_freq = dyncon.getDiagnosticPrintFrequency();
  const int cmpg_freq = dyncon.getCenterOfMassMotionPurgeFrequency();

  // Cutoffs and critical non-bonded constants.  Selection of the Ewald coefficient would be done
  // for the setup of objects used in the GPU workflow, but on the CPU the Ewald coefficient can be
  // extracted 
  const double cutoff = cg->getCutoff();
  const double dsum_tol = pmecon.getDirectSumTolerance();
  const double ew_coeff = ewaldCoefficient(cutoff, dsum_tol);
  const VdwSumMethod vdw_sum = dyncon.getVdwSummation();
  
  // Produce critical abstracts for other objects as needed
  switch (preccon.getValenceMethod()) {
  case PrecisionModel::DOUBLE:
    {
      ThermostatWriter<double> tstw = heat_bath->dpData();
      const SyValenceKit<double> poly_vk = poly_ag.getDoublePrecisionValenceKit();
      const SyNonbondedKit<double, double2> poly_nbk = poly_ag.getDoublePrecisionNonbondedKit();
      const SyAtomUpdateKit<double,
                            double2, double4> poly_auk = poly_ag.getDoublePrecisionAtomUpdateKit();
      const SyRestraintKit<double,
                           double2, double4> poly_rk = poly_ag.getDoublePrecisionRestraintKit();
      for (int step = 0; step < dyncon.getStepCount(); step++) {

        // If the thermostat's random number cache has, by this time, been used up, refresh it.
        if (step % tstw.depth == 0 && step > 0) {
          heat_bath->refresh(0, tstw.padded_natom);
        }

        // If requested, purge motion of the center of mass for each system.
        if (cmpg_freq > 0 && step > 0 && step % cmpg_freq == 0) {
          removeMomentum(poly_ps, poly_ag, &mos);
        }

        // Initialize forces and energy accumulators.
        poly_ps->initializeForces();
        cg->initializeForces();
        sc->initialize();
        
        // Perform the step.
        if (step & 0x1) {
          dynaStep<double, llint, double4,
                   double, double2, double4>(&poly_psw_alt, &cgv_alt, sc, &tstw, poly_vk,
                                             poly_nbk, poly_rk, poly_auk, lemr, cutoff, ew_coeff,
                                             vdw_sum, diag_freq);
        }
        else {
          dynaStep<double, llint, double4,
                   double, double2, double4>(&poly_psw, &cgv, sc, &tstw, poly_vk, poly_nbk,
                                             poly_rk, poly_auk, lemr, cutoff, ew_coeff, vdw_sum,
                                             diag_freq);
        }

        // Update the cycle positions and time step.
        poly_ps->updateCyclePosition();
        cg->updateCyclePosition();
        tstw.step += 1;
        heat_bath->incrementStep();
      }
    }
    break;
  case PrecisionModel::SINGLE:
    {
      ThermostatWriter<float> tstw = heat_bath->spData();
      const SyValenceKit<float> poly_vk = poly_ag.getSinglePrecisionValenceKit();
      const SyNonbondedKit<float, float2> poly_nbk = poly_ag.getSinglePrecisionNonbondedKit();
      const SyAtomUpdateKit<float,
                            float2, float4> poly_auk = poly_ag.getSinglePrecisionAtomUpdateKit();
      const SyRestraintKit<float,
                           float2, float4> poly_rk = poly_ag.getSinglePrecisionRestraintKit();
      for (int step = 0; step < dyncon.getStepCount(); step++) {

        // If the thermostat's random number cache has, by this time, been used up, refresh it.
        if (step % tstw.depth == 0 && step > 0) {
          heat_bath->refresh(0, tstw.padded_natom);
        }

        // If requested, purge motion of the center of mass for each system.
        if (cmpg_freq > 0 && step > 0 && step % cmpg_freq == 0) {
          removeMomentum(poly_ps, poly_ag, &mos);
        }

        // Initialize forces and energy accumulators.
        poly_ps->initializeForces();
        cg->initializeForces();
        sc->initialize();

        // Perform the step.
        if (step & 0x1) {
          dynaStep<float, int, float4,
                   float, float2, float4>(&poly_psw_alt, &cgv_alt, sc, &tstw, poly_vk, poly_nbk,
                                          poly_rk, poly_auk, lemr, cutoff, ew_coeff, vdw_sum,
                                          diag_freq);
        }
        else {
          dynaStep<float, int, float4,
                   float, float2, float4>(&poly_psw, &cgv, sc, &tstw, poly_vk, poly_nbk, poly_rk,
                                          poly_auk, lemr, cutoff, ew_coeff, vdw_sum, diag_freq);
        }

        // Update the cycle positions and time step.
        poly_ps->updateCyclePosition();
        cg->updateCyclePosition();
        tstw.step += 1;
        heat_bath->incrementStep();
        
      }
    }
    break;
  }
}

} // namespace mm
} // namespace stormm
