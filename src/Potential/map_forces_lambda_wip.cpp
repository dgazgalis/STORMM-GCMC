// -*-c++-*-
#include "copyright.h"
#include "DataTypes/common_types.h"
#include "map_forces_lambda_wip.h"

namespace stormm {
namespace energy {

using data_types::double_type_index;
using data_types::float_type_index;
using data_types::int_type_index;
using data_types::llint_type_index;
using data_types::double4_type_index;
using data_types::float4_type_index;
using data_types::int4_type_index;

//-------------------------------------------------------------------------------------------------
void mapForcesLambda(PMIGrid *pm, const AtomGraphSynthesis *poly_ag,
                     PhaseSpaceSynthesis *ps_synthesis,
                     const double* lambda_ele, const double lambda_threshold) {
  // Extract the cell grid type information from the PMIGrid
  const size_t cg_tmat = pm->getCellGridMatrixTypeID();
  const size_t cg_tacc = pm->getCellGridAccumulatorTypeID();
  const size_t cg_tcalc = pm->getCellGridCalculationTypeID();
  const size_t cg_tcoord = pm->getCellGridCoordinateTypeID();

  // Unroll the template calls based on the coordinate type
  if (cg_tmat == double_type_index) {
    if (cg_tcoord == double4_type_index) {
      unrollMapForcesLambdaCall<double, double4>(pm, cg_tacc, cg_tcalc, poly_ag, ps_synthesis,
                                                 lambda_ele, lambda_threshold);
    }
    else if (cg_tcoord == float4_type_index) {
      unrollMapForcesLambdaCall<double, float4>(pm, cg_tacc, cg_tcalc, poly_ag, ps_synthesis,
                                                lambda_ele, lambda_threshold);
    }
    else if (cg_tcoord == int4_type_index) {
      unrollMapForcesLambdaCall<double, int4>(pm, cg_tacc, cg_tcalc, poly_ag, ps_synthesis,
                                              lambda_ele, lambda_threshold);
    }
    else {
      rtErr("Unrecognized coordinate tuple type " + std::to_string(cg_tcoord) + " in CellGrid "
            "object referenced by the PMIGrid.", "mapForcesLambda");
    }
  }
  else if (cg_tmat == float_type_index) {
    if (cg_tcoord == double4_type_index) {
      unrollMapForcesLambdaCall<float, double4>(pm, cg_tacc, cg_tcalc, poly_ag, ps_synthesis,
                                                lambda_ele, lambda_threshold);
    }
    else if (cg_tcoord == float4_type_index) {
      unrollMapForcesLambdaCall<float, float4>(pm, cg_tacc, cg_tcalc, poly_ag, ps_synthesis,
                                               lambda_ele, lambda_threshold);
    }
    else if (cg_tcoord == int4_type_index) {
      unrollMapForcesLambdaCall<float, int4>(pm, cg_tacc, cg_tcalc, poly_ag, ps_synthesis,
                                             lambda_ele, lambda_threshold);
    }
    else {
      rtErr("Unrecognized coordinate tuple type " + std::to_string(cg_tcoord) + " in CellGrid "
            "object referenced by the PMIGrid.", "mapForcesLambda");
    }
  }
  else if (cg_tmat == int_type_index) {
    if (cg_tcoord == double4_type_index) {
      unrollMapForcesLambdaCall<int, double4>(pm, cg_tacc, cg_tcalc, poly_ag, ps_synthesis,
                                              lambda_ele, lambda_threshold);
    }
    else if (cg_tcoord == float4_type_index) {
      unrollMapForcesLambdaCall<int, float4>(pm, cg_tacc, cg_tcalc, poly_ag, ps_synthesis,
                                             lambda_ele, lambda_threshold);
    }
    else if (cg_tcoord == int4_type_index) {
      unrollMapForcesLambdaCall<int, int4>(pm, cg_tacc, cg_tcalc, poly_ag, ps_synthesis,
                                           lambda_ele, lambda_threshold);
    }
    else {
      rtErr("Unrecognized coordinate tuple type " + std::to_string(cg_tcoord) + " in CellGrid "
            "object referenced by the PMIGrid.", "mapForcesLambda");
    }
  }
  else if (cg_tmat == llint_type_index) {
    if (cg_tcoord == double4_type_index) {
      unrollMapForcesLambdaCall<llint, double4>(pm, cg_tacc, cg_tcalc, poly_ag, ps_synthesis,
                                                lambda_ele, lambda_threshold);
    }
    else if (cg_tcoord == float4_type_index) {
      unrollMapForcesLambdaCall<llint, float4>(pm, cg_tacc, cg_tcalc, poly_ag, ps_synthesis,
                                               lambda_ele, lambda_threshold);
    }
    else if (cg_tcoord == int4_type_index) {
      unrollMapForcesLambdaCall<llint, int4>(pm, cg_tacc, cg_tcalc, poly_ag, ps_synthesis,
                                             lambda_ele, lambda_threshold);
    }
    else {
      rtErr("Unrecognized coordinate tuple type " + std::to_string(cg_tcoord) + " in CellGrid "
            "object referenced by the PMIGrid.", "mapForcesLambda");
    }
  }
  else {
    rtErr("Unrecognized base coordinate type " + std::to_string(cg_tmat) + " in CellGrid object "
          "referenced by the PMIGrid.", "mapForcesLambda");
  }
}

//-------------------------------------------------------------------------------------------------
void mapForcesLambda(PMIGrid *pm, const AtomGraphSynthesis &poly_ag,
                     PhaseSpaceSynthesis &ps_synthesis,
                     const double* lambda_ele, const double lambda_threshold) {
  mapForcesLambda(pm, poly_ag.getSelfPointer(), &ps_synthesis,
                 lambda_ele, lambda_threshold);
}

//-------------------------------------------------------------------------------------------------
void mapForcesLambda(PMIGrid &pm, const AtomGraphSynthesis *poly_ag,
                     PhaseSpaceSynthesis *ps_synthesis,
                     const double* lambda_ele, const double lambda_threshold) {
  mapForcesLambda(&pm, poly_ag, ps_synthesis,
                 lambda_ele, lambda_threshold);
}

//-------------------------------------------------------------------------------------------------
void mapForcesLambda(PMIGrid &pm, const AtomGraphSynthesis &poly_ag,
                     PhaseSpaceSynthesis &ps_synthesis,
                     const double* lambda_ele, const double lambda_threshold) {
  mapForcesLambda(&pm, poly_ag.getSelfPointer(), &ps_synthesis,
                 lambda_ele, lambda_threshold);
}

} // namespace energy
} // namespace stormm