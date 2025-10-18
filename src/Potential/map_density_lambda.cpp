#include "copyright.h"
#include "map_density_lambda.h"

namespace stormm {
namespace energy {

//-------------------------------------------------------------------------------------------------
void mapDensityLambda(PMIGrid *pm, const AtomGraphSynthesis *poly_ag,
                      const double* lambda_ele, const double lambda_threshold) {

  // Extract the cell grid pointer and unroll its templating
  const size_t cg_tmat  = pm->getCellGridMatrixTypeID();
  const size_t cg_tacc  = pm->getCellGridAccumulatorTypeID();
  const size_t cg_tcalc = pm->getCellGridCalculationTypeID();

  // Unroll based on matrix type (determines coordinate tuple type)
  if (cg_tmat == double_type_index) {
    unrollMapDensityLambdaCall<double, double4>(pm, cg_tacc, cg_tcalc, poly_ag, lambda_ele,
                                                lambda_threshold);
  }
  else if (cg_tmat == float_type_index) {
    unrollMapDensityLambdaCall<float, float4>(pm, cg_tacc, cg_tcalc, poly_ag, lambda_ele,
                                              lambda_threshold);
  }
  else if (cg_tmat == llint_type_index) {
    unrollMapDensityLambdaCall<llint, llint4>(pm, cg_tacc, cg_tcalc, poly_ag, lambda_ele,
                                              lambda_threshold);
  }
  else if (cg_tmat == int_type_index) {
    unrollMapDensityLambdaCall<int, int4>(pm, cg_tacc, cg_tcalc, poly_ag, lambda_ele,
                                          lambda_threshold);
  }
}

//-------------------------------------------------------------------------------------------------
void mapDensityLambda(PMIGrid *pm, const AtomGraphSynthesis &poly_ag,
                      const double* lambda_ele, const double lambda_threshold) {
  mapDensityLambda(pm, poly_ag.getSelfPointer(), lambda_ele, lambda_threshold);
}

} // namespace energy
} // namespace stormm
