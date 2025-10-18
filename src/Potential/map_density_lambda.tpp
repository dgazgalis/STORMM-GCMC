// -*-c++-*-
#include "copyright.h"
#include "map_density.h"

namespace stormm {
namespace energy {

//-------------------------------------------------------------------------------------------------
template <typename T, typename Tacc, typename Tcalc, typename Tcalc2, typename T4>
void accumulateCellDensityLambda(PMIGridWriter *pm_wrt, const int sysid, const int cell_i,
                                 const int cell_j, const int cell_k,
                                 const CellGridReader<T, Tacc, Tcalc, T4> &cgr,
                                 const SyNonbondedKit<Tcalc, Tcalc2> &synbk,
                                 const double* lambda_ele,
                                 const double lambda_threshold) {

  // Re-derive the cell and cell grid boundaries (same as standard version)
  const ullint cell_bounds = cgr.system_cell_grids[sysid];
  const int cell_offset = (cell_bounds & 0xfffffffLLU);
  const int cell_na = ((cell_bounds >> 28) & 0xfffLLU);
  const int cell_nb = ((cell_bounds >> 40) & 0xfffLLU);
  const int cell_nc = ((cell_bounds >> 52) & 0xfffLLU);

  // Determine limits and other critical constants
  const bool coord_in_real = (cgr.lpos_scale < 1.01);
  const bool tcalc_is_double = (std::type_index(typeid(Tcalc)).hash_code() == double_type_index);
  const size_t ijk_cellidx = cell_offset + (((cell_k * cell_nb) + cell_j) * cell_na) + cell_i;
  const uint2 ijk_bounds = cgr.cell_limits[ijk_cellidx];
  const uint mllim = ijk_bounds.x;
  const uint mhlim = mllim + (ijk_bounds.y >> 16);
  const int xfrm_stride = roundUp(9, warp_size_int);
  Tcalc umat[9];
  for (int i = 0; i < 9; i++) {
    umat[i] = cgr.system_cell_umat[(sysid * xfrm_stride) + i];
  }

  // Get system atom offset for lambda indexing
  const int system_atom_offset = synbk.atom_offsets[sysid];

  // Lay out arrays to collect B-spline coefficients
  const uint4 grid_dims = pm_wrt->dims[sysid];
  switch (pm_wrt->mode) {
  case PrecisionModel::DOUBLE:
    {
      std::vector<double> a_cof(pm_wrt->order), b_cof(pm_wrt->order), c_cof(pm_wrt->order);
      for (uint m = mllim; m < mhlim; m++) {
        const T4 atom_m = cgr.image[m];

        // NEW: Get atom index for lambda lookup
        // Extract the local atom index from atom_m.w, then compute global index
        const int local_atom_idx = sourceIndex<T>(pm_wrt->theme, cgr.theme, atom_m.w, coord_in_real);
        const int global_atom_idx = system_atom_offset + local_atom_idx;

        // NEW: Get lambda value and check threshold
        const double lambda = lambda_ele[global_atom_idx];
        if (lambda < lambda_threshold) {
          continue;  // Skip ghost atoms - they contribute zero charge
        }

        // Compute B-spline alignment (same as standard version)
        int grid_root_a, grid_root_b, grid_root_c;
        particleAlignment<Tcalc, double>(atom_m.x, atom_m.y, atom_m.z, cgr.inv_lpos_scale, umat,
                                         cgr.mesh_ticks, cell_i, cell_j, cell_k, a_cof.data(),
                                         b_cof.data(), c_cof.data(), pm_wrt->order, &grid_root_a,
                                         &grid_root_b, &grid_root_c);

        // Get atomic charge (same as standard version)
        const double q = sourceMagnitude<T, Tcalc, Tcalc2>(pm_wrt->theme, cgr.theme, atom_m.w,
                                                           coord_in_real, sysid, synbk);

        // NEW: Scale charge by lambda
        const double scaled_q = q * lambda;

        // Scale B-spline coefficients by lambda-scaled charge
        for (int i = 0; i < pm_wrt->order; i++) {
          a_cof[i] *= scaled_q;  // Was: a_cof[i] *= q
        }

        // Spread to grid (same as standard version)
        spreadDensity<double, double>(a_cof.data(), b_cof.data(), c_cof.data(), pm_wrt->order,
                                      grid_root_a, grid_root_b, grid_root_c, pm_wrt->dims[sysid],
                                      pm_wrt->fftm, pm_wrt->ddata);
      }
    }
    break;
  case PrecisionModel::SINGLE:
    {
      std::vector<float> a_cof(pm_wrt->order), b_cof(pm_wrt->order), c_cof(pm_wrt->order);
      for (uint m = mllim; m < mhlim; m++) {
        const T4 atom_m = cgr.image[m];

        // NEW: Get lambda and check threshold (same logic as double precision case)
        const int local_atom_idx = sourceIndex<T>(pm_wrt->theme, cgr.theme, atom_m.w, coord_in_real);
        const int global_atom_idx = system_atom_offset + local_atom_idx;
        const double lambda = lambda_ele[global_atom_idx];
        if (lambda < lambda_threshold) {
          continue;
        }

        // Compute alignment
        int grid_root_a, grid_root_b, grid_root_c;
        particleAlignment<Tcalc, float>(atom_m.x, atom_m.y, atom_m.z, cgr.inv_lpos_scale, umat,
                                        cgr.mesh_ticks, cell_i, cell_j, cell_k, a_cof.data(),
                                        b_cof.data(), c_cof.data(), pm_wrt->order, &grid_root_a,
                                        &grid_root_b, &grid_root_c);

        // Get charge and scale by lambda
        const float q = sourceMagnitude<T, Tcalc, Tcalc2>(pm_wrt->theme, cgr.theme, atom_m.w,
                                                          coord_in_real, sysid, synbk);
        const float scaled_q = q * static_cast<float>(lambda);

        for (int i = 0; i < pm_wrt->order; i++) {
          a_cof[i] *= scaled_q;
        }

        // Spread to grid
        spreadDensity<float, float>(a_cof.data(), b_cof.data(), c_cof.data(), pm_wrt->order,
                                    grid_root_a, grid_root_b, grid_root_c, pm_wrt->dims[sysid],
                                    pm_wrt->fftm, pm_wrt->fdata);
      }
    }
    break;
  }
}

//-------------------------------------------------------------------------------------------------
template <typename T, typename Tacc, typename Tcalc, typename Tcalc2, typename T4>
void accumulateCellDensityLambda(PMIGridAccumulator *pm_acc, const int sysid, const int cell_i,
                                 const int cell_j, const int cell_k,
                                 const CellGridReader<T, Tacc, Tcalc, T4> &cgr,
                                 const SyNonbondedKit<Tcalc, Tcalc2> &synbk,
                                 const double* lambda_ele,
                                 const double lambda_threshold) {

  // Re-derive the cell and cell grid boundaries
  const ullint cell_bounds = cgr.system_cell_grids[sysid];
  const int cell_offset = (cell_bounds & 0xfffffffLLU);
  const int cell_na = ((cell_bounds >> 28) & 0xfffLLU);
  const int cell_nb = ((cell_bounds >> 40) & 0xfffLLU);
  const int cell_nc = ((cell_bounds >> 52) & 0xfffLLU);

  // Determine limits and constants
  const bool coord_in_real = (cgr.lpos_scale < 1.01);
  const bool tcalc_is_double = (std::type_index(typeid(Tcalc)).hash_code() == double_type_index);
  const size_t ijk_cellidx = cell_offset + (((cell_k * cell_nb) + cell_j) * cell_na) + cell_i;
  const uint2 ijk_bounds = cgr.cell_limits[ijk_cellidx];
  const uint mllim = ijk_bounds.x;
  const uint mhlim = mllim + (ijk_bounds.y >> 16);
  const int xfrm_stride = roundUp(9, warp_size_int);
  Tcalc umat[9];
  for (int i = 0; i < 9; i++) {
    umat[i] = cgr.system_cell_umat[(sysid * xfrm_stride) + i];
  }

  const int system_atom_offset = synbk.atom_offsets[sysid];

  // Fixed-precision accumulation
  const uint4 grid_dims = pm_acc->dims[sysid];
  switch (pm_acc->mode) {
  case PrecisionModel::DOUBLE:
    {
      std::vector<double> a_cof(pm_acc->order), b_cof(pm_acc->order), c_cof(pm_acc->order);
      for (uint m = mllim; m < mhlim; m++) {
        const T4 atom_m = cgr.image[m];

        // Lambda threshold check
        const int local_atom_idx = sourceIndex<T>(pm_acc->theme, cgr.theme, atom_m.w, coord_in_real);
        const int global_atom_idx = system_atom_offset + local_atom_idx;
        const double lambda = lambda_ele[global_atom_idx];
        if (lambda < lambda_threshold) {
          continue;
        }

        int grid_root_a, grid_root_b, grid_root_c;
        particleAlignment<Tcalc, double>(atom_m.x, atom_m.y, atom_m.z, cgr.inv_lpos_scale, umat,
                                         cgr.mesh_ticks, cell_i, cell_j, cell_k, a_cof.data(),
                                         b_cof.data(), c_cof.data(), pm_acc->order, &grid_root_a,
                                         &grid_root_b, &grid_root_c);

        const double q = sourceMagnitude<T, Tcalc, Tcalc2>(pm_acc->theme, cgr.theme, atom_m.w,
                                                           coord_in_real, sysid, synbk);
        const double scaled_q = q * lambda;

        for (int i = 0; i < pm_acc->order; i++) {
          a_cof[i] *= scaled_q;
        }

        // Spread with fixed-precision accumulation
        spreadDensity<double, llint>(a_cof.data(), b_cof.data(), c_cof.data(), pm_acc->order,
                                     grid_root_a, grid_root_b, grid_root_c, pm_acc->dims[sysid],
                                     pm_acc->fftm, pm_acc->lldata, pm_acc->overflow,
                                     pm_acc->fp_scale);
      }
    }
    break;
  case PrecisionModel::SINGLE:
    {
      std::vector<float> a_cof(pm_acc->order), b_cof(pm_acc->order), c_cof(pm_acc->order);
      for (uint m = mllim; m < mhlim; m++) {
        const T4 atom_m = cgr.image[m];

        const int local_atom_idx = sourceIndex<T>(pm_acc->theme, cgr.theme, atom_m.w, coord_in_real);
        const int global_atom_idx = system_atom_offset + local_atom_idx;
        const double lambda = lambda_ele[global_atom_idx];
        if (lambda < lambda_threshold) {
          continue;
        }

        int grid_root_a, grid_root_b, grid_root_c;
        particleAlignment<Tcalc, float>(atom_m.x, atom_m.y, atom_m.z, cgr.inv_lpos_scale, umat,
                                        cgr.mesh_ticks, cell_i, cell_j, cell_k, a_cof.data(),
                                        b_cof.data(), c_cof.data(), pm_acc->order, &grid_root_a,
                                        &grid_root_b, &grid_root_c);

        const float q = sourceMagnitude<T, Tcalc, Tcalc2>(pm_acc->theme, cgr.theme, atom_m.w,
                                                          coord_in_real, sysid, synbk);
        const float scaled_q = q * static_cast<float>(lambda);

        for (int i = 0; i < pm_acc->order; i++) {
          a_cof[i] *= scaled_q;
        }

        spreadDensity<float, int>(a_cof.data(), b_cof.data(), c_cof.data(), pm_acc->order,
                                  grid_root_a, grid_root_b, grid_root_c, pm_acc->dims[sysid],
                                  pm_acc->fftm, pm_acc->idata, pm_acc->overflow, pm_acc->fp_scale);
      }
    }
    break;
  }
}

//-------------------------------------------------------------------------------------------------
template <typename T, typename Tacc, typename Tcalc, typename T4>
void mapDensityLambda(PMIGrid *pm, const CellGrid<T, Tacc, Tcalc, T4> *cg,
                      const AtomGraphSynthesis *poly_ag,
                      const double* lambda_ele,
                      const double lambda_threshold) {

  // Match themes (ensure PMI grid is for electrostatics)
  matchThemes(pm->getTheme(), cg->getTheme());

  // Get abstracts (same as standard mapDensity)
  const bool tcalc_is_double = (std::type_index(typeid(Tcalc)).hash_code() == double_type_index);
  const bool tcrd_is_real = isFloatingPointScalarType<T>();

  // Get non-bonded kits at appropriate precision
  const SyNonbondedKit<double, double2> dsynbk = poly_ag->getDoublePrecisionNonbondedKit();
  const SyNonbondedKit<float, float2> fsynbk = poly_ag->getSinglePrecisionNonbondedKit();
  const CellGridReader<void, void, void, void> cgr_v = cg->templateFreeData();

  // Create properly typed cell grid readers
  const CellGridReader<T, Tacc, double, T4> dcgr = restoreType<T, Tacc, double, T4>(cgr_v);
  const CellGridReader<T, Tacc, float, T4>  fcgr = restoreType<T, Tacc, float, T4>(cgr_v);

  const bool acc_density_in_real = (pm->fixedPrecisionEnabled() == false);

  // Initialize the grids
  pm->initialize();

  // Loop over all systems in the synthesis
  for (int sysid = 0; sysid < dsynbk.nsys; sysid++) {
    const ullint cell_bounds = (tcalc_is_double) ? dcgr.system_cell_grids[sysid] :
                                                   fcgr.system_cell_grids[sysid];
    const int cell_offset = (cell_bounds & 0xfffffffLLU);
    const int cell_na = ((cell_bounds >> 28) & 0xfffLLU);
    const int cell_nb = ((cell_bounds >> 40) & 0xfffLLU);
    const int cell_nc = ((cell_bounds >> 52) & 0xfffLLU);

    if (acc_density_in_real) {
      // Real-valued accumulation
      PMIGridWriter pm_wrt = pm->data();

      for (int i = 0; i < cell_na; i++) {
        for (int j = 0; j < cell_nb; j++) {
          for (int k = 0; k < cell_nc; k++) {

            if (tcalc_is_double) {
              accumulateCellDensityLambda<T, Tacc, double, double2, T4>(&pm_wrt, sysid, i, j, k,
                                                                         dcgr, dsynbk, lambda_ele,
                                                                         lambda_threshold);
            }
            else {
              accumulateCellDensityLambda<T, Tacc, float, float2, T4>(&pm_wrt, sysid, i, j, k,
                                                                       fcgr, fsynbk, lambda_ele,
                                                                       lambda_threshold);
            }
          }
        }
      }
    }
    else {
      // Fixed-precision accumulation
      PMIGridAccumulator pm_acc = pm->fpData();

      for (int i = 0; i < cell_na; i++) {
        for (int j = 0; j < cell_nb; j++) {
          for (int k = 0; k < cell_nc; k++) {

            if (tcalc_is_double) {
              accumulateCellDensityLambda<T, Tacc, double, double2, T4>(&pm_acc, sysid, i, j, k,
                                                                         dcgr, dsynbk, lambda_ele,
                                                                         lambda_threshold);
            }
            else {
              accumulateCellDensityLambda<T, Tacc, float, float2, T4>(&pm_acc, sysid, i, j, k,
                                                                       fcgr, fsynbk, lambda_ele,
                                                                       lambda_threshold);
            }
          }
        }
      }
    }
  }

  // Convert grid to real format if needed
  pm->convertToReal();
}

//-------------------------------------------------------------------------------------------------
template <typename T, typename Tacc, typename Tcalc, typename T4>
void mapDensityLambda(PMIGrid *pm, const CellGrid<T, Tacc, Tcalc, T4> &cg,
                      const AtomGraphSynthesis &poly_ag,
                      const double* lambda_ele,
                      const double lambda_threshold) {
  mapDensityLambda(pm, cg.getSelfPointer(), poly_ag.getSelfPointer(), lambda_ele, lambda_threshold);
}

//-------------------------------------------------------------------------------------------------
template <typename T, typename T4>
void unrollMapDensityLambdaCall(PMIGrid *pm, const size_t cg_tacc, const size_t cg_tcalc,
                                const AtomGraphSynthesis *poly_ag,
                                const double* lambda_ele,
                                const double lambda_threshold) {
  if (cg_tacc == int_type_index) {
    unrollMapDensityLambdaCall<T, int, T4>(pm, cg_tcalc, poly_ag, lambda_ele, lambda_threshold);
  }
  else if (cg_tacc == llint_type_index) {
    unrollMapDensityLambdaCall<T, llint, T4>(pm, cg_tcalc, poly_ag, lambda_ele, lambda_threshold);
  }
}

//-------------------------------------------------------------------------------------------------
template <typename T, typename Tacc, typename T4>
void unrollMapDensityLambdaCall(PMIGrid *pm, const size_t cg_tcalc,
                                const AtomGraphSynthesis *poly_ag,
                                const double* lambda_ele,
                                const double lambda_threshold) {
  if (cg_tcalc == double_type_index) {
    const CellGrid<T, Tacc, double, T4>* cgp = pm->getCellGridPointer<T, Tacc, double, T4>();
    mapDensityLambda<T, Tacc, double, T4>(pm, cgp, poly_ag, lambda_ele, lambda_threshold);
  }
  else if (cg_tcalc == float_type_index) {
    const CellGrid<T, Tacc, float, T4>* cgp = pm->getCellGridPointer<T, Tacc, float, T4>();
    mapDensityLambda<T, Tacc, float, T4>(pm, cgp, poly_ag, lambda_ele, lambda_threshold);
  }
}

} // namespace energy
} // namespace stormm
