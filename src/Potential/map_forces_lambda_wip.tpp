// -*-c++-*-
#include "copyright.h"
#include "Math/bspline.h"
#include "Numerics/split_fixed_precision.h"

namespace stormm {
namespace energy {

using stmath::bSpline;
using numerics::hostSplitAccumulation;

//-------------------------------------------------------------------------------------------------
template <typename Tcalc, typename Tgrid>
void interpolateForcesLambda(const Tcalc* da_cof, const Tcalc* b_cof, const Tcalc* c_cof,
                             const Tcalc* a_cof, const Tcalc* db_cof, const Tcalc* dc_cof,
                             const int bspline_order, const int grid_root_a, const int grid_root_b,
                             const int grid_root_c, const uint4 grid_dims, const FFTMode fft_staging,
                             const Tgrid* grid_data, Tcalc* fx, Tcalc* fy, Tcalc* fz,
                             const Tcalc* umat, const Tcalc mesh_dims_a, const Tcalc mesh_dims_b,
                             const Tcalc mesh_dims_c) {
  // Determine grid padding based on FFT mode
  uint padded_gdim_x;
  switch (fft_staging) {
  case FFTMode::IN_PLACE:
    padded_gdim_x = 2 * ((grid_dims.x / 2) + 1);
    break;
  case FFTMode::OUT_OF_PLACE:
    padded_gdim_x = grid_dims.x;
    break;
  }

  // Initialize force components
  Tcalc force_a = 0.0;
  Tcalc force_b = 0.0;
  Tcalc force_c = 0.0;

  // Interpolate forces using B-spline derivatives
  // Force = -charge * gradient(potential)
  // gradient computed using B-spline derivative coefficients
  for (int k = 0; k < bspline_order; k++) {
    int kg_pos = grid_root_c + k;
    kg_pos += ((kg_pos < 0) - (kg_pos >= grid_dims.z)) * grid_dims.z;

    for (int j = 0; j < bspline_order; j++) {
      int jg_pos = grid_root_b + j;
      jg_pos += ((jg_pos < 0) - (jg_pos >= grid_dims.y)) * grid_dims.y;
      const size_t jk_gidx = grid_dims.w + (((kg_pos * grid_dims.y) + jg_pos) * padded_gdim_x);

      for (int i = 0; i < bspline_order; i++) {
        int ig_pos = grid_root_a + i;
        ig_pos += ((ig_pos < 0) - (ig_pos >= grid_dims.x)) * grid_dims.x;
        const size_t gidx = jk_gidx + static_cast<size_t>(ig_pos);

        const Tcalc potential = grid_data[gidx];

        // Accumulate force components using B-spline derivatives
        // F_a = -q * dφ/da where a is fractional coordinate
        force_a += potential * da_cof[i] * b_cof[j] * c_cof[k];
        force_b += potential * a_cof[i] * db_cof[j] * c_cof[k];
        force_c += potential * a_cof[i] * b_cof[j] * dc_cof[k];
      }
    }
  }

  // Scale by mesh dimensions to convert from fractional to real derivatives
  force_a *= mesh_dims_a;
  force_b *= mesh_dims_b;
  force_c *= mesh_dims_c;

  // Transform forces from fractional to Cartesian coordinates
  // The transformation matrix is transposed for force transformation
  *fx += (umat[0] * force_a);
  *fy += (umat[3] * force_a) + (umat[4] * force_b);
  *fz += (umat[6] * force_a) + (umat[7] * force_b) + (umat[8] * force_c);
}

//-------------------------------------------------------------------------------------------------
template <typename T, typename Tacc, typename Tcalc, typename Tcalc2, typename T4>
void accumulateCellForcesLambda(const PMIGridReader &pm_reader,
                                PsSynthesisWriter &ps_writer,
                                const int sysid,
                                const int cell_i, const int cell_j, const int cell_k,
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

  // Determine limits and other critical constants
  const bool coord_in_real = (cgr.lpos_scale < 1.01);
  const bool tcalc_is_double = (std::type_index(typeid(Tcalc)).hash_code() == double_type_index);
  const size_t ijk_cellidx = cell_offset + (((cell_k * cell_nb) + cell_j) * cell_na) + cell_i;
  const uint2 ijk_bounds = cgr.cell_limits[ijk_cellidx];
  const uint mllim = ijk_bounds.x;
  const uint mhlim = mllim + (ijk_bounds.y >> 16);
  const int xfrm_stride = roundUp(9, warp_size_int);

  // Get transformation matrix for this system
  Tcalc umat[9];
  for (int i = 0; i < 9; i++) {
    umat[i] = cgr.system_cell_umat[(sysid * xfrm_stride) + i];
  }

  // Get grid dimensions and mesh parameters
  const uint4 grid_dims = pm_reader.dims[sysid];
  const Tcalc mesh_dims_a = static_cast<Tcalc>(grid_dims.x);
  const Tcalc mesh_dims_b = static_cast<Tcalc>(grid_dims.y);
  const Tcalc mesh_dims_c = static_cast<Tcalc>(grid_dims.z);

  // Lay out arrays to collect B-spline coefficients and derivatives
  switch (pm_reader.mode) {
  case PrecisionModel::DOUBLE:
    {
      std::vector<double> a_cof(pm_reader.order), b_cof(pm_reader.order), c_cof(pm_reader.order);
      std::vector<double> da_cof(pm_reader.order), db_cof(pm_reader.order), dc_cof(pm_reader.order);
      double umat_d[9];
      for (int i = 0; i < 9; i++) {
        umat_d[i] = static_cast<double>(umat[i]);
      }

      for (uint m = mllim; m < mhlim; m++) {
        const T4 atom_m = cgr.image[m];
        const int atom_idx = cgr.nonimg_atom_idx[m];

        // Skip atoms with lambda below threshold (ghosts)
        if (lambda_ele[atom_idx] < lambda_threshold) {
          continue;
        }

        // Get particle alignment on grid and compute B-spline coefficients with derivatives
        int grid_root_a, grid_root_b, grid_root_c;
        particleAlignment<Tcalc, double>(atom_m.x, atom_m.y, atom_m.z, cgr.inv_lpos_scale, umat,
                                         cgr.mesh_ticks, cell_i, cell_j, cell_k, a_cof.data(),
                                         b_cof.data(), c_cof.data(), pm_reader.order, &grid_root_a,
                                         &grid_root_b, &grid_root_c);

        // Compute B-spline derivatives for force interpolation
        // Note: particleAlignment already computed B-splines, now we need derivatives
        const Tcalc rel_x = atom_m.x * cgr.inv_lpos_scale;
        const Tcalc rel_y = atom_m.y * cgr.inv_lpos_scale;
        const Tcalc rel_z = atom_m.z * cgr.inv_lpos_scale;

        // Transform to fractional coordinates
        const Tcalc frac_a = (umat[0] * rel_x) + (umat[3] * rel_y) + (umat[6] * rel_z);
        const Tcalc frac_b = (umat[4] * rel_y) + (umat[7] * rel_z);
        const Tcalc frac_c = (umat[8] * rel_z);

        // Get fractional position within grid cell
        const Tcalc dmt = cgr.mesh_ticks;
        Tcalc da = frac_a * dmt - floor(frac_a * dmt);
        Tcalc db = frac_b * dmt - floor(frac_b * dmt);
        Tcalc dc = frac_c * dmt - floor(frac_c * dmt);

        // Compute B-spline and derivatives
        bSpline<double>(da, pm_reader.order, a_cof.data(), da_cof.data());
        bSpline<double>(db, pm_reader.order, b_cof.data(), db_cof.data());
        bSpline<double>(dc, pm_reader.order, c_cof.data(), dc_cof.data());

        // Get charge and apply lambda scaling
        const double q = synbk.charge[atom_idx] * lambda_ele[atom_idx];

        // Interpolate forces from grid
        double fx = 0.0, fy = 0.0, fz = 0.0;
        interpolateForcesLambda<double, double>(da_cof.data(), b_cof.data(), c_cof.data(),
                                                a_cof.data(), db_cof.data(), dc_cof.data(),
                                                pm_reader.order, grid_root_a, grid_root_b,
                                                grid_root_c, grid_dims, pm_reader.fftm,
                                                pm_reader.ddata, &fx, &fy, &fz, umat_d,
                                                mesh_dims_a, mesh_dims_b, mesh_dims_c);

        // Scale forces by negative charge (force = -charge * gradient(potential))
        fx *= -q;
        fy *= -q;
        fz *= -q;

        // Accumulate forces into phase space arrays
        ps_writer.xfrc[atom_idx] += fx;
        ps_writer.yfrc[atom_idx] += fy;
        ps_writer.zfrc[atom_idx] += fz;
      }
    }
    break;

  case PrecisionModel::SINGLE:
    {
      std::vector<float> a_cof(pm_reader.order), b_cof(pm_reader.order), c_cof(pm_reader.order);
      std::vector<float> da_cof(pm_reader.order), db_cof(pm_reader.order), dc_cof(pm_reader.order);
      float umat_f[9];
      for (int i = 0; i < 9; i++) {
        umat_f[i] = static_cast<float>(umat[i]);
      }

      for (uint m = mllim; m < mhlim; m++) {
        const T4 atom_m = cgr.image[m];
        const int atom_idx = cgr.nonimg_atom_idx[m];

        // Skip atoms with lambda below threshold
        if (lambda_ele[atom_idx] < lambda_threshold) {
          continue;
        }

        int grid_root_a, grid_root_b, grid_root_c;
        particleAlignment<Tcalc, float>(atom_m.x, atom_m.y, atom_m.z, cgr.inv_lpos_scale, umat,
                                        cgr.mesh_ticks, cell_i, cell_j, cell_k, a_cof.data(),
                                        b_cof.data(), c_cof.data(), pm_reader.order, &grid_root_a,
                                        &grid_root_b, &grid_root_c);

        // Compute fractional coordinates and B-spline derivatives
        const Tcalc rel_x = atom_m.x * cgr.inv_lpos_scale;
        const Tcalc rel_y = atom_m.y * cgr.inv_lpos_scale;
        const Tcalc rel_z = atom_m.z * cgr.inv_lpos_scale;

        const Tcalc frac_a = (umat[0] * rel_x) + (umat[3] * rel_y) + (umat[6] * rel_z);
        const Tcalc frac_b = (umat[4] * rel_y) + (umat[7] * rel_z);
        const Tcalc frac_c = (umat[8] * rel_z);

        const Tcalc dmt = cgr.mesh_ticks;
        Tcalc da = frac_a * dmt - floor(frac_a * dmt);
        Tcalc db = frac_b * dmt - floor(frac_b * dmt);
        Tcalc dc = frac_c * dmt - floor(frac_c * dmt);

        bSpline<float>(da, pm_reader.order, a_cof.data(), da_cof.data());
        bSpline<float>(db, pm_reader.order, b_cof.data(), db_cof.data());
        bSpline<float>(dc, pm_reader.order, c_cof.data(), dc_cof.data());

        // Get lambda-scaled charge
        const float q = synbk.charge[atom_idx] * lambda_ele[atom_idx];

        // Interpolate forces
        float fx = 0.0f, fy = 0.0f, fz = 0.0f;
        interpolateForcesLambda<float, float>(da_cof.data(), b_cof.data(), c_cof.data(),
                                              a_cof.data(), db_cof.data(), dc_cof.data(),
                                              pm_reader.order, grid_root_a, grid_root_b,
                                              grid_root_c, grid_dims, pm_reader.fftm,
                                              pm_reader.fdata, &fx, &fy, &fz, umat_f,
                                              mesh_dims_a, mesh_dims_b, mesh_dims_c);

        fx *= -q;
        fy *= -q;
        fz *= -q;

        // Accumulate forces
        ps_writer.xfrc[atom_idx] += fx;
        ps_writer.yfrc[atom_idx] += fy;
        ps_writer.zfrc[atom_idx] += fz;
      }
    }
    break;
  }
}

//-------------------------------------------------------------------------------------------------
template <typename T, typename Tacc, typename Tcalc, typename T4>
void mapForcesLambda(PMIGrid *pm, const CellGrid<T, Tacc, Tcalc, T4> *cg,
                    const AtomGraphSynthesis *poly_ag,
                    PhaseSpaceSynthesis *ps_synthesis,
                    const double* lambda_ele,
                    const double lambda_threshold) {
  // This form of the function operates on the CPU
  const bool tcalc_is_double = (std::type_index(typeid(Tcalc)).hash_code() == double_type_index);
  const bool tcrd_is_real = isFloatingPointScalarType<T>();

  // Get non-bonded kits for both precisions
  const SyNonbondedKit<double, double2> dsynbk = poly_ag->getDoublePrecisionNonbondedKit();
  const SyNonbondedKit<float, float2> fsynbk = poly_ag->getSinglePrecisionNonbondedKit();
  const CellGridReader<void, void, void, void> cgr_v = cg->templateFreeData();

  // Create properly typed cell grid readers
  const CellGridReader<T, Tacc, double, T4> dcgr = restoreType<T, Tacc, double, T4>(cgr_v);
  const CellGridReader<T, Tacc, float, T4> fcgr = restoreType<T, Tacc, float, T4>(cgr_v);

  // Get PMI grid reader and phase space writer
  PMIGridReader pm_reader = pm->data();
  PsSynthesisWriter ps_writer = ps_synthesis->data();

  // Loop over all cells in all systems
  for (int sysid = 0; sysid < dsynbk.nsys; sysid++) {
    const ullint cell_bounds = (tcalc_is_double) ? dcgr.system_cell_grids[sysid] :
                                                   fcgr.system_cell_grids[sysid];
    const int cell_offset = (cell_bounds & 0xfffffffLLU);
    const int cell_na = ((cell_bounds >> 28) & 0xfffLLU);
    const int cell_nb = ((cell_bounds >> 40) & 0xfffLLU);
    const int cell_nc = ((cell_bounds >> 52) & 0xfffLLU);

    for (int i = 0; i < cell_na; i++) {
      for (int j = 0; j < cell_nb; j++) {
        for (int k = 0; k < cell_nc; k++) {
          // Accumulate forces based on calculation precision
          if (tcalc_is_double) {
            accumulateCellForcesLambda<T, Tacc, double, double2, T4>(
                pm_reader, ps_writer, sysid, i, j, k, dcgr, dsynbk,
                lambda_ele, lambda_threshold);
          }
          else {
            accumulateCellForcesLambda<T, Tacc, float, float2, T4>(
                pm_reader, ps_writer, sysid, i, j, k, fcgr, fsynbk,
                lambda_ele, lambda_threshold);
          }
        }
      }
    }
  }
}

//-------------------------------------------------------------------------------------------------
template <typename T, typename Tacc, typename Tcalc, typename T4>
void mapForcesLambda(PMIGrid *pm, const CellGrid<T, Tacc, Tcalc, T4> &cg,
                    const AtomGraphSynthesis &poly_ag,
                    PhaseSpaceSynthesis &ps_synthesis,
                    const double* lambda_ele,
                    const double lambda_threshold) {
  mapForcesLambda(pm, cg.getSelfPointer(), poly_ag.getSelfPointer(),
                  ps_synthesis.getSelfPointer(), lambda_ele, lambda_threshold);
}

//-------------------------------------------------------------------------------------------------
template <typename T, typename Tacc, typename T4>
void unrollMapForcesLambdaCall(PMIGrid *pm, const size_t cg_tcalc,
                               const AtomGraphSynthesis *poly_ag,
                               PhaseSpaceSynthesis *ps_synthesis,
                               const double* lambda_ele,
                               const double lambda_threshold) {
  if (cg_tcalc == double_type_index) {
    const CellGrid<T, Tacc, double, T4>* cgp = pm->getCellGridPointer<T, Tacc, double, T4>();
    mapForcesLambda<T, Tacc, double, T4>(pm, cgp, poly_ag, ps_synthesis,
                                         lambda_ele, lambda_threshold);
  }
  else if (cg_tcalc == float_type_index) {
    const CellGrid<T, Tacc, float, T4>* cgp = pm->getCellGridPointer<T, Tacc, float, T4>();
    mapForcesLambda<T, Tacc, float, T4>(pm, cgp, poly_ag, ps_synthesis,
                                        lambda_ele, lambda_threshold);
  }
}

//-------------------------------------------------------------------------------------------------
template <typename T, typename T4>
void unrollMapForcesLambdaCall(PMIGrid *pm, const size_t cg_tacc, const size_t cg_tcalc,
                               const AtomGraphSynthesis *poly_ag,
                               PhaseSpaceSynthesis *ps_synthesis,
                               const double* lambda_ele,
                               const double lambda_threshold) {
  if (cg_tacc == int_type_index) {
    unrollMapForcesLambdaCall<T, int, T4>(pm, cg_tcalc, poly_ag, ps_synthesis,
                                          lambda_ele, lambda_threshold);
  }
  else if (cg_tacc == llint_type_index) {
    unrollMapForcesLambdaCall<T, llint, T4>(pm, cg_tcalc, poly_ag, ps_synthesis,
                                            lambda_ele, lambda_threshold);
  }
}

} // namespace energy
} // namespace stormm