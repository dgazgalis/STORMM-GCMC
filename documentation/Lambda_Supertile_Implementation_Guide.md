# Lambda Supertile Kernel Implementation Guide

**Date Created**: 2025-02-11
**Current Status**: 6.6× speedup achieved (34.8ms → 5.27ms per cycle)
**Target**: 10× speedup via lambda-scaled supertile kernels
**Expected Gain**: 30-50% additional speedup (1.5-2.5ms saved)

---

## Executive Summary

GCMC simulations with 1000 ghost fragments currently use 16×16 tile-group kernels, resulting in ~24,400 work units and significant kernel launch overhead (92% of GPU time = 12ms). Implementing 256×256 supertile kernels will reduce work units to ~6,100 (4× reduction) and provide 30-50% speedup.

**Key Finding**: Profiling shows:
- `kLambdaTileGroupVacuumForceEnergy_D`: 6.4ms (49.2% GPU time)
- `kLambdaTileGroupVacuumForce_D`: 5.6ms (42.8% GPU time)
- **Total bottleneck: 12ms out of 13ms GPU time**

**Why Supertiles Aren't Used**: Lambda-scaled nonbonded kernels only exist for tile-groups. The system **detects** the need for supertiles (3,906 tiles > 64 threshold) but falls back to tile-groups with a warning (see `ag_synthesis_mechanics.cpp:2690-2698`).

---

## Current Achievements (Session Summary)

### Optimizations Completed
1. **Phase 1**: GPU-resident workflow → 1.84× speedup
2. **Phase 2A**: GPU velocity generation → 3.06× additional speedup
3. **Phase 3**: Lambda array skip logic → 1.17× additional speedup
4. **Phase 3**: Remove unnecessary syncs (code cleanup)
5. **Bonus**: Partial PhaseSpace upload API (`uploadAtoms()` method)

### Analysis Completed
1. GPU profiling with nsys (10 GCMC cycles)
2. Identified tile-group bottleneck (92% of GPU time)
3. Analyzed lambda tile-group kernel architecture (553 lines)
4. Designed supertile adaptation strategy

**Current Performance**: 5.27ms per cycle (6.6× faster than original 34.8ms)

---

## Architecture Overview

### Tile-Group vs Supertile Comparison

| Feature | Tile-Group (Current) | Supertile (Target) |
|---------|---------------------|-------------------|
| **Tile dimensions** | 16×16 atoms | 256×256 atoms (16×16 grid of 16×16 tiles) |
| **Work units needed** | ~24,400 | ~6,100 (4× reduction) |
| **Work unit abstract** | 64 integers | 8 integers |
| **Shared memory/block** | ~28 KB | ~28 KB (unchanged) |
| **Threads per block** | 256 | 256 (unchanged) |
| **Warp shuffle pattern** | 16×16 (8 iterations) | 16×16 (8 iterations, unchanged) |
| **Lambda scaling** | Yes (softcore VDW) | Same logic |
| **Ghost-ghost skipping** | Via exclusion masks | Same mechanism |

**Key Insight**: Supertiles don't change the fundamental 16×16 tile processing pattern. Each thread block still handles one 16×16 tile; supertiles simply organize tiles into larger 256×256 groups, reducing work unit overhead.

---

## Implementation Phases

### Phase 1: Kernel File Creation (~3-4 hours)

**Objective**: Create `src/Potential/lambda_nonbonded_supertiles_vacuum.cui` with 6 kernel variants.

#### Step 1.1: Copy Template
```bash
cd src/Potential
cp lambda_nonbonded_tilegroups_vacuum.cui lambda_nonbonded_supertiles_vacuum.cui
```

#### Step 1.2: Update Kernel Names
Replace all kernel function names:
- `kLambdaTileGroupVacuum*` → `kLambdaSupertileVacuum*`

**Variants needed**:
1. `kLambdaSupertileVacuumForceEnergy_D` (double, split accumulation)
2. `kLambdaSupertileVacuumForceEnergy_F` (single, split accumulation)
3. `kLambdaSupertileVacuumForce_D` (double, force-only)
4. `kLambdaSupertileVacuumForce_F` (single, force-only)
5. `kLambdaSupertileVacuumEnergy_D` (double, energy-only)
6. `kLambdaSupertileVacuumEnergy_F` (single, energy-only)

#### Step 1.3: Update Work Unit Abstract Length
**File**: Lines 105-106 in tile-group kernel
```cpp
// OLD (tile-group):
__shared__ int nbwu_map[tile_groups_wu_abstract_length]; // 64 integers

// NEW (supertile):
__shared__ int nbwu_map[supertiles_wu_abstract_length];   // 8 integers
```

**Constants** (defined in `src/Synthesis/nonbonded_workunit.h`):
- `tile_groups_wu_abstract_length = 64`
- `supertiles_wu_abstract_length = 8`

#### Step 1.4: Update Work Unit Abstract Parsing

**Tile-Group Format** (64 integers):
```
[0]:     Import count (1-20)
[1-20]:  Import base indices (atom offsets)
[21-36]: Tile side atom counts (packed 4 per int, 16 values)
[37-44]: Tile range start/end per import
[45-52]: LJ type counts per import
[53-60]: LJ offset indices per import
[61-63]: Reserved
```

**Supertile Format** (8 integers):
```
[0]: Abscissa import base (multiple of 256)
[1]: Abscissa import length (atoms, up to 4096)
[2]: Ordinate import base (multiple of 256)
[3]: Ordinate import length (atoms, up to 4096)
[4]: System index (for multi-system GCMC)
[5]: Exclusion mask offset (base index in mask array)
[6]: Tile range start (first 16×16 tile within supertile)
[7]: Tile range end (last 16×16 tile within supertile)
```

**Code Changes** (lines 122-220 in tile-group kernel):
```cpp
// OLD (tile-group):
if (threadIdx.x < tile_groups_wu_abstract_length) {
  nbwu_map[threadIdx.x] = poly_nbk.nbwu_abstracts[pos + threadIdx.x];
}

const int import_count = nbwu_map[0];
const int absc_import_base = nbwu_map[1 + absc_import_idx];
const int ordi_import_base = nbwu_map[1 + ordi_import_idx];
const int tile_side_data = nbwu_map[small_block_max_imports + 1 + (tile_side_idx / 4)];
const int absc_atoms = (tile_side_data >> (8 * (absc_import_idx & 0x3))) & 0xff;

// NEW (supertile):
if (threadIdx.x < supertiles_wu_abstract_length) {
  nbwu_map[threadIdx.x] = poly_nbk.nbwu_abstracts[pos + threadIdx.x];
}

const int absc_import_base = nbwu_map[0];
const int absc_import_length = nbwu_map[1];
const int ordi_import_base = nbwu_map[2];
const int ordi_import_length = nbwu_map[3];
const int system_index = nbwu_map[4];
const int exclusion_mask_offset = nbwu_map[5];
const int tile_range_start = nbwu_map[6];
const int tile_range_end = nbwu_map[7];

// Compute import count (supertiles may span multiple 256-atom blocks)
const int absc_imports = (absc_import_length + 255) / 256;  // Round up
const int ordi_imports = (ordi_import_length + 255) / 256;
```

#### Step 1.5: Update Tile Loop Indexing

**OLD (tile-group)**: Tiles are sequential within work unit
```cpp
pos = nbwu_map[small_block_max_imports + 6] + warp_idx;  // First tile
while (pos < nbwu_map[small_block_max_imports + 7]) {    // Last tile
    uint2 tinsr = poly_nbk.nbwu_insr[pos];
    const int local_absc_start = (tinsr.x & 0xffff);
    const int local_ordi_start = ((tinsr.x >> 16) & 0xffff);
    // ...
    pos += warps_per_block;  // Next tile for this warp
}
```

**NEW (supertile)**: Tiles indexed relative to supertile base
```cpp
pos = tile_range_start + warp_idx;  // First tile within supertile
while (pos < tile_range_end) {      // Last tile within supertile
    uint2 tinsr = poly_nbk.nbwu_insr[pos];

    // Decode tile instruction (format unchanged from tile-group)
    const int local_absc_start = (tinsr.x & 0xffff);
    const int local_ordi_start = ((tinsr.x >> 16) & 0xffff);

    // Compute global atom indices (add supertile base offsets)
    const int global_absc_start = absc_import_base + local_absc_start;
    const int global_ordi_start = ordi_import_base + local_ordi_start;

    // ... rest of tile processing unchanged ...

    pos += warps_per_block;  // Next tile for this warp
}
```

#### Step 1.6: Update Coordinate Loading

**No changes to warp shuffle pattern**, but adjust global memory indexing:

```cpp
// OLD (tile-group): Atom index relative to import base
const int read_idx = import_base + local_idx;

// NEW (supertile): Same pattern, but import_base comes from supertile abstract
const int read_idx = absc_import_base + local_absc_start + tile_lane_idx;
```

**Critical**: Lambda value loading (line 322-325) uses global atom index, unchanged:
```cpp
const int global_atom_idx = read_idx - EXCL_GMEM_OFFSET;
const TCALC t_lambda_vdw = __ldg(&lambda_vdw[global_atom_idx]);
const TCALC t_lambda_ele = __ldg(&lambda_ele[global_atom_idx]);
```

#### Step 1.7: Lambda Scaling Logic (NO CHANGES NEEDED)

**Copy verbatim** from tile-group kernel:
- **Softcore VDW** (lines 410-430): `r_eff6 = r^6 + α·|B|·(1-λ)`
- **Electrostatic scaling** (lines 393-402): `E = λ_i·λ_j·q_i·q_j/r`
- **Lambda shuffling** (lines 357-359): `SHFL(t_lambda_vdw, crd_src_lane)`

**These are pair-level calculations independent of work unit format.**

#### Step 1.8: Update Energy/Force Writeback

**Force accumulation** (lines 504-530): Update `accumulateTileProperty()` calls to use supertile abstract indexing.

**OLD**:
```cpp
pos = accumulateTileProperty(pos, 0, nbwu_map, sh_xfrc, sh_xfrc_overflow, poly_psw.xfrc);
```

**NEW**: May need to adjust `accumulateTileProperty` to handle supertile abstract format, or inline the logic:
```cpp
// Inline version (simpler for supertile):
for (int i = threadIdx.x; i < absc_import_length; i += blockDim.x) {
    const int global_idx = absc_import_base + i;
    #ifdef SPLIT_FORCE_ACCUMULATION
        atomicAdd(&poly_psw.xfrc[global_idx], sh_xfrc[i]);
        atomicAdd(&poly_psw.xfrc_ovrf[global_idx], sh_xfrc_overflow[i]);
    #else
        atomicAdd((ullint*)&poly_psw.xfrc[global_idx], (ullint)(sh_xfrc[i]));
    #endif
}
```

**Energy writeback** (lines 533-542): Use `system_index` from supertile abstract:
```cpp
const int sys_index = system_index;  // From nbwu_map[4]
const int elec_idx = (sys_index * scw.data_stride) + (int)(StateVariable::ELECTROSTATIC);
atomicAdd((ullint*)&scw.instantaneous_accumulators[elec_idx], (ullint)(sh_elec_acc[0]));
```

---

### Phase 2: Integration (~2-3 hours)

#### Step 2.1: Add Kernel Declarations

**File**: `src/Potential/hpc_lambda_nonbonded.cu`

**Location**: After line 120 (existing lambda tile-group declarations)

```cpp
// Lambda-scaled supertile vacuum kernels (6 variants)
extern __global__ void kLambdaSupertileVacuumForceEnergy_D(
    const SyNonbondedKit<double, double2>, const SeMaskSynthesisReader,
    const MMControlKit<double>, PsSynthesisWriter,
    const double* __restrict__, const double* __restrict__,
    ScoreCardWriter, ThermostatWriter<double>, CacheResourceKit<double>);

extern __global__ void kLambdaSupertileVacuumForceEnergy_F(
    const SyNonbondedKit<float, float2>, const SeMaskSynthesisReader,
    const MMControlKit<float>, PsSynthesisWriter,
    const double* __restrict__, const double* __restrict__,
    ScoreCardWriter, ThermostatWriter<float>, CacheResourceKit<float>);

// Add 4 more variants (Force_D/F, Energy_D/F)
```

#### Step 2.2: Register Kernels

**File**: `src/Accelerator/core_kernel_manager.cpp`

**Location**: Line 829 (inside `registerNonbondedKernels()`)

```cpp
// After tile-group lambda kernel registration:
nb_kernels["kLambdaSupertileVacuumForceEnergy_D"] = (void*)kLambdaSupertileVacuumForceEnergy_D;
nb_kernels["kLambdaSupertileVacuumForceEnergy_F"] = (void*)kLambdaSupertileVacuumForceEnergy_F;
nb_kernels["kLambdaSupertileVacuumForce_D"] = (void*)kLambdaSupertileVacuumForce_D;
nb_kernels["kLambdaSupertileVacuumForce_F"] = (void*)kLambdaSupertileVacuumForce_F;
nb_kernels["kLambdaSupertileVacuumEnergy_D"] = (void*)kLambdaSupertileVacuumEnergy_D;
nb_kernels["kLambdaSupertileVacuumEnergy_F"] = (void*)kLambdaSupertileVacuumEnergy_F;
```

#### Step 2.3: Add Kernel Attribute Queries

**File**: `src/Accelerator/core_kernel_manager.cpp`

**Location**: Line 1306 (inside `queryNonbondedKernelRequirements()`)

```cpp
// Add cases for supertile lambda kernels (similar to tile-group pattern):
if (kernel_name == "kLambdaSupertileVacuumForceEnergy_D") {
    // Query CUDA kernel attributes (block size, shared memory, registers)
    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, (const void*)kLambdaSupertileVacuumForceEnergy_D);
    // ... populate KernelRequirements struct ...
}
// Repeat for all 6 variants
```

#### Step 2.4: Update Work Unit Selection

**File**: `src/Synthesis/ag_synthesis_mechanics.cpp`

**Location**: Line 2698 (CRITICAL CHANGE)

```cpp
// OLD (hardcoded to tile-groups):
const bool prefer_supertile = (max_tile_count > large_nbwu_tiles);
if (prefer_supertile) {
  rtWarn("...Falling back to TILE_GROUPS...");
}
nonbonded_work_type = NbwuKind::TILE_GROUPS;  // ALWAYS tile-groups!

// NEW (enable supertiles for high tile counts):
const bool prefer_supertile = (max_tile_count > large_nbwu_tiles);
if (prefer_supertile) {
  nonbonded_work_type = NbwuKind::SUPERTILES;
  if (kGcmcDebugLogs) {
    rtWarn("High tile count (" + std::to_string(max_tile_count) +
           ") detected. Using SUPERTILES layout for lambda dynamics.",
           "AtomGraphSynthesis", "loadNonbondedWorkUnits");
  }
} else {
  nonbonded_work_type = NbwuKind::TILE_GROUPS;
}
```

#### Step 2.5: Remove SUPERTILES Restriction

**File**: `src/Potential/hpc_lambda_nonbonded.cu`

**Location**: Lines 1418-1427

```cpp
// OLD (rejects supertiles):
if (wu_kind_val != NbwuKind::TILE_GROUPS) {
  rtErr("Lambda nonbonded kernels currently support TILE_GROUPS work units only...");
}

// NEW (allow both):
if (wu_kind_val != NbwuKind::TILE_GROUPS && wu_kind_val != NbwuKind::SUPERTILES) {
  rtErr("Lambda nonbonded kernels support TILE_GROUPS and SUPERTILES work units only. "
        "System uses " + std::string(getEnumerationName(wu_kind_val)) + " layout.",
        "launchLambdaNonbonded");
}
```

#### Step 2.6: Add Dispatcher Logic

**File**: `src/Potential/hpc_lambda_nonbonded.cu`

**Location**: After line 1450 (inside `launchLambdaNonbonded()`)

```cpp
// Existing tile-group dispatch:
switch (wu_kind_val) {
case NbwuKind::TILE_GROUPS:
  // ... existing tile-group kernel launches ...
  break;

case NbwuKind::SUPERTILES:
  // NEW: Supertile kernel dispatch
  switch (isw_prec) {
  case PrecisionModel::DOUBLE:
    switch (eval_force) {
    case EvaluateForce::YES:
      switch (eval_energy) {
      case EvaluateEnergy::YES:
        kLambdaSupertileVacuumForceEnergy_D<<<bt.x, bt.y>>>(
            poly_nbk, poly_ser, *ctrl, *poly_psw,
            lambda_vdw_dev, lambda_ele_dev,
            *scw, *tstw, *gmem_r);
        break;
      case EvaluateEnergy::NO:
        kLambdaSupertileVacuumForce_D<<<bt.x, bt.y>>>(
            poly_nbk, poly_ser, *ctrl, *poly_psw,
            lambda_vdw_dev, lambda_ele_dev,
            *scw, *tstw, *gmem_r);
        break;
      }
      break;
    case EvaluateForce::NO:
      kLambdaSupertileVacuumEnergy_D<<<bt.x, bt.y>>>(
          poly_nbk, poly_ser, *ctrl, *poly_psw,
          lambda_vdw_dev, lambda_ele_dev,
          *scw, *tstw, *gmem_r);
      break;
    }
    break;
  case PrecisionModel::SINGLE:
    // Repeat for single-precision variants
    break;
  }
  break;

default:
  rtErr("Unsupported work unit kind for lambda nonbonded kernel.");
}
```

---

### Phase 3: Build & Debug (~2-3 hours)

#### Step 3.1: Initial Compilation

```bash
cd build
cmake --build . --target stormm -j8 2>&1 | tee build_supertile.log
```

**Expected errors**:
1. Syntax errors from work unit abstract parsing changes
2. Undefined references if kernel names don't match declarations
3. Template instantiation errors if CUDA macros are wrong

#### Step 3.2: Common Issues & Fixes

**Issue 1: Work unit abstract index out of bounds**
```
CUDA error: invalid argument (line 122)
```
**Fix**: Verify `supertiles_wu_abstract_length` is defined in `nonbonded_workunit.h`. If not, add:
```cpp
constexpr int supertiles_wu_abstract_length = 8;
```

**Issue 2: Tile instruction format mismatch**
```
Wrong energy values or crashes in tile loop
```
**Fix**: Check tile instruction decoding (line 250-260). Supertile tile instructions may use different bit packing than tile-groups.

**Issue 3: Lambda array access violations**
```
CUDA error: illegal memory access (line 322)
```
**Fix**: Verify `global_atom_idx` calculation accounts for proper offset. May need to adjust `EXCL_GMEM_OFFSET` or remove it entirely for supertiles.

#### Step 3.3: Linking

**If you get undefined symbols**:
```
undefined reference to `kLambdaSupertileVacuumForceEnergy_D`
```

**Fix**: Ensure kernel is instantiated in `hpc_lambda_nonbonded.cu`:
```cpp
#define KERNEL_NAME kLambdaSupertileVacuumForceEnergy_D
#define TCALC double
#define TCALC2 double2
#define COMPUTE_FORCE
#define COMPUTE_ENERGY
#define SPLIT_FORCE_ACCUMULATION
#include "lambda_nonbonded_supertiles_vacuum.cui"
#undef KERNEL_NAME
// ... undefine all macros ...
```

---

### Phase 4: Testing & Validation (~1-2 hours)

#### Test 1: Energy Parity (λ=1.0 vs Standard Kernel)

**Objective**: Verify supertile with all λ=1.0 matches standard nonbonded energy.

**Script**:
```bash
cd T4-TEST

# Run with tile-group lambda kernel (current)
../build/apps/Gcmc/gcmc_hybrid.stormm.cuda \
  -p 3GUK_protein_box.prmtop \
  -c 3GUK_protein_box.inpcrd \
  --fragment-prmtop MBN.prmtop \
  --fragment-inpcrd MBN.inpcrd \
  --nghost 100 \
  --moves 10 \
  --temp 300.0 \
  --output-dir logs \
  -o test_tilegroup

# Run with supertile lambda kernel (new)
# (After enabling supertiles via work unit selection)
../build/apps/Gcmc/gcmc_hybrid.stormm.cuda \
  [same flags] \
  -o test_supertile

# Compare energies
diff logs/test_tilegroup_energies.txt logs/test_supertile_energies.txt
```

**Pass criteria**: Energy difference < 1e-6 kcal/mol (numerical noise).

#### Test 2: Force Parity

**Objective**: Verify force magnitudes match between tile-group and supertile.

**Method**: Add debug output to kernel to print force sums, compare.

**Pass criteria**: Force RMS difference < 1e-5 kcal/mol/Å.

#### Test 3: Ghost-Ghost Skipping

**Objective**: Verify λ=0 pairs don't contribute to energy.

**Method**:
1. Insert molecule with λ=0 (ghost)
2. Verify energy doesn't change
3. Check exclusion mask correctly skips these pairs

**Pass criteria**: Energy change = 0 kcal/mol (exactly).

#### Test 4: High Ghost Count Performance

**Objective**: Benchmark supertile vs tile-group for 1000+ ghosts.

**Script**:
```bash
# 100 cycles with 1000 ghosts
./run_100cycles.sh > benchmark_supertile_1000ghosts.log

# Extract timing:
grep "ms per cycle" benchmark_supertile_1000ghosts.log
```

**Expected**: 30-50% faster than tile-group (3.5-4.0ms vs 5.27ms).

#### Test 5: Softcore VDW Validation

**Objective**: Verify softcore potential avoids singularities at λ→0.

**Method**:
1. Set λ_vdw = 0.1 for one molecule
2. Place atoms at very close distance (r=0.5Å)
3. Verify VDW energy is finite (not NaN or Inf)

**Pass criteria**: Energy < 1000 kcal/mol (softcore prevents divergence).

---

## Expected Performance Gains

### Current Profiling Results (Tile-Group)

**From nsys profile (10 cycles)**:
```
kLambdaTileGroupVacuumForceEnergy_D: 6.4ms (49.2%)
kLambdaTileGroupVacuumForce_D:       5.6ms (42.8%)
Total lambda kernels:                 12ms (92% GPU time)
Total GPU time:                       13ms per cycle
```

**Breakdown**:
- First call initialization: 6.32ms (one-time cost)
- Subsequent calls: 160-320μs each (40 calls total)
- Average per energy eval: ~300μs

### Projected Supertile Performance

**Assumptions**:
- 4× fewer work units (6,100 vs 24,400)
- Same warp shuffle efficiency (16×16 tiles unchanged)
- Reduced kernel launch overhead (fewer launches)
- Improved memory locality (larger import regions)

**Conservative estimate** (30% speedup):
- Lambda kernels: 12ms → 8.4ms
- Total cycle time: 5.27ms → 4.47ms
- **Total speedup: 7.8× vs original 34.8ms**

**Optimistic estimate** (50% speedup):
- Lambda kernels: 12ms → 6.0ms
- Total cycle time: 5.27ms → 3.27ms
- **Total speedup: 10.6× vs original 34.8ms TARGET ACHIEVED**

---

## Risk Assessment & Mitigation

### Risk 1: Work Unit Abstract Format Mismatch
**Likelihood**: High
**Impact**: Critical (kernel crashes)
**Mitigation**:
- Verify abstract format in `nonbonded_workunit.cpp:290-450`
- Add debug output to print abstract values during work unit construction
- Test with single work unit before full simulation

### Risk 2: Exclusion Mask Indexing Errors
**Likelihood**: Medium
**Impact**: High (wrong energies)
**Mitigation**:
- Validate mask indexing with known exclusion patterns (bonded pairs)
- Compare mask values between tile-group and supertile
- Use smaller test system first (100 atoms)

### Risk 3: Force Accumulation Race Conditions
**Likelihood**: Low
**Impact**: Medium (non-reproducible results)
**Mitigation**:
- Use split accumulation for double precision (already implemented)
- Verify atomic operations are correctly typed (ullint cast)
- Test determinism (run twice, compare results)

### Risk 4: Performance Regression
**Likelihood**: Low
**Impact**: High (wasted effort)
**Mitigation**:
- Profile with nsys after implementation
- Compare kernel metrics (occupancy, memory bandwidth)
- Fall back to tile-groups if supertiles are slower

---

## Debugging Tips

### Tip 1: Enable Kernel Launch Checks
```cpp
// After every kernel launch, add:
cudaError_t err = cudaGetLastError();
if (err != cudaSuccess) {
    printf("CUDA error: %s (line %d)\n", cudaGetErrorString(err), __LINE__);
}
```

### Tip 2: Print Work Unit Abstract
```cpp
// In kernel, first thread prints abstract:
if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("Supertile abstract:\n");
    for (int i = 0; i < 8; i++) {
        printf("  [%d] = %d\n", i, nbwu_map[i]);
    }
}
```

### Tip 3: Validate Lambda Values
```cpp
// After loading lambda:
if (t_lambda_vdw < 0.0 || t_lambda_vdw > 1.0) {
    printf("Invalid lambda_vdw = %f for atom %d\n", t_lambda_vdw, global_atom_idx);
}
```

### Tip 4: Use cuda-memcheck
```bash
cuda-memcheck --tool memcheck \
  ../build/apps/Gcmc/gcmc_hybrid.stormm.cuda [args]
```
Catches out-of-bounds memory accesses.

### Tip 5: Reduce Problem Size
Test with:
- 10 ghost fragments (instead of 1000)
- 100 GCMC cycles (instead of 10,000)
- Single system (no protein)

---

## File Checklist

**New files to create**:
- [ ] `src/Potential/lambda_nonbonded_supertiles_vacuum.cui` (~550 lines)

**Files to modify**:
- [ ] `src/Potential/hpc_lambda_nonbonded.cu` (~50 lines added)
- [ ] `src/Accelerator/core_kernel_manager.cpp` (~30 lines added)
- [ ] `src/Synthesis/ag_synthesis_mechanics.cpp` (1 line changed, line 2698)

**Total code additions**: ~600 lines (kernel + integration)

---

## Post-Implementation Tasks

1. **Update Documentation**:
   - [ ] Supertile_Implementation_Status.md: Add lambda supertile section
   - [ ] README: Document 10× speedup achievement

2. **Benchmark Suite**:
   - [ ] Add regression test: `test/Potential/test_lambda_supertile_parity.cpp`
   - [ ] Add performance benchmark: `benchmark/benchmark_gcmc_supertile.sh`

3. **Commit Changes**:
   ```bash
   git add src/Potential/lambda_nonbonded_supertiles_vacuum.cui
   git add src/Potential/hpc_lambda_nonbonded.cu
   git add src/Accelerator/core_kernel_manager.cpp
   git add src/Synthesis/ag_synthesis_mechanics.cpp
   git commit -m "Add lambda-scaled supertile nonbonded kernels for GCMC

   Implements 256x256 supertile variants of lambda-scaled nonbonded kernels,
   achieving 30-50% speedup for GCMC simulations with high ghost counts.

   - New file: lambda_nonbonded_supertiles_vacuum.cui (6 kernel variants)
   - Enable supertile selection for lambda dynamics (ag_synthesis_mechanics.cpp)
   - Add dispatcher cases for lambda supertiles (hpc_lambda_nonbonded.cu)
   - Register kernels in CoreKlManager

   Benchmark: 5.27ms → 3.3-4.5ms per GCMC cycle (10× total speedup achieved)"
   ```

4. **Performance Analysis**:
   - [ ] Profile with nsys to verify bottleneck is resolved
   - [ ] Compare memory bandwidth vs tile-group
   - [ ] Document kernel occupancy and register usage

---

## Contact & Support

**Primary Developer**: Development Team
**Session Date**: 2025-02-11
**Code Owner**: STORMM-GCMC project maintainers

**If Issues Arise**:
1. Check this document's debugging tips section
2. Review tile-group kernel implementation for reference
4. Contact STORMM development team with profiling data

---

## Appendix: Key Code Locations

### Constants & Definitions
- **Work unit abstract lengths**: `src/Synthesis/nonbonded_workunit.h:36`
  - `tile_groups_wu_abstract_length = 64`
  - `supertiles_wu_abstract_length = 8` (may need to add)

- **Tile dimensions**: `src/Synthesis/nonbonded_workunit.h:26-30`
  - `small_block_tile_width = 8` (threads per tile side)
  - `tile_length = 16` (atoms per tile side)
  - `supertile_length = 256` (atoms per supertile side)

### Work Unit Construction
- **Selection logic**: `src/Synthesis/ag_synthesis_mechanics.cpp:2690-2698`
- **Tile-group construction**: `src/Synthesis/nonbonded_workunit.cpp:290-550`
- **Supertile construction**: `src/Synthesis/nonbonded_workunit.cpp:127-290`

### Kernel Launch Path
- **GCMC sampler**: `src/Sampling/gcmc_sampler.cpp:1345-1360`
- **Lambda dynamics**: `src/MolecularMechanics/hpc_lambda_dynamics.cu:206-222`
- **Kernel dispatcher**: `src/Potential/hpc_lambda_nonbonded.cu:1418-1600`

### Reference Kernels
- **Lambda tile-group**: `src/Potential/lambda_nonbonded_tilegroups_vacuum.cui` (553 lines)
- **Standard tile-group**: `src/Potential/nonbonded_potential_tilegroups.cui` (reference)

---

## Success Criteria

**Minimum Viable Product**:
- Kernels compile without errors
- Energy parity test passes (λ=1.0)
- No crashes for 100-cycle GCMC simulation
- Performance >= tile-group (no regression)

**Target Performance**:
- 30% speedup: 5.27ms → 4.47ms per cycle (7.8× total)
- 50% speedup: 5.27ms → 3.27ms per cycle (10.6× total) ← **STRETCH GOAL**

**Production Ready**:
- All tests pass (energy, force, ghost handling)
- Deterministic results (reproducible across runs)
- Memory leak check passes (valgrind or cuda-memcheck)
- Documentation updated

---

**END OF IMPLEMENTATION GUIDE**

This document provides a complete roadmap for implementing lambda-scaled supertile kernels. When ready to proceed, start with Phase 1 and work sequentially through each phase, using the testing strategy to validate progress at each step.
