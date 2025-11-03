# GCMC GPU Acceleration Design Document

**Date:** 2025-10-29
**Author:** Performance Analysis & Refactoring Plan
**Status:** Design Phase

## Executive Summary

Performance profiling of the hybrid GCMC code revealed that **95.6% of execution time** is spent on GPU to/from CPU data transfers (downloading 8-byte energy values), while GPU kernel execution is only **4.4%** of the time. The GPU infrastructure for full GPU-accelerated GCMC **already exists but is not being used** by the insertion/deletion code. This document outlines the refactoring needed to achieve an estimated **12-18× speedup**.

---

## 1. Performance Bottleneck Analysis

### Current Timing (per energy evaluation):
```
Total time:       36.5 ms
├─ GPU kernel:     1.6 ms  (4.4%)  FAST!
└─ Energy download: 34.9 ms (95.6%) BOTTLENECK!
```

### Root Cause:
The GCMC insertion/deletion code uses a **CPU-centric architecture**:
1. CPU manipulates coordinates (rotation, translation, PBC)
2. Upload coordinates to GPU
3. GPU computes energy (fast!)
4. **Download energy to CPU (34.9ms bottleneck!)**
5. CPU evaluates Metropolis acceptance
6. CPU adjusts lambda values
7. Upload lambda back to GPU
8. Repeat...

**Each insertion/deletion requires 2 energy evaluations** → 73ms lost to synchronous GPU/CPU transfers per move!

---

## 2. Existing GPU Infrastructure (Already Implemented!)

The codebase **already has** GPU-accelerated MC kernels in `src/Sampling/hpc_mc_moves.{h,cu}`:

### Coordinate Manipulation
- `launchTranslateMolecule()` - GPU translation
- `launchRotateMolecule()` - GPU rotation
- `launchBackupCoordinates()` - GPU coordinate backup
- `launchRestoreCoordinates()` - GPU coordinate restore
- `launchConditionalRestore()` - GPU conditional restore (only if rejected)

### Energy Evaluation (GPU-only)
- `launchSumScoreCardEnergy()` - **KEY**: Sums energy components on GPU (no download!)
- `evaluateTotalEnergyGPU()` - Already uses GPU-only energy path

### Metropolis Acceptance
- `launchMetropolisAccept()` - GPU Metropolis criterion (`exp(-β·ΔE)`)
- `launchComputeDeltaE()` - GPU ΔE calculation

### Utilities
- `launchCalculateCOG()` - GPU center of geometry (downloads only 12 bytes)

### Currently Used By:
- `mc_mover.cpp` - Translation/rotation MC moves **already use** these GPU kernels (lines 178, 499)

### NOT Used By:
- `gcmc_sampler.cpp` - GCMC insertion/deletion still uses CPU-centric path

---

## 3. Missing GPU Kernels

Only 2 kernels need to be added:

### 3.1 Lambda Adjustment Kernel
```cuda
__global__ void kAdjustLambda(
    const int n_atoms,
    const int* __restrict__ atom_indices,
    const double new_lambda_vdw,
    const double new_lambda_ele,
    double* __restrict__ lambda_vdw,
    double* __restrict__ lambda_ele)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_atoms) return;

  const int atom = atom_indices[idx];
  lambda_vdw[atom] = new_lambda_vdw;
  lambda_ele[atom] = new_lambda_ele;
}
```

**Rationale:** Sets lambda values for a molecule's atoms. Thread-parallel over atoms. Trivial operation that currently forces CPU to/from GPU roundtrip.

### 3.2 PBC Wrapping Kernel
```cuda
__global__ void kApplyPBC(
    const int n_atoms,
    const int* __restrict__ atom_indices,
    const double box_x, const double box_y, const double box_z,
    double* __restrict__ xcrd,
    double* __restrict__ ycrd,
    double* __restrict__ zcrd)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_atoms) return;

  const int atom = atom_indices[idx];

  // Minimum image convention: wrap into [0, box)
  xcrd[atom] = xcrd[atom] - floor(xcrd[atom] / box_x) * box_x;
  ycrd[atom] = ycrd[atom] - floor(ycrd[atom] / box_y) * box_y;
  zcrd[atom] = zcrd[atom] - floor(zcrd[atom] / box_z) * box_z;
}
```

**Rationale:** Wraps coordinates into primary periodic box. Thread-parallel over atoms. Currently done on CPU, forcing coordinate download/upload.

---

## 4. Refactored GPU-Only Workflow

### Current CPU-Centric Insertion (SLOW):
```
1. [CPU]  Select random ghost molecule
2. [CPU]  Generate random rotation matrix (9 floats)
3. [CPU]  Generate random insertion site (3 floats)
4. [CPU]  Manipulate coordinates (rotation, translation, PBC)
5. [CPU→GPU] Upload coordinates                          SLOW
6. [GPU]  Evaluate energy
7. [GPU→CPU] Download energy (8 bytes)                   34.9ms BOTTLENECK!
8. [CPU]  Metropolis acceptance
9. [CPU]  Adjust lambda
10. [CPU→GPU] Upload lambda                              SLOW
```

### Proposed GPU-Accelerated Insertion (FAST):
```
1. [CPU]  Select random ghost molecule (negligible - just pick index)
2. [CPU]  Generate random rotation matrix (9 floats - negligible)
3. [CPU]  Generate random insertion site (3 floats - negligible)
4. [CPU]  Upload rotation matrix + site to GPU (48 bytes - negligible)
5. [GPU]  launchRotateMolecule()                         ~0.01ms
6. [GPU]  launchTranslateMolecule()                      ~0.01ms
7. [GPU]  kApplyPBC()                                    ~0.01ms
8. [GPU]  launchAdjustLambda() → λ=0.998                ~0.01ms
9. [GPU]  evaluateTotalEnergyGPU() → E_initial          1.6ms (energy stays on GPU!)
10. [GPU]  launchAdjustLambda() → λ=1.0                  ~0.01ms
11. [GPU]  evaluateTotalEnergyGPU() → E_final            1.6ms (energy stays on GPU!)
12. [GPU]  launchComputeDeltaE()                         ~0.001ms
13. [GPU]  launchMetropolisAccept()                      ~0.001ms
14. [GPU]  launchConditionalRestore() (if rejected)      ~0.01ms
15. [GPU→CPU] Download acceptance flag (1 byte)          ~0.1ms (negligible!)
```

**Total GPU-only time:** ~3.3ms (vs current 36.5ms)
**Expected speedup:** **~11× per energy evaluation, ~22× per insertion/deletion move**

---

## 5. Implementation Roadmap

### Phase 1: Add Missing GPU Kernels (1-2 hours)
**File:** `src/Sampling/hpc_mc_moves.cu`

1. Add `kAdjustLambda` kernel
2. Add `kApplyPBC` kernel
3. Add `launchAdjustLambda()` wrapper
4. Add `launchApplyPBC()` wrapper

**File:** `src/Sampling/hpc_mc_moves.h`

5. Add function declarations

### Phase 2: Refactor `attemptInsertionGPU()` (2-3 hours)
**File:** `src/Sampling/gcmc_sampler.cpp`

Create new method `bool GCMCSystemSampler::attemptInsertionGPU()`:

```cpp
bool GCMCSystemSampler::attemptInsertionGPU() {
  // 1. Select ghost molecule (CPU - negligible)
  GCMCMolecule* ghost_mol = selectRandomGhostMolecule();

  // 2. Generate rotation matrix & insertion site (CPU - negligible)
  double rot_matrix[9];
  generateRandomRotationMatrix(rot_matrix);  // 9 floats
  double3 site = selectInsertionSite();       // 3 floats

  // 3. Upload to GPU (48 bytes - negligible)
  cudaMemcpy(d_rot_matrix_, rot_matrix, 9*sizeof(double), cudaMemcpyHostToDevice);

  // 4. Get device pointers
  PhaseSpaceWriter psw = phase_space_->data(HybridTargetLevel::DEVICE);
  int* d_atom_indices = ghost_mol->d_atom_indices;  // Already on GPU
  int n_atoms = ghost_mol->atom_indices.size();

  // 5. Calculate COG on GPU (downloads only 12 bytes)
  double cogx, cogy, cogz;
  launchCalculateCOG(n_atoms, d_atom_indices, psw.xcrd, psw.ycrd, psw.zcrd,
                     &cogx, &cogy, &cogz);

  // 6. Backup coordinates on GPU
  launchBackupCoordinates(n_atoms, d_atom_indices,
                          psw.xcrd, psw.ycrd, psw.zcrd,
                          d_saved_xcrd_, d_saved_ycrd_, d_saved_zcrd_);

  // 7. Apply rotation on GPU
  launchRotateMolecule(n_atoms, d_atom_indices, cogx, cogy, cogz,
                       d_rot_matrix_, psw.xcrd, psw.ycrd, psw.zcrd);

  // 8. Translate to insertion site on GPU
  double dx = site.x - cogx;
  double dy = site.y - cogy;
  double dz = site.z - cogz;
  launchTranslateMolecule(n_atoms, d_atom_indices, dx, dy, dz,
                          psw.xcrd, psw.ycrd, psw.zcrd);

  // 9. Apply PBC on GPU
  const double* box = phase_space_->getBoxDimensions();
  launchApplyPBC(n_atoms, d_atom_indices, box[0], box[1], box[2],
                 psw.xcrd, psw.ycrd, psw.zcrd);

  // 10. Set lambda=0 on GPU (ghost state)
  launchAdjustLambda(n_atoms, d_atom_indices, 0.0, 0.0,
                     lambda_vdw_.data(HybridTargetLevel::DEVICE),
                     lambda_ele_.data(HybridTargetLevel::DEVICE));

  // 11. Evaluate E_initial (energy stays on GPU!)
  evaluateTotalEnergyGPU(mc_energy_before_.data(HybridTargetLevel::DEVICE));

  // 12. Set lambda=0.998 on GPU (active state for interaction calculation)
  launchAdjustLambda(n_atoms, d_atom_indices, 0.998, 0.998,
                     lambda_vdw_.data(HybridTargetLevel::DEVICE),
                     lambda_ele_.data(HybridTargetLevel::DEVICE));

  // 13. Evaluate E_final (energy stays on GPU!)
  evaluateTotalEnergyGPU(mc_energy_after_.data(HybridTargetLevel::DEVICE));

  // 14. Restore lambda=0 on GPU
  launchAdjustLambda(n_atoms, d_atom_indices, 0.0, 0.0,
                     lambda_vdw_.data(HybridTargetLevel::DEVICE),
                     lambda_ele_.data(HybridTargetLevel::DEVICE));

  // 15. Compute ΔE on GPU
  launchComputeDeltaE(mc_energy_before_.data(HybridTargetLevel::DEVICE),
                      mc_energy_after_.data(HybridTargetLevel::DEVICE),
                      mc_delta_e_.data(HybridTargetLevel::DEVICE));

  // 16. Generate random number (CPU - negligible)
  double rand_val = rng_.uniformRandomNumber();
  cudaMemcpy(mc_random_number_.data(HybridTargetLevel::DEVICE),
             &rand_val, sizeof(double), cudaMemcpyHostToDevice);

  // 17. Metropolis acceptance on GPU
  double acc_prob_for_gcmc = std::exp(B_ - beta_ * ..);  // GCMC-specific factor
  launchMetropolisAccept(mc_delta_e_.data(HybridTargetLevel::DEVICE),
                         beta_,
                         mc_random_number_.data(HybridTargetLevel::DEVICE),
                         mc_accepted_.data(HybridTargetLevel::DEVICE));

  // 18. Conditional restore on GPU (restores coords if rejected)
  launchConditionalRestore(n_atoms, mc_accepted_.data(HybridTargetLevel::DEVICE),
                           d_atom_indices,
                           d_saved_xcrd_, d_saved_ycrd_, d_saved_zcrd_,
                           psw.xcrd, psw.ycrd, psw.zcrd);

  // 19. Download acceptance flag ONLY (1 byte - negligible!)
  int accepted_flag;
  mc_accepted_.download();
  accepted_flag = mc_accepted_.readHost(0);

  // 20. Update CPU-side bookkeeping
  if (accepted_flag) {
    launchAdjustLambda(n_atoms, d_atom_indices, 1.0, 1.0,
                       lambda_vdw_.data(HybridTargetLevel::DEVICE),
                       lambda_ele_.data(HybridTargetLevel::DEVICE));
    ghost_mol->status = GCMCMoleculeStatus::ACTIVE;
    N_active_++;
    // Update cached lists...
  }

  return (accepted_flag == 1);
}
```

### Phase 3: Refactor `attemptDeletionGPU()` (1-2 hours)
Similar pattern to insertion but simpler (no rotation/translation needed).

### Phase 4: Testing & Validation (2-3 hours)
1. Add timing instrumentation to GPU-only path
2. Run parity tests: GPU vs CPU results should match exactly
3. Measure speedup
4. Stress test with large systems (10,000+ ghosts)

### Phase 5: Migration & Cleanup (1 hour)
1. Replace old CPU-centric paths with GPU-accelerated versions
2. Remove obsolete coordinate download/upload calls
3. Update documentation

---

## 6. Expected Performance Gains

### Before (CPU-centric):
- **36.5ms per energy evaluation**
- **73ms per insertion/deletion move** (2 energy evals)
- **100 moves = 7.3 seconds**

### After (GPU-accelerated):
- **~3.3ms per energy evaluation** (~11× faster)
- **~6.6ms per insertion/deletion move** (~11× faster)
- **100 moves = 0.66 seconds** (11× faster)

### Scaling Benefits:
- Larger systems → bigger speedup (more atoms = more GPU parallelism)
- More ghost molecules → same speedup (GPU handles all efficiently)
- PBC vs vacuum → same speedup (kernel time stays ~1.6ms regardless)

---

## 7. Risk Mitigation

### Potential Issues:
1. **Random number generation on GPU** - Requires GPU RNG library (cuRAND)
   - *Mitigation:* Use CPU RNG for now (negligible cost - just 1-2 floats per move)

2. **Lambda-dependent work unit regeneration** - Might need GPU kernel updates
   - *Mitigation:* `ensureCoupledAtomList()` already handles this; keep on CPU if needed

3. **Debugging GPU kernels harder than CPU code**
   - *Mitigation:* Implement GPU path alongside old CPU path; validate with parity tests

4. **Memory allocation for new GPU arrays** (rotation matrix, PBC boxes, etc.)
   - *Mitigation:* Pre-allocate in constructor; reuse across moves

### Testing Strategy:
1. **Parity tests:** Run identical GCMC simulation with CPU and GPU paths, verify energies match
2. **Timing validation:** Confirm 10-15× speedup as predicted
3. **Stress tests:** 10,000 ghosts, 10,000 moves, ensure no memory leaks or crashes

---

## 8. Code Locations

### Files to Modify:
- `src/Sampling/hpc_mc_moves.cu` - Add 2 new kernels (lambda, PBC)
- `src/Sampling/hpc_mc_moves.h` - Add function declarations
- `src/Sampling/gcmc_sampler.cpp` - Add `attemptInsertionGPU()`, `attemptDeletionGPU()`
- `src/Sampling/gcmc_sampler.h` - Add method declarations

### Files for Reference:
- `src/Sampling/mc_mover.cpp:178,499` - Example GPU-accelerated MC moves
- `src/Sampling/hpc_mc_moves.cu:454-475` - Metropolis kernel implementation
- `src/Sampling/gcmc_sampler.cpp:1558-1646` - `evaluateTotalEnergyGPU()` (already GPU-only)

---

## 9. Alternative Approaches Considered

### Approach 1: Async Downloads (Rejected)
Use `cudaMemcpyAsync` to overlap energy downloads with other work.
- **Problem:** Still 34.9ms per download; can't overlap with serial MC logic
- **Verdict:** Won't achieve >2× speedup

### Approach 2: Batch Multiple Moves (Rejected)
Queue 100 moves, execute on GPU, download results.
- **Problem:** GCMC is inherently serial (each move depends on previous acceptance)
- **Verdict:** Not applicable to GCMC algorithm

### Approach 3: GPU-Only Path (SELECTED)
Keep ALL data on GPU, only download acceptance flag.
- **Advantages:**
  - 11× speedup per move
  - Leverages existing GPU infrastructure
  - Minimal new code (2 kernels)
- **Disadvantages:**
  - Requires careful testing
  - Slightly harder to debug
- **Verdict:** Best performance, infrastructure already exists

---

## 10. Conclusion

The 34.9ms energy download bottleneck can be eliminated by using the **existing GPU-accelerated MC infrastructure** that is already in the codebase but not used by GCMC insertion/deletion. Only 2 new GPU kernels are needed (lambda adjustment, PBC wrapping). Expected speedup: **~11× for GCMC moves**, which directly improves the 95.6% bottleneck identified in timing analysis.

**Recommendation:** Proceed with refactoring in 5 phases as outlined above. Estimated total implementation time: 8-12 hours including testing.

---

## Appendix A: Timing Data (Collected 2025-10-29)

```
# [TIMING] evaluateTotalEnergyFast() call 200:
#   Total:          40.912 ms (avg: 36.509 ms)
#   GPU kernel:     1.586 ms (avg: 1.614 ms)   ← 4.4%
#   Energy DL:      39.326 ms (avg: 34.894 ms) ← 95.6% BOTTLENECK
```

**System:** T4 GPU, GCMC with 1000 ghost MBN molecules, 100 moves
**Log file:** `T4-TEST/MBN_classical_gcmc.log`
**Command:** `DRY_RUN=0 bash run_classical_gcmc.sh`
