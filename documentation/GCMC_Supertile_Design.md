# GCMC / Lambda Dynamics Supertiles Design Notes

This document captures the ongoing work to extend STORMM’s GPU path for **grand canonical Monte Carlo (GCMC)** and **lambda dynamics** by introducing supertile-based non-bonded kernels. It serves both as internal design guidance and as a starting point for a methods write-up once the supertile implementation is complete.

## Motivation

- **High ghost counts:** Modern GCMC workflows routinely stage 3 000–10 000 ghost fragments. Each fragment participates in dense, all-to-all electrostatics and van-der Waals calculations whenever it becomes partially coupled. The legacy 16×16 “tile group” kernels explode in count (millions of tiles), overloading the scheduler and repeatedly reallocating GPU scratch space.
- **Lambda integration:** The lambda-scaled non-bonded kernels evaluate the same dense matrices to compute work values. Their performance degrades in sync with the GCMC path, and repeated host/device synchronisation becomes a bottleneck.
- **VRAM fragmentation concerns:** Heavy ghost pools trigger frequent `cudaMalloc` / `cudaFree` cycles when large work units are required. Supertile kernels reduce the number of active work units, easing pressure on the hybrid caches used by both GCMC and lambda dynamics.

## Current Architecture (Tile Groups)

| Concept | Tile Groups |
| ------- | ----------- |
| Tile size | 16×16 interactions |
| Work-unit abstract | 64 integers (imports, per-tile instructions, exclusion lists) |
| Kernel prefix | `ktgd...`, `ktgds...`, `ktgf...`, `ktgfs...` |
| Launch shape | 256 threads per block, one tile at a time |
| Usage | Default for all GCMC, lambda dynamics, minimization, MD |

The manager enumerates every tile explicitly, which keeps masking flexible but scales poorly for dense systems. Once an insertion requires more than 64 tiles, STORMM currently warns and falls back to tile groups because no supertile kernel exists.

## Supertile Design

| Concept | Supertile Target |
| ------- | ---------------- |
| Coverage | Single work unit spans 256×256 interactions (implicit 16×16 sub-tiles) |
| Abstract | 8 integers (abscissa start/length, ordinate start/length, system id, accumulator masks) |
| Kernel prefix | `ksts...` (double, split), `kstw...` (double, whole), `ksff...` (single, energy-only), `kssf...` / `kswf...` (single precision force variants) |
| Launch shape | 512+ threads per block, iterating through 16×16 sub-tiles internally |

Key characteristics:

1. **Implicit tile enumeration:** Instead of recording each 16×16 tile, the kernel loops over all sub-tiles derived from the 8-entry abstract. Exclusions use the existing static mask synthesis (`supertile_map_idx`, `tile_map_idx`) to skip masked tiles at runtime.
2. **Shared-memory footprint:** Forces, charges, and (optionally) GB intermediates remain in shared memory, but allocations are based on `supertile_length` (256) rather than individual tile groups.
3. **Naming consistency:** The new prefixes are registered in `CoreKlManager`. Tile-group kernels remain untouched, guaranteeing backward compatibility during rollout.

### Kernel Matrix

| Precision | Force? | Energy? | Accumulation | Vacuum | GB | GBNeck | Clash Forgiven |
| --------- | ------ | ------- | ------------ | ------ | -- | ------ | -------------- |
| Double    | No     | Yes     | N/A          | `kstfVacuumEnergy` | `kstfGBEnergy` | `kstfGBNeckEnergy` | `...NonClash` suffix |
| Double    | Yes    | No/Yes  | Split / Whole | `kstsVacuumForce`, `kstsVacuumForceEnergy`, `kstwVacuumForce`, `kstwVacuumForceEnergy` | analogous GB variants | analogous GBNeck variants | `...NonClash` suffix |
| Single    | No     | Yes     | N/A          | `ksffVacuumEnergy` | `ksffGBEnergy` | `ksffGBNeckEnergy` | `...NonClash` suffix |
| Single    | Yes    | No/Yes  | Split / Whole | `kssfVacuumForce`, `kssfVacuumForceEnergy`, `kswfVacuumForce`, `kswfVacuumForceEnergy` | analogous GB variants | analogous GBNeck variants | `...NonClash` suffix |

Only the double-precision, energy-only (vacuum, clash-aware) kernel is required to resolve the immediate GCMC crash, but the table documents the complete matrix for future work.

## Integration Plan

1. **Phase 1 (complete):**  
   - Centralise kernel naming logic in `CoreKlManager`.  
   - Register supertile prefixes without changing runtime behaviour.

2. **Phase 2 (in progress):**  
   - Implement double-precision, energy-only supertile kernels (`kstfVacuumEnergy`, `kstfVacuumEnergyNonClash`).  
   - Wire `queryNonbondedKernelRequirements` and the launch wrappers to call these kernels for `NbwuKind::SUPERTILES`.  
   - Ensure lambda dynamics retains access to the same kernels (they share the non-bonded launch path).

3. **Phase 3:**  
   - Implement split-force supertile kernels (double precision first).  
   - Extend GB / GBNeck variants and clash-forgiven launch paths.  
   - Update `MolecularMechanicsControls::primeWorkUnitCounters` so the progress counters are correct for the new launch shapes.

4. **Phase 4:**  
   - Re-enable supertile work-unit synthesis in `AtomGraphSynthesis::loadNonbondedWorkUnits`.  
   - Remove the current “falling back to TILE_GROUPS” warning once force kernels are stable.  
   - Benchmark GCMC + lambda workloads (e.g., benzene in 3GUK) with 3 000–10 000 ghosts.

## Interaction with GCMC and Lambda Dynamics

- **Shared control flow:** Both paths rely on `CoreKlManager::launchNonbonded()` and the same `SyNonbondedKit`, so supertile kernels immediately benefit GCMC insertions, λ-schedule propagation, and any hybrid MD/MC loops.
- **Cache usage:** Reducing the number of work units limits Hybrid cache churn (`CacheResource`) and avoids the `cudaFree` failures observed with `work_unit_prog_data`.
- **Telemetry:** The hybrid runner already supports `--log-memory` and per-cycle timers. These should remain optional (off by default) to keep hot loops free of extra device synchronisations.

## Performance Expectations

- **Launch reduction:** Switching dense regions to 256×256 supertiles cuts the number of kernel launches by ~256× per dimension (e.g., 16 384 tile-group launches → 256 supertile launches).
- **Occupancy:** Initial tuning targets 512 or 768 threads per block to maintain high occupancy on Ampere (sm_86) while fitting the expanded shared-memory footprint.
- **Throughput:** Early tests (tile-group fallback only) show GCMC cycles dropping from <5 cycles before crash to 17 cycles after rebalancing. Supertile kernels should extend this further, ultimately removing the crash while preserving GPU residency.

## Testing Strategy

1. Unit-level checks:
   - CUDA `ctest` patterns that isolate `kstsVacuumEnergy` vs. CPU reference.
   - Regression on lambda work integrals and acceptance statistics.
2. Integration:
   - Existing `gcmc_hybrid_constant` / `gcmc_hybrid_adaptive` CTest entries.  
   - Targeted high-ghost GCMC runs (`benzene_5000ghosts`) with `--log-memory` off/on.
3. Performance profiling:
   - `nvprof` / `nsys` to confirm block occupancy and shared-memory usage.
   - Compare kernel counts before/after supertile enablement.

## Future Work

- Full force/energy parity for GB / GBNeck and clash-forgiven modes.
- Adaptive selection between tile groups, supertiles, and honeycomb neighbor lists based on system sparsity.
- Additional automation in the hybrid runner (YAML/JSON configs, output directory management, memory logging toggles).
- Extended documentation for manuscript preparation (method, benchmarks, applications).

## References & Further Reading

- **Source files:**  
  - `src/Accelerator/core_kernel_manager.cpp` – kernel registry and naming.  
  - `src/Potential/nonbonded_potential_tilegroups.cui` – reference implementation for the new kernel bodies.  
  - `src/Synthesis/nonbonded_workunit.cpp` – supertile abstract construction.  
  - `apps/Gcmc/src/gcmc_hybrid_runner.cpp` – hybrid GCMC/λ application using the kernels.
- **Existing notes:** `NOTES_ON_GCMC.md` (repository root) and hybrid runner README additions.

---

_Maintained by the GCMC / Lambda Dynamics development effort. Contributions and benchmark reports are welcome (open an issue or PR referencing this document)._ 
