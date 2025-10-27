# Hybrid GCMC Runner

The `gcmc_hybrid.stormm` executable combines the GPU‑accelerated lambda‑dynamics kernels,
the instantaneous GCMC moves, and the rigid‑body Monte‑Carlo movers into a single workflow.
It takes a protein topology/coordinate pair plus a fragment topology/coordinate pair, freezes
the protein (all atomic masses set to 0.0), and runs GCMC with optional MC perturbations on the
fragment ghosts.

## Quick start

```bash
./apps/Gcmc/gcmc_hybrid.stormm \
  -p protein.prmtop \
  -c protein.inpcrd \
  --fragment-prmtop ligand.prmtop \
  --fragment-inpcrd ligand.inpcrd \
  --nghost 1000 \
  --moves 500 \
  --temp 300 \
  --mc-translation 1.0 \
  --mc-rotation 30.0 \
  --mc-frequency 5 \
  --output-dir runs/protein_ligand \
  -o protein_ligand_run
```

This will create `runs/protein_ligand/` (if needed) and write:

- `protein_ligand_run_gcmc.log` – detailed sampler log  
- `protein_ligand_run_ghosts.txt` – ghost occupancy history  
- `protein_ligand_run_occupancy.dat` – cycle vs. active fragment count  
- `protein_ligand_run_gpu_memory.dat` – (optional) GPU/RSS telemetry  

### Adaptive B schedule

Enable the three‑stage adaptive Adams‑B controller with:

```bash
  --adaptive-b \
  --stage1-moves 300 \
  --stage2-moves 300 \
  --stage3-moves 300 \
  --b-discovery 8.0 \
  --target-occupancy 0.5 \
  --coarse-rate 0.5 \
  --fine-rate 0.25 \
  --b-min -5.0 \
  --b-max 10.0
```

Omit `--adaptive-b` (or pass `--adaptive-b false`) to keep a constant B set by `--bvalue/-b`
and `--mu-ex`.

### Saving / reusing configurations

Instead of repeating a long command line you can point to a simple configuration file:

```
--config hybrid_run.cfg
```

`hybrid_run.cfg` is a plain text file; each non‑comment line contains the exact CLI flags you
would normally pass, one token per whitespace‑separated entry. Example:

```
# hybrid_run.cfg
-p protein.prmtop
-c protein.inpcrd
--fragment-prmtop ligand.prmtop
--fragment-inpcrd ligand.inpcrd
--nghost 1000
--moves 500
--temp 300
--mc-translation 1.0
--mc-rotation 30.0
--mc-frequency 5
-o protein_ligand_run
--log-memory true
```

The `--config` contents are expanded exactly as if you had typed them on the command line,
and you can still append or override options after the `--config` flag. Nested configuration
files are supported and relative paths are resolved against the including file. Cycles are
detected to prevent infinite recursion.

#### JSON configs

JSON files are also accepted—just point `--config` (or `--config-json`) at a `.json` file. Keys
map directly to CLI switches: omit the dashes to generate the long form (`"nghost": 1000` →
`--nghost 1000`) or supply the exact flag (`"-p": "protein.prmtop"`). Arrays repeat the flag,
and booleans become `true` / `false`. An optional `"args"` array lets you drop in raw tokens.

```json
{
  "config": "hybrid_common.cfg",
  "nghost": 16,
  "moves": 9,
  "adaptive-b": true,
  "stage1-moves": 3,
  "stage2-moves": 3,
  "stage3-moves": 3,
  "b-discovery": 6.0,
  "mc-frequency": 0,
  "o": "adaptive_run"
}
```

### Logging & telemetry

- `--output-dir <path>` places all logs in a dedicated directory (created if missing).
- `--log-memory` controls GPU / RSS telemetry. The `STORMM_LOG_MEMORY` environment variable
  (`0/1`, `true/false`, `on/off`, `yes/no`) can toggle the default without touching CLI
  scripts. Explicit command-line values still take precedence.
- Each run prints midpoint acceptance summaries (every ~10 % of the trajectory) and a final
  block of statistics covering overall/insert/delete acceptance, active fragment counts, and
  the terminal Adams‑B state.
- MC movers warn and disable themselves automatically when the amplitudes or frequencies are
  zero/negative, avoiding confusing “enabled but inert” setups.
- `--final-pdb auto` writes a concluding PDB snapshot (`output-prefix-final.pdb`). Pair with
  `--final-pdb-active-only` to keep only the currently active fragments; otherwise the entire
  system is exported.
- When adaptive B is disabled, `--constant-b-pdb-prefix debug/constant_b` dumps the first
  100 cycles as `…_###.pdb` files alongside the usual telemetry.
