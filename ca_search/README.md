# CA Search Tooling

This package is the first structured step toward searching for potentially complex
number-conserving binary Moore-neighborhood rules.

It covers two immediate needs:

1. exact filters for analytically simple microdynamics;
2. a batched simulator and cheap metric scaffold for later large-scale screening.

It now also contains the first exact reversibility-rejection pipeline for the
same rule families.

## Modules

- `lut.py`: LUT encoding helpers, rigid shift construction, and isolated particle / hole
  indexing conventions.
- `catalog.py`: loader for the checked-in JSON rule catalog used by the web visualizer.
- rule identifiers:
  - legacy positional `id` values are snapshot-specific,
  - `stableIndex` is deterministic within a given catalog,
  - `stableId` is the permanent LUT-derived identifier to use in new analysis.
- `simple_filters.py`: LUT-level classification for:
  - identity,
  - rigid shifts,
  - isolated particle velocity,
  - isolated hole velocity,
  - embedded von Neumann and diagonal-von-Neumann non-shift families.
- `simulator.py`: pairwise batched CA simulator with:
  - a verified `numpy` backend,
  - an optional `mlx` backend for Apple Silicon,
  - cheap metric collection,
  - throughput benchmarking.
- `build_motion_filter_specs.py`: code generator for the isolated particle / hole identity files.
- `reversibility.py`: exact early-stage reversibility checks:
  - sparse particle-sector injectivity,
  - sparse hole-sector injectivity,
  - exact full-torus bijectivity on small periodic grids.
- `reversibility_catalog.py`: batched catalog-wide version of the same exact
  early-stage reversibility screen.
- `raw_reversibility_screen.py`: single-pass streaming screen for the legacy
  `witek/worker_*.bin` raw solver outputs. It reads each 64-byte rule record
  once and applies the `4x4` particle-sector tests directly to the packed form.
- `density_classification.py`: scheduled 2D CA experiments for Fukś-style
  density-classification studies, including built-in 2D majority rules,
  embedded traffic preprocessors, and metrics for local density equalization.

## Exact Motion Filters

The generated filter family currently includes:

- `identities/isolated_particle_<velocity>.func`
- `identities/isolated_hole_<velocity>.func`

for all nine velocities:

- `static`
- `north`, `south`, `east`, `west`
- `northeast`, `northwest`, `southeast`, `southwest`

Their generated sparse systems are checked in as:

- `isolated_particle_<velocity>_equations.txt`
- `isolated_hole_<velocity>_equations.txt`

These are intended to be used as additional property systems on top of the
number-conserving base search.

## Usage

Simple analytical screening of the checked-in nonzero-mask catalog:

```bash
python3 -m ca_search.cli screen-simple
```

CPU benchmark:

```bash
python3 -m ca_search.cli benchmark --backend numpy --batch 128 --width 128 --height 128 --steps 128
```

Cheap metric collection:

```bash
python3 -m ca_search.cli metrics --backend numpy --batch 8 --width 64 --height 64 --steps 32
```

Run the first 2D density-classification study:

```bash
python3 -m ca_search.cli density-classification-study \
  --binary expanded_property_panel_nonzero.bin \
  --metadata expanded_property_panel_nonzero.json \
  --backend numpy \
  --width 64 --height 64 \
  --trials 32 \
  --probabilities 0.45 0.475 0.49 0.51 0.525 0.55 \
  --preprocess-short 32 \
  --preprocess-long 64 \
  --majority-tail 32 \
  --out density_classification_study.json
```

The current note for this track is:

- [DENSITY_CLASSIFICATION_2D_NOTE.md](/Users/witoldbolt/Documents/solvw/omp/DENSITY_CLASSIFICATION_2D_NOTE.md)

Rank candidate NCCAs as density-equalizing preprocessors:

```bash
python3 -m ca_search.cli density-preprocessor-screen \
  --binary expanded_property_panel_nonzero.bin \
  --metadata expanded_property_panel_nonzero.json \
  --backend numpy \
  --width 32 --height 32 \
  --probability 0.49 \
  --trials 16 \
  --preprocess-steps 32 \
  --max-rules 2000 \
  --top-k 25 \
  --out density_preprocessor_screen.json
```

Important: this screen measures preprocessing quality only.  The current note
shows that many top-ranked preprocessors are one-sided and fail badly when the
initial density is mirrored from `p` to `1-p`.  So the next-stage search
should use balanced paired-density objectives rather than a single-density
ranking alone.

Rank candidate NCCAs with a balanced paired-density score:

```bash
python3 -m ca_search.cli density-preprocessor-balanced-screen \
  --binary expanded_property_panel_nonzero.bin \
  --metadata expanded_property_panel_nonzero.json \
  --backend numpy \
  --width 32 --height 32 \
  --probabilities 0.49 0.51 \
  --trials 16 \
  --preprocess-steps 32 \
  --majority-tail 32 \
  --include-property outer_monotone orthogonal_monotone center_blind \
  --sample-property diagonal_monotone=2000 \
  --selection-seed 0 \
  --top-k 25 \
  --out density_preprocessor_balanced_subset_screen.json
```

This command evaluates each candidate at both `p` and `1-p`, includes the
majority phase in the ranking, and penalizes side bias directly.

Rank candidate NCCAs directly under the repeated-block objective:

```bash
python3 -m ca_search.cli density-repeated-block-screen \
  --binary expanded_property_panel_nonzero.bin \
  --metadata expanded_property_panel_nonzero.json \
  --backend numpy \
  --width 32 --height 32 \
  --probabilities 0.49 0.51 \
  --trials 16 \
  --preprocess-steps 8 \
  --amplifier-steps 8 \
  --repetitions 16 \
  --amplifier majority_vn \
  --include-property outer_monotone orthogonal_monotone center_blind \
  --sample-property diagonal_monotone=2000 \
  --selection-seed 0 \
  --top-k 40 \
  --out density_repeated_block_balanced_subset_vn.json
```

This is now best treated as a **CA baseline** command. Repeated blocks are
empirically useful, but they still define an ordinary binary CA when viewed
stroboscopically, so they are not the final Fukś-style target.

Search directly for a finite globally synchronized switch schedule
`F^(T1) -> G^(T2)`:

```bash
python3 -m ca_search.cli density-finite-switch-screen \
  --binary expanded_property_panel_nonzero.bin \
  --metadata expanded_property_panel_nonzero.json \
  --backend numpy \
  --width 32 --height 32 \
  --probabilities 0.49 0.51 \
  --trials 16 \
  --preprocess-steps 16 32 64 96 \
  --amplifier-steps 16 32 64 96 \
  --amplifier majority_vn \
  --include-property outer_monotone orthogonal_monotone center_blind \
  --sample-property diagonal_monotone=2000 \
  --selection-seed 0 \
  --top-k 40 \
  --out density_finite_switch_balanced_subset_vn.json
```

This is the current beyond-CA DCP search command, since it searches a fixed
finite number of globally synchronized phase switches rather than periodic
repetition.

Search explicitly for checkerboardizing or anti-ferromagnetic prephases:

```bash
python3 -m ca_search.cli density-checkerboard-prephase-screen \
  --binary expanded_property_panel_nonzero.bin \
  --metadata expanded_property_panel_nonzero.json \
  --backend numpy \
  --width 32 --height 32 \
  --probabilities 0.49 0.51 \
  --trials 16 \
  --steps-per-rule 1 2 \
  --block-repetitions 16 32 64 \
  --amplifier majority_vn \
  --amplifier-steps 32 64 \
  --top-k 30 \
  --out density_checkerboard_phase_search.json
```

This command ranks finite prephases by a mixed objective:
- downstream classification quality after one final majority switch,
- checkerboard `2x2` density,
- orthogonal edge-disagreement,
- staggered checkerboard alignment.

Rule-selector note for the analysis scripts:

- bare integers still mean legacy snapshot indices for backward compatibility;
- `rank:<n>` means deterministic stable rank within the current catalog;
- `stable:<hex-prefix>` or `sid:<hex-prefix>` means the LUT-derived stable identifier.

Run an exact reversibility probe against a binary catalog rule:

```bash
python3 -m ca_search.cli reversibility-probe \
  --binary expanded_property_panel_nonzero.bin \
  --metadata expanded_property_panel_nonzero.json \
  --rule sid:029b09cea
```

This command:

- skips the trivial reversible rules `identity` and rigid shifts,
- tests injectivity on small fixed-population sectors,
- then tests exact bijectivity on small periodic tori,
- and stops on the first exact collision witness.

The current implementation focuses on reviewable exact tests rather than
generic undecidable reversibility logic. See
[REVERSIBILITY_PLAN.md](/Users/witoldbolt/Documents/solvw/omp/REVERSIBILITY_PLAN.md)
for the staged roadmap.

Run the same early exact screen over an entire binary catalog:

```bash
python3 -m ca_search.cli reversibility-screen-catalog \
  --binary expanded_property_panel_nonzero.bin \
  --metadata expanded_property_panel_nonzero.json \
  --sector-width 4 --sector-height 4 \
  --particle-populations 2 3 \
  --hole-populations 1 2 \
  --torus 3 3 \
  --out expanded_property_panel_reversibility_screen.json
```

This batched screen is intended as the first high-throughput filter before any
deeper inverse-radius or SAT-based analysis. On the current expanded
`133713`-rule nonzero-mask catalog, the checked-in result file leaves only `16`
nontrivial survivors after these exact early stages.

The stricter follow-up artifact
[expanded_property_panel_reversibility_screen_strict.json](/Users/witoldbolt/Documents/solvw/omp/expanded_property_panel_reversibility_screen_strict.json)
adds the `4`-particle sector on `4x4` and eliminates those `16` survivors as
well, leaving no nontrivial survivors within the expanded nonzero-mask catalog.

Run the same fast screen directly over the full legacy raw solution set:

```bash
python3 -m ca_search.cli reversibility-screen-raw \
  --input witek \
  --stage-populations 2 3 4 \
  --records-per-chunk 16384 \
  --survivor-out witek_reversibility_survivors.bin \
  --out witek_reversibility_screen.json
```

This raw scanner avoids unpacking all `510` rule bits. For the current
`4x4` particle-sector stages it touches only:

- `45` LUT entries for `k = 2`,
- `129` LUT entries for `k = 3`,
- `255` LUT entries for `k = 4`.

The checked-in full run:

- [witek_reversibility_screen.json](/Users/witoldbolt/Documents/solvw/omp/witek_reversibility_screen.json)
- [witek_reversibility_survivors.bin](/Users/witoldbolt/Documents/solvw/omp/witek_reversibility_survivors.bin)
- [witek_reversibility_survivors_strict_details.json](/Users/witoldbolt/Documents/solvw/omp/witek_reversibility_survivors_strict_details.json)

shows that:

- the fast streamed screen reduces all `147309433` rules to `65` candidates,
- the stricter exact follow-up on those `65` leaves only `9` survivors,
- and those `9` are exactly the identity rule plus the eight rigid shifts.

Generate or refresh the isolated motion specs:

```bash
python3 -m ca_search.build_motion_filter_specs
```

## MLX Notes

The optional MLX backend is exposed explicitly through `--backend mlx` and the `search`
dependency extra in `pyproject.toml`.

In this execution environment, importing MLX aborts during Metal device initialization, so the
`auto` backend is intentionally conservative and defaults to `numpy`. On a normal interactive
Apple Silicon session with a usable Metal device, the same code path should be tested again and
benchmarked before relying on it for the main large-scale search.
