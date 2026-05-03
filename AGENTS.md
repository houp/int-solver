# AGENTS

This file is the entry-point briefing for AI agents working in this repo.
For human-facing documentation, see [`README.md`](README.md).

## Repository purpose (short)

A research workspace on **2D binary number-conserving cellular automata
(NCCAs)** with the Moore neighborhood. Three intertwined lines of work:

1. **Catalogue construction.** Functional identities → linear system →
   exhaustive enumeration of all `147,309,433` NCCAs → property-tagged
   catalogues.
2. **Reversibility screening.** Exact small-torus particle-sector tests on
   the full catalogue. Only the identity + 8 rigid shifts survive.
3. **2D density classification.** Search for a Fukś-style schedule that
   classifies arbitrary 2D inputs by their global majority. Current best:
   an L-scaled stochastic classifier (see §5 of the technical report).

The compiled write-up is [`docs/technical_report.pdf`](docs/technical_report.pdf).

## Repository layout (the part you need to remember)

```
src/                — solver pipeline (C++ solvers + python pre/post-processing)
classifier/         — 2D density-classification scripts (current research line)
ca_search/          — python package: catalog loading, simulators, screens
identities/         — .func specs (input to the equation generator)
catalogs/           — enumerated rule catalogs (.bin + .json metadata)
equations/          — generated linear systems (raw / dedup / matrices subdirs)
results/            — JSON experiment outputs (subfoldered by topic)
snapshots/          — captured lattice states (.npz, snapshot subdirs)
docs/               — markdown notes + LaTeX report + bibliography
tests/              — pytest suite (37 tests; run from repo root)
witek/              — raw 9.4 GiB solver outputs (do not commit; do not move)
solutions_zstd/     — zstd-compressed solver outputs
web-visualization/  — browser-based rule explorer (data + UI)
```

Top-level files: `README.md`, `AGENTS.md`, `pyproject.toml`, `uv.lock`,
`build.sh`, `.latexmkrc`.

## Operating conventions

### Run from repo root

All Python scripts in `classifier/` and `src/` self-bootstrap their
`sys.path` via
```python
_REPO_ROOT = Path(__file__).resolve().parent.parent
_sys.path.insert(0, str(_REPO_ROOT))
```
which makes `ca_search` and sibling-script imports resolve regardless of
`cwd`. Catalog paths default to `catalogs/expanded_property_panel_nonzero.{bin,json}`
via a `_CATALOG_DIR` constant.

For consistency, prefer running commands from the repo root:
```bash
python classifier/scaled_schedule.py --grids 192 ...
```

### Build the C++ solvers

```bash
./build.sh src/solve_omp_opt
CXXFLAGS=-DUSE_ZSTD ./build.sh src/solve_omp_opt_zstd
```
The argument is the target path **without** the `.cpp` suffix. The script
expects `$1.cpp` to exist.

### Build the LaTeX report

```bash
latexmk             # uses .latexmkrc → xelatex → docs/technical_report.pdf
latexmk -c          # clean aux files; PDF kept
```

The `.latexmkrc` pins `pdf_mode=5` (xelatex) and routes outputs to `docs/`.
**Use latexmk + xelatex** for all LaTeX builds; do not invoke `pdflatex`
directly.

### Run the tests

```bash
pytest                      # uses pyproject testpaths = ["tests"]
```

37 tests, all passing as of 2026-04-26. Test collection is restricted to
`tests/` to avoid pytest accidentally picking up
`ca_search/reversibility.py:test_sector_injective` (which is a public
helper named with the `test_` prefix, not a pytest fixture target).

### Result file conventions

- New experiment scripts should write their JSON output via an `--output`
  flag and default to a clear filename.
- When committing or organizing results, place them under
  `results/<category>/`. Subcategories already in use:
  `density_classification/2026-04-24/`, `density_classification/earlier/`,
  `reversibility/`, `catalog_studies/`.
- Snapshots (`.npz`) go under `snapshots/`.

## Things to know that are not obvious

### The structural impossibility theorem (and why we use stochastic noise)

Any **shift-invariant deterministic Moore-neighborhood rule** F maps a
horizontal-stripe configuration to a stripe configuration (possibly its
complement). For NCCAs, density conservation forces preserve-or-complement
only. The same holds for vertical stripes and various checkerboard-like
patterns. **Tipped stripes** (density `1/2 + 1/L²`) are therefore permanent
obstructions for any pure deterministic shift-invariant schedule on 2D
DCP — a sharpening of the Land–Belew (1995) impossibility.

The current classifier escapes this by using **density-conserving random
pair swaps** as the stochastic component. Bernoulli per-cell flip noise
does not work: it destroys the global-density bias that distinguishes
near-tied inputs. Random swaps preserve that bias exactly per realization.

Full proof + experiments: §5 of the technical report;
[`docs/STOCHASTIC_CLASSIFIER_NOTE.md`](docs/STOCHASTIC_CLASSIFIER_NOTE.md)
for the lab journal.

### The repeated-block trap (do not confuse with finite-switch)

`[F^k M^k]^m` (periodic alternation of an NCCA `F` with a majority `M`) is
**indistinguishable from a single CA at larger effective radius** when
viewed stroboscopically. Per the methodological note in
[`docs/DENSITY_CLASSIFICATION_IDEAS.md`](docs/DENSITY_CLASSIFICATION_IDEAS.md),
this is a *CA baseline track* and cannot perfectly solve 2D DCP. A genuine
Fukś-style schedule has only a **finite number of global rule switches**.

### MLX vs NumPy

All density-classification scripts default to `--backend mlx`. On Apple
Silicon Metal, the pairwise-step kernel is 10–25× faster than NumPy. Some
scripts that pre-compute things in numpy (e.g. cluster labeling via
`scipy.ndimage.label`) keep numpy for the post-processing step but use
MLX for the hot CA loop.

Python 3.14 was tested but is **slower** than 3.12 here because the hot
loop is GPU dispatch — interpreter improvements don't help. Stay on 3.12
in `.venv`.

### `witek/` and `solutions_zstd/`

These contain the raw and compressed solver outputs (about `9.4 GiB` and
`1.3 GiB` respectively). They are derived artifacts; do not edit. The
streaming reversibility screen reads from `witek/` directly without
materializing.

### Original AGENTS.md

This file was inadvertently overwritten on 2026-04-26 during the directory
reorganization (an `mv AGENTS.md README.md` command ran when both files
were intended to be preserved). The original was not under git tracking,
so the lossy reconstruction here is based on the human author's memory of
its content plus the reorganized state of the repo. If the user has a copy
of the prior AGENTS.md, it should replace this file.

## Files that matter (cheat sheet)

| Path | Why |
|---|---|
| [`src/generate_functional_equation_system.py`](src/generate_functional_equation_system.py) | source of truth: identity → linear system |
| [`src/solve_omp_opt.cpp`](src/solve_omp_opt.cpp) | main solver implementation |
| [`identities/`](identities/) | identity specs; one `.func` per family |
| [`ca_search/cli.py`](ca_search/cli.py) | `python -m ca_search.cli ...` entry point |
| [`classifier/scaled_schedule.py`](classifier/scaled_schedule.py) | current best 2D DCP classifier |
| [`classifier/conservative_noise.py`](classifier/conservative_noise.py) | density-conserving swap noise primitive |
| [`classifier/amplifier_library.py`](classifier/amplifier_library.py) | local-majority rules library |
| [`docs/technical_report.tex`](docs/technical_report.tex) | full write-up |
| [`docs/STOCHASTIC_CLASSIFIER_NOTE.md`](docs/STOCHASTIC_CLASSIFIER_NOTE.md) | lab journal of the DCP work |
| [`docs/DENSITY_CLASSIFICATION_IDEAS.md`](docs/DENSITY_CLASSIFICATION_IDEAS.md) | active idea ledger (status-tagged) |
| [`pyproject.toml`](pyproject.toml) | uv project metadata + pytest config |

## Things NOT to do without explicit permission

- Do not modify `witek/` or `solutions_zstd/` (large derived data).
- Do not commit or rewrite `equations/dedup/dominik_equations.txt` — that
  is the human-prepared solver input.
- Do not rewrite generated equation files by hand — regenerate via
  `src/generate_functional_equation_system.py`.
- Do not switch the LaTeX engine to `pdflatex`/`lualatex` — the project
  uses **xelatex via latexmk**.
- Do not move catalog `.bin` / `.json` files out of `catalogs/` — many
  scripts default to that path.
- Do not commit `.aux`, `.log`, `.bbl`, `.fdb_latexmk` etc. (`latexmk -c`
  cleans them).
