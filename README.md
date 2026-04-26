# OMP — Number-Conserving 2D Cellular Automata Workspace

This repository is a research workspace for **two-dimensional binary
number-conserving cellular automata (NCCAs)** with the Moore neighborhood.
It contains:

1. A pipeline that turns functional identities into a binary linear system
   and enumerates all 0/1 solutions with a custom OpenMP-parallel C++ solver.
2. A property-classification catalogue of the resulting `147,309,433`-rule
   class against ~30 structural identity families.
3. An exact reversibility screen on the full catalogue (only the identity
   and 8 rigid shifts survive).
4. A program of two-dimensional density-classification experiments,
   culminating in an L-scaled stochastic classifier that achieves 100%
   correct consensus at grid sizes 192×192 and larger.

The full technical write-up is at
[`docs/technical_report.tex`](docs/technical_report.tex) and its compiled
PDF [`docs/technical_report.pdf`](docs/technical_report.pdf).

---

## Repository Layout

```
.
├── README.md                 ← you are here
├── AGENTS.md                 ← guidance for AI agents working in this repo
├── pyproject.toml            ← uv-managed Python project (numpy, mlx, scipy, ...)
├── uv.lock
├── build.sh                  ← C++ solver build helper (see below)
├── .latexmkrc                ← latexmk config (xelatex; builds docs/technical_report.tex)
│
├── src/                      ← Solver pipeline (C++ + Python)
│   ├── solve_omp.cpp         baseline OpenMP solver
│   ├── solve_omp_opt.cpp     optimized solver (gap pruning, Zstd, etc.)
│   ├── generate_functional_equation_system.py  identity → linear system
│   ├── list_eqn.py           thin compatibility wrapper
│   └── decode_solutions.py   raw / Zstd output decoder
│
├── identities/               ← Functional-identity specs (.func files) — input to generator
├── ca_search/                ← Python search & analysis package (catalogs, simulators, screens)
│   ├── cli.py                python -m ca_search.cli ...
│   ├── simulator.py          numpy + MLX simulator backends
│   ├── density_classification.py
│   ├── reversibility.py, reversibility_catalog.py
│   └── ...
│
├── classifier/               ← 2D density-classification experiments (Apr 2026)
│   ├── amplifier_library.py  sub-Moore + radius-2+ majorities
│   ├── scaled_schedule.py    final L-scaled stochastic classifier
│   ├── conservative_noise.py density-conserving random-swap noise
│   ├── adversarial_realistic.py adversarial structured-input battery
│   ├── ...                   18 more experiment scripts; see docs/STOCHASTIC_CLASSIFIER_NOTE.md
│
├── catalogs/                 ← Enumerated rule catalogs (binary + JSON metadata)
│   ├── expanded_property_panel_nonzero.{bin,json}    ~133k rules with property tags
│   ├── full_property_panel_nonzero.{bin,json}
│   └── full_property_panel_nonzero_fixed.{bin,json}
│
├── equations/                ← Generated equation systems (input + by-property)
│   ├── raw/                  pre-deduplication identity instantiations
│   ├── dedup/                deduplicated sparse equations (.txt)
│   └── matrices/             dense CSV exports
│
├── results/                  ← All experiment result files (JSON)
│   ├── density_classification/
│   │   ├── 2026-04-24/      stochastic-classifier journey (this run)
│   │   └── earlier/          previous deterministic runs
│   ├── reversibility/        catalog reversibility-screen results
│   └── catalog_studies/      property-panel exploration, candidate analyses
│
├── snapshots/                ← Captured lattice states (.npz, snapshot dirs)
│
├── docs/                     ← Markdown notes + LaTeX report
│   ├── technical_report.tex / .pdf / references.bib
│   ├── DENSITY_CLASSIFICATION_2D_NOTE.md
│   ├── DENSITY_CLASSIFICATION_IDEAS.md
│   ├── STOCHASTIC_CLASSIFIER_NOTE.md   detailed lab journal
│   ├── MECHANISM_FINDINGS.md
│   ├── SEARCH_PLAN.md, REVERSIBILITY_PLAN.md
│   ├── candidate_deep_analysis.md, expanded_property_panel_exploration.md
│   ├── focused_rule_study.md, mechanistic_rule_study.md
│   ├── object_atlas_study.md, followup_rule_study_85932_98759.md
│
├── tests/                    ← pytest suite (covers solver pipeline + ca_search)
├── web-visualization/        ← Browser-facing rule explorer (data + UI)
├── witek/                    ← Raw uncompressed solver output shards (~9.4 GiB)
├── solutions_zstd/           ← Zstandard-compressed solver outputs
└── .venv/                    ← (gitignored) Python virtualenv
```

---

## Quick Start

### Python environment

The project uses `uv` for dependency management.

```bash
# create + populate venv (needs `uv` installed)
uv sync --extra search --extra analysis --extra compression

# activate
source .venv/bin/activate
```

### Build the C++ solvers

```bash
./build.sh src/solve_omp           # baseline
./build.sh src/solve_omp_opt       # optimized
CXXFLAGS=-DUSE_ZSTD ./build.sh src/solve_omp_opt_zstd  # with Zstd output
```

Requirements: macOS Homebrew `libomp`, or Linux g++ with OpenMP. C++23.

### Generate the equation system

```bash
python src/generate_functional_equation_system.py \
    identities/number_conserving.func \
    --sparse-out equations/dedup/simplified_equations.txt \
    --matrix-out equations/matrices/simplified_equations_matrix.csv \
    --raw-out    equations/raw/all_512_raw_equations.txt
```

### Solve

```bash
./src/solve_omp_opt equations/dedup/dominik_equations.txt \
    --threads 12 --spawn-depth 24 --count-only
```

### Decode raw / compressed solutions

```bash
python src/decode_solutions.py <solutions_dir_or_file> <nvars>
```

### Run the density-classification pipeline

The current best stochastic classifier (100% correct consensus at L ≥ 192
on random + structured-adversarial inputs):

```bash
python classifier/scaled_schedule.py \
    --grids 64 128 192 256 384 512 \
    --trials-per-side-random 64 \
    --adv-densities 0.51 0.52 0.55 \
    --output results/density_classification/scaled_validation.json
```

The schedule is

```
F^{L/2}  [moore81^{L}  F^{L/4}  swap^{8L}]^{2}  moore81^{4L}
```

with `F = sid:58ed6b657afb`, `moore81` the radius-2 Moore majority, and
`swap^K` density-conserving random pair swaps. See
[`docs/STOCHASTIC_CLASSIFIER_NOTE.md`](docs/STOCHASTIC_CLASSIFIER_NOTE.md)
for the full investigation history (mechanism probes, failed approaches,
the structural impossibility theorem for shift-invariant rules, the
literature review, and the path to the final result).

### Run the catalog reversibility screen

```bash
python -m ca_search.cli reversibility-screen-catalog \
    --binary catalogs/expanded_property_panel_nonzero.bin \
    --metadata catalogs/expanded_property_panel_nonzero.json
```

### Build the technical report

```bash
latexmk            # uses .latexmkrc; output → docs/technical_report.pdf
latexmk -c         # clean aux files (keeps PDF)
```

The `.latexmkrc` is preconfigured for **xelatex** (per project preference)
and outputs to `docs/`.

### Run the test suite

```bash
pytest             # 37 tests across solver pipeline + ca_search internals
```

Test discovery is restricted to `tests/` (configured in `pyproject.toml`).

---

## Headline Results

### Property-classification catalogue

`147,309,433` solutions of the number-conserving 9-bit identity. After
property matching across ~30 structural families, the
`expanded_property_panel_nonzero` catalogue contains `133,713` rules with
nonzero property masks. Full results in §3 of the technical report.

### Reversibility screen

Streaming exact screen of the full `147M`-rule class (`witek/`). After
4×4-torus particle-sector tests (`k=2,3,4`) plus hole sectors, the only
surviving rules are the **identity and the 8 rigid one-step shifts**. No
nontrivial reversible rule exists in this class. (§4 of the report.)

### 2D density classification

The L-scaled stochastic classifier achieves **100% correct consensus** on
both random Bernoulli inputs (density 0.49/0.51) **and** a 30-case
structured-adversarial battery (stripes, checkerboards, block-checkers,
half-half splits at density margins ≥1%) for grid sizes
`L ∈ {192, 256, 384, 512}`. Total work per trial ≈ `O(L)` deterministic CA
steps plus `O(L)` random swaps, substantially faster than Fatès–Regnault
(2016)'s `O(L²)` 2D construction. (§5 of the report.)

The structural theorem behind the construction: any shift-invariant
deterministic Moore-neighborhood rule maps a horizontal-stripe
configuration to a stripe (possibly complemented). For NCCAs, density
conservation forces preserve-or-complement only. Hence pure deterministic
schedules in this framework cannot escape stripe-tipped initial
configurations — a sharper impossibility than Land–Belew 1995 — and
density-conserving stochastic perturbation is necessary.

---

## Conventions

- **Run scripts from the repo root.** All Python scripts (in `classifier/`
  and `src/`) detect the repo root via `Path(__file__).resolve().parent.parent`
  and resolve catalog paths and ca_search imports accordingly. They will
  work regardless of cwd, but documented commands above assume cwd=root.
- **Catalog access.** Default catalog paths in `classifier/*.py` resolve
  to `catalogs/expanded_property_panel_nonzero.{bin,json}`. Override with
  `--binary` / `--metadata` flags.
- **Result outputs.** Scripts write JSON to `--output` (default to cwd or
  a script-named file). Place outputs under `results/<category>/` to keep
  the layout tidy.
- **MLX vs NumPy.** All density-classification scripts default to
  `--backend mlx`. Apple Silicon Metal GPU is ~10–25× faster than NumPy on
  the pairwise-step kernel; see §5.10 of the report. NumPy is available as
  a fallback (`--backend numpy`).
- **Build artifacts (LaTeX).** `.latexmkrc` directs aux files to `docs/`.
  `latexmk -c` from the repo root cleans them.

---

## Where to look next

| Question | File |
|---|---|
| What are the high-level project goals? | This README + [`docs/SEARCH_PLAN.md`](docs/SEARCH_PLAN.md) |
| How do I navigate as an AI agent? | [`AGENTS.md`](AGENTS.md) |
| What did the density-classification work find today? | [`docs/STOCHASTIC_CLASSIFIER_NOTE.md`](docs/STOCHASTIC_CLASSIFIER_NOTE.md) |
| What's the formal statement / theorem? | [`docs/technical_report.tex`](docs/technical_report.tex) §5 |
| What experiments have already been tried? | [`docs/DENSITY_CLASSIFICATION_IDEAS.md`](docs/DENSITY_CLASSIFICATION_IDEAS.md) |
| What did the structural mechanism probes show? | [`docs/MECHANISM_FINDINGS.md`](docs/MECHANISM_FINDINGS.md) |
| What's the reversibility plan? | [`docs/REVERSIBILITY_PLAN.md`](docs/REVERSIBILITY_PLAN.md) |
| How to add a new identity / property? | [`identities/README.md`](identities/README.md) |
| How does the search package work? | [`ca_search/README.md`](ca_search/README.md) |
