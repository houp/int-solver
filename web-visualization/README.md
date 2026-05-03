# Web Visualization

This directory contains a fully client-side desktop web application for exploring the materialized
materialized 2D binary Moore-neighborhood cellular automata rules that are:

- number-conserving, and
- satisfy at least one additional property from the identity catalog.

The browser app lets you:

- filter rules by any combination of the available CA properties,
- inspect each rule's exact 512-entry LUT encoded as hex,
- choose a rule and simulate it on a periodic grid,
- edit the initial condition by hand,
- randomize the grid with configurable density,
- run, pause, or single-step the automaton.

## Layout

- [IMPLEMENTATION_PLAN.md](/Users/witoldbolt/Documents/solvw/omp/web-visualization/IMPLEMENTATION_PLAN.md): implementation plan for the web app.
- [index.html](/Users/witoldbolt/Documents/solvw/omp/web-visualization/index.html): static SPA entry point.
- [styles.css](/Users/witoldbolt/Documents/solvw/omp/web-visualization/styles.css): application styling.
- [src](/Users/witoldbolt/Documents/solvw/omp/web-visualization/src): browser modules for data loading, filtering, LUT decoding, simulation, rendering, and app orchestration.
- [data/rules-index.json](/Users/witoldbolt/Documents/solvw/omp/web-visualization/data/rules-index.json): lightweight browser index for filtering and search.
- [data/rule-details](/Users/witoldbolt/Documents/solvw/omp/web-visualization/data/rule-details): on-demand LUT detail shards loaded only for selected rules.
- [scripts/build_dataset.py](/Users/witoldbolt/Documents/solvw/omp/web-visualization/scripts/build_dataset.py): dataset builder from solver artifacts.
- [tests](/Users/witoldbolt/Documents/solvw/omp/web-visualization/tests): Node-based tests for the pure logic modules and dataset structure.

## Dataset

The app does not parse the solver's binary format in the browser. Instead it precomputes a split
JSON catalog from:

- [expanded_property_panel_nonzero.bin](/Users/witoldbolt/Documents/solvw/omp/expanded_property_panel_nonzero.bin)
- [expanded_property_panel_nonzero.json](/Users/witoldbolt/Documents/solvw/omp/expanded_property_panel_nonzero.json)

The lightweight index stores:

- `legacyIndex`: explicit copy of the legacy positional index,
- `stableIndex`: deterministic lexicographic rank by LUT within the current dataset,
- `stableId`: permanent SHA-256 identifier derived from the full LUT,
- `mask`: little-endian property-membership bitmask,
- `ones`: population of ones in the LUT.

The heavy `lutHex` payload is stored separately in `data/rule-details/shard-XXXX.json` and fetched
only when a rule is selected or jumped to by `sid`.

The solver output stores only `x_1..x_510`; the builder reconstructs the full truth table by
restoring the double-legal boundary entries:

- `x_0 = 0`
- `x_511 = 1`

## Neighborhood Convention

The simulator uses the same Moore-neighborhood variable layout as the equation generator:

```text
x y z
t u w
a b c
```

The LUT index is:

```text
(x, y, z, t, u, w, a, b, c) -> x*256 + y*128 + z*64 + t*32 + u*16 + w*8 + a*4 + b*2 + c
```

This matches the solver and generator conventions, so the visualized dynamics correspond to the
stored rule tables exactly.

## Running Locally

Build the dataset:

```bash
cd /Users/witoldbolt/Documents/solvw/omp/web-visualization
python3 scripts/build_dataset.py
```

Run the tests:

```bash
cd /Users/witoldbolt/Documents/solvw/omp/web-visualization
node --test
```

Serve the static app:

```bash
cd /Users/witoldbolt/Documents/solvw/omp/web-visualization
python3 -m http.server 4173
```

Then open [http://localhost:4173](http://localhost:4173).

## Scope

The application lists the materialized nonzero-mask rules, not all
`147,309,433` number-conserving rules. The larger base class is summarized in the UI, but only the
stored nonzero-mask subset is interactively browsable.

## Extensibility

The current structure is designed to absorb future work without a rewrite:

- add more solver-side properties by regenerating `rules-index.json` and `rule-details/`,
- replace or augment the renderer without changing filtering or simulation logic,
- add richer analysis panels around the same dataset schema,
- switch to a bundler later if the app grows beyond static ES modules.
