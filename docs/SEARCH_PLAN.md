# Search Plan For Interesting Number-Conserving Moore Rules

## Goal

Reduce the full set of `147,309,433` number-conserving binary Moore-neighborhood rules to a much
smaller set of candidates with plausible nontrivial dynamics, then support large-scale simulation
and metric extraction on Apple Silicon.

This work splits into two coupled tracks:

1. exact filtering of analytically simple rules, first by equations and then by cheap LUT-level
   post-classification;
2. high-throughput simulation and metric collection for the surviving candidate rules.

## Track A: Narrow The Search Space

### A1. Equation-Level Filters

Add exact property families that can be compiled by the existing functional-identity generator and
tested inside `solve_omp_opt.cpp` as additional properties. The first batch should focus on
isolated-particle and isolated-hole microdynamics.

Immediate filters to add:

- isolated-particle velocity classes:
  - `isolated_particle_static`
  - `isolated_particle_north`
  - `isolated_particle_south`
  - `isolated_particle_east`
  - `isolated_particle_west`
  - `isolated_particle_northeast`
  - `isolated_particle_northwest`
  - `isolated_particle_southeast`
  - `isolated_particle_southwest`
- isolated-hole velocity classes:
  - `isolated_hole_static`
  - `isolated_hole_north`
  - `isolated_hole_south`
  - `isolated_hole_east`
  - `isolated_hole_west`
  - `isolated_hole_northeast`
  - `isolated_hole_northwest`
  - `isolated_hole_southeast`
  - `isolated_hole_southwest`

For an isolated particle, the filter should fully specify the one-step image of the 3x3 support:
the target site is `1` and the other eight affected sites are `0`. The hole filters are the exact
complementary analogue on the all-ones background.

These filters are useful because they immediately split the base class into transport regimes:

- stationary particle / stationary hole,
- matching particle and hole drift,
- asymmetric drift,
- diagonal vs orthogonal motion.

### A2. Post-Equation Filters

Add a cheap LUT-level screening layer for analytically simple rules. This will be applied after the
solver or after loading a rule catalog.

Immediate post-filters:

- exact identity detection;
- exact rigid shift detection in all eight directions;
- isolated-particle velocity classification;
- isolated-hole velocity classification;
- detection of already-understood embedded one-dimensional families using property masks plus LUT
  checks:
  - von Neumann embedded traffic family,
  - diagonal von Neumann embedded traffic family.

The output of this stage should not only be a boolean “simple / not simple” tag, but a structured
explanation such as:

- `identity`
- `rigid_shift:north`
- `embedded_von_neumann_traffic`
- `embedded_diagonal_traffic`
- `unclassified`

## Track B: High-Performance Simulation

### B1. Backend Strategy

Implement a batched simulator with a backend abstraction:

- `numpy` backend for correctness and local fallback;
- `mlx` backend for Apple Silicon GPU execution.

The public API should be backend-neutral so that later backends can be added without changing the
search logic.

### B2. Simulation Scope

The first implementation should support:

- pairwise batched simulation of many rules at once;
- periodic boundary conditions;
- binary Moore-neighborhood rules encoded by the 512-entry LUT;
- repeated stepping over a fixed grid size without recompilation when shapes are stable;
- rule input as LUT hex strings or decoded LUT bit tables.

### B3. Metrics Scaffold

The first metric layer should be deliberately cheap:

- population density over time;
- temporal activity / Hamming distance between consecutive states;
- per-step Shannon entropy of the cell field;
- pairwise damage spreading for nearby initial conditions.

The simulator API should make it straightforward to add more expensive metrics later, such as:

- spatial correlation lengths,
- interface velocity,
- approximate fractal dimension,
- connected-component statistics,
- recurrence and compression-based scores.

### B4. Performance Verification

Benchmark the simulator on Apple Silicon with:

- a fixed rule batch,
- multiple grid sizes,
- multiple step counts,
- both `numpy` and `mlx` backends when available.

The benchmark output should report cells updated per second and effective rule-steps per second.

## Immediate Milestones

### Milestone 1

- add the search plan and implementation scaffolding;
- add the initial exact microdynamics filter specs;
- add LUT-level simple-family classification utilities.

### Milestone 2

- implement the batched simulator API;
- implement the `numpy` backend;
- implement the `mlx` backend behind optional dependencies;
- add a benchmark CLI.

### Milestone 3

- connect the catalog loader, filters, and simulator into a screening workflow;
- run small pilot experiments on the currently materialized nonzero-mask catalog.

## Deliverables In This Iteration

- project plan document;
- initial search package with:
  - catalog loading,
  - LUT helpers,
  - simple-rule post-filters,
  - batched simulator and metrics scaffold;
- generated exact filter specs for isolated particle / hole motion;
- documentation updates describing how these tools fit into the main solver workflow.
