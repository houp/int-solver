# Reversibility Search Plan For Number-Conserving Binary Moore Rules

## Goal

Find, or strongly narrow down, the reversible subclass inside the set of
`147,309,433` number-conserving binary Moore-neighborhood cellular automata.

The key constraint is theoretical: reversibility of general two-dimensional
cellular automata is undecidable, so we should not expect a complete generic
decision procedure comparable to the current linear-equation pipeline for
number conservation. Instead, we build a staged rejection-and-investigation
workflow:

1. exact fast rejection tests that are strong necessary conditions for
   reversibility;
2. targeted counterexample search on the small set of survivors;
3. rule-by-rule mathematical analysis if only a tiny set remains.

## Principles

- Prefer exact tests over heuristics in the first stages.
- Exploit number conservation aggressively: reversible NCCAs must act as
  permutations on each fixed-population sector of every finite torus.
- Keep the implementation human-reviewable:
  - small self-contained modules,
  - explicit data formats,
  - regression tests on identity, shifts, and known nonreversible traffic
    rules.
- Use stable `sid:` identifiers in all outputs and reports.

## Stage 0: Immediate Exclusions

The following rules are already understood and should be treated separately:

- identity;
- the eight rigid shifts.

These are trivially reversible and should not dominate the reversible search.

## Stage 1: Exact Sparse-Sector Injectivity Tests

### Idea

For a binary NCCA on an `H x W` torus, reversibility implies bijectivity on
every fixed-population sector.

Therefore, for small `k`, we can test:

- injectivity on the `k`-particle sector;
- injectivity on the `k`-hole sector.

If a collision exists in any such sector, the rule is not reversible.

### Why It Should Work Well

- For small `k`, the state count is combinatorial rather than exponential.
- Many transport rules are expected to collide already for `k = 2` or `3`.
- The test is exact and returns an explicit counterexample pair.

### Initial Parameters

- torus sizes:
  - `6x6`
  - `8x8`
- particle counts:
  - `k = 1, 2, 3, 4`
- hole counts:
  - `k = 1, 2, 3, 4`

The `k = 1` test is mostly a sanity check: any NCCA that fails it is already
pathological. The first useful rejection power is expected at `k = 2` and
`k = 3`.

## Stage 2: Exact Small-Torus Bijectivity

### Idea

Test whether a rule is bijective on the full finite torus state space.

Initial sizes:

- `3x3`: `512` states
- `4x4`: `65536` states

These tests are exact. Failure on any finite torus immediately disproves
reversibility.

### Initial Strategy

- batch the state enumeration;
- evolve each batch with the vectorized simulator kernel;
- encode each output state canonically;
- detect duplicates with compact bitsets / boolean arrays.

This stage should eliminate many candidates that survive the sparse-sector
checks.

## Stage 3: Radius-1 Inverse Existence Test

### Idea

Ask a stronger local question:

> Does the rule admit an inverse that is itself Moore radius `1`?

This can be tested exactly by checking consistency of the induced mapping

`5x5 preimage patch -> central 3x3 image patch`.

If two distinct `5x5` preimages produce the same central `3x3` image patch but
different center states, then a radius-1 inverse does not exist.

### Role In The Pipeline

- not equivalent to full reversibility;
- but a successful result is a strong structural certificate;
- a failure gives a clean local obstruction.

## Stage 4: SAT / Constraint-Based Collision Search

For the small set of survivors, move beyond brute-force finite enumeration.

Target problem:

- find distinct torus configurations `x != y` such that `F(x) = F(y)`.

Use this for:

- `5x5` and `6x6` tori;
- sparse-support collision witnesses;
- hand-picked rules surviving earlier stages.

This stage is likely only needed if the reversible candidate set does not
collapse quickly.

## Stage 5: Rule-By-Rule Investigation

If only a tiny set remains after Stages 1-3, switch from generic screening to
individual investigation:

- search for explicit collisions;
- inspect particle/hole motion;
- inspect finite-order behavior on tori;
- attempt to construct inverse local rules;
- compare against reversible partitioned / lattice-gas mechanisms from the
  literature.

## Implementation Milestones

### Milestone 1

- add this plan;
- implement:
  - trivial reversible detection,
  - sparse-sector injectivity test,
  - exact `3x3` / `4x4` torus bijectivity test;
- add regression tests.

### Milestone 2

- expose the checks through `ca_search.cli`;
- support rule selection by `sid:` and `rank:`;
- produce machine-readable reports.

### Milestone 3

- run the first screen on the nontrivial number-conserving catalog;
- identify the survivor set;
- decide whether to proceed with SAT-style search or with rule-by-rule study.

## Expected Outcome

Given nearby literature on reversible number-conserving cellular automata, it
is plausible that the reversible subclass in the binary Moore case is tiny and
possibly trivial. The first exact filters may therefore reduce the search to a
very small number of survivors, after which direct mathematical analysis
becomes realistic.
