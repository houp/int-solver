# Focused Study of Candidate Rules 55897, 24795, and 2370

All bare numeric rule labels in this note are legacy snapshot indices from
`expanded_property_panel_nonzero.bin`. The permanent stable identifiers are:
- `55897` -> `sid:812b7dae7aa7`
- `24795` -> `sid:029b09cea0b5`
- `2370` -> `sid:f076fc6ffb58`

This note summarizes a deeper automated study of the three candidate number-conserving rules that stood out in the earlier broad screen of the `133713` nonzero-mask catalog:

- `55897`
- `24795`
- `2370`

Raw machine-readable output is in [focused_rule_study.json](/Users/witoldbolt/Documents/solvw/omp/focused_rule_study.json).

## Study Setup

For each rule, the analysis used:

- random initial states on `256 x 256` grids for `512` steps at densities `0.05`, `0.12`, `0.30`, and `0.50`
- four random seeds per density
- one-bit and `2 x 2` patch perturbation probes for damage spreading
- structured seeds:
  - checkerboard
  - vertical interface
  - horizontal stripes
  - diagonal stripes
- motif trajectories on `65 x 65` grids for:
  - single particle in zero background
  - single hole in one background
  - `2 x 2` block
  - 3-cell L-shape

The full run used the MLX backend on GPU.

## High-Level Conclusion

All three rules are still best understood as anisotropic transport rules rather than Life-like birth/death systems.

What they do have:

- persistent ballistic transport
- strong directional bias
- nontrivial medium-density collision response
- structured periodic behavior on stripe/interface seeds

What they do not currently show:

- clear self-maintaining localized object ecologies
- rich long-lived emitters or glider guns
- strong evidence of open-ended defect amplification

The most promising rule for further manual study is `24795`, followed by `55897`. Rule `2370` appears substantially simpler.

## Comparative Summary

| Rule | Property tags | Isolated particle | Isolated hole | Strongest late damage (patch) | Most notable structured behavior |
| --- | --- | --- | --- | ---: | --- |
| `55897` | `diagonal_monotone` | `northwest` | `east` | `0.0501` at random density `0.30` | all tested structured seeds become exact period-2 traveling textures |
| `24795` | `diagonal_monotone` | `southwest` | `north` | `0.1075` at random density `0.30` | diagonal stripes enter a genuine period-6 regime |
| `2370` | `diagonal_monotone` | `east` | `west` | `0.0389` at random density `0.50` | checkerboard and diagonal stripes are exact fixed points |

Average late patch damage across the four random-density ensembles:

- `24795`: `0.0350`
- `55897`: `0.0211`
- `2370`: `0.0101`

This ranking matches the qualitative impression: `24795` is the most collision-sensitive of the three, while `2370` is the most regular.

## Rule 24795

This is the strongest candidate of the three.

Key observations:

- isolated particles drift `southwest`
- isolated holes drift `north`
- low-density random states behave almost like clean ballistic transport
- at density `0.30`, perturbations amplify strongly:
  - late mean one-bit damage `0.1069`
  - late mean patch damage `0.1075`
  - max patch damage `0.2173`
- threshold crossing is relatively early:
  - patch damage exceeds `0.01` around step `74`
  - patch damage exceeds `0.05` around step `173`

Structured seeds:

- checkerboard, vertical interface, and horizontal stripes all settle into exact period-2 behavior
- diagonal stripes are more interesting:
  - late activity `0.834`
  - exact tail period `6`
  - dominant stepwise drift `north`

Motifs:

- a single particle remains a rigid translating particle
- a `2 x 2` block and a 3-cell L-shape both reorganize into sparse traveling patterns
- the L-shape settles faster and more tightly than in `55897`

Interpretation:

`24795` looks like a transport rule with the strongest genuinely nonlinear collision layer among the three studied cases. It still does not look Life-like, but it does look like a plausible source of nontrivial scattering phenomena.

## Rule 55897

This rule is also nontrivial, but less collision-sensitive than `24795`.

Key observations:

- isolated particles drift `northwest`
- isolated holes drift `east`
- at low densities, it behaves almost like a rigid `northwest` conveyor
- at densities `0.30` and `0.50`, the dominant bulk drift changes to `west`
- medium-density perturbation growth is noticeable but clearly weaker than for `24795`:
  - late mean patch damage `0.0501` at density `0.30`
  - max patch damage `0.0945`

Structured seeds:

- checkerboard, vertical interface, horizontal stripes, and diagonal stripes all enter exact period-2 regimes
- all of these remain highly active and highly entropic, but the behavior is still strongly locked to simple periodic transport

Motifs:

- single particles and single holes are rigid ballistic objects
- the `2 x 2` block and L-shape both settle into sparse translating patterns after about `35` steps

Interpretation:

`55897` is a clear transport-and-scattering rule, but its long-time behavior appears more locked into simple advection than `24795`. It remains worth studying, especially for medium-density collision dynamics, but it is less compelling as a candidate for richer complexity.

## Rule 2370

This rule appears to be the simplest of the three.

Key observations:

- isolated particles drift `east`
- isolated holes drift `west`
- low- and medium-density random states are almost pure eastward ballistic transport
- damage stays very small until the density reaches `0.50`
- the only clearly noticeable perturbation growth occurs in the dense regime:
  - late mean one-bit damage `0.0579`
  - late mean patch damage `0.0389`

Structured seeds:

- checkerboard is an exact fixed point
- diagonal stripes are also an exact fixed point
- vertical interfaces and horizontal stripes become period-2 traveling textures

Motifs:

- the single particle is a clean east-moving carrier
- the single hole is a clean west-moving carrier
- the `2 x 2` block and L-shape both settle quickly into sparse translating patterns

Interpretation:

`2370` looks closest to a structured conveyor or traffic-like rule. It is not trivial, but it currently shows the least evidence of rich collision-mediated complexity among the three candidates.

## Final Assessment

If the goal is to continue searching for potentially interesting dynamics inside the number-conserving class, the priority order should be:

1. `24795`
2. `55897`
3. `2370`

The next useful analyses should be more targeted than broad random-state metrics:

- explicit two-particle collision catalogs
- small-cluster collision catalogs
- interface roughening and interface-velocity measurements
- longer visual runs focused on medium densities around `0.30`

That is the regime where the three rules differ the most, and where `24795` in particular begins to depart from simple rigid transport.
