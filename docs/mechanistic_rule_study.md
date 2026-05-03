# Mechanistic Study of Rules 55897, 24795, and 2370

All bare numeric rule labels in this note are legacy snapshot indices from
`expanded_property_panel_nonzero.bin`. The permanent stable identifiers are:
- `55897` -> `sid:812b7dae7aa7`
- `24795` -> `sid:029b09cea0b5`
- `2370` -> `sid:f076fc6ffb58`

This note extends the earlier coarse screening and focused ensemble study with more explicit
mechanistic probes for the three shortlisted rules:

- `55897`
- `24795`
- `2370`

Raw output is in [mechanistic_rule_study.json](/Users/witoldbolt/Documents/solvw/omp/mechanistic_rule_study.json).

## What Was Measured

For each rule, the study added three targeted probes:

1. `Two-carrier collision scans`
   - pairs of isolated particles in zero background
   - pairs of isolated holes in one background
   - 12 representative relative offsets
   - comparison against free ballistic transport implied by the isolated-carrier velocity

2. `Interface probes`
   - vertical interface
   - horizontal stripes
   - diagonal stripes
   - measurement of dominant drift, exact tail period, activity, and interface width

3. `Longer carrier runs`
   - single particle
   - single hole
   - longer horizon to confirm rigid transport and absence of spontaneous growth

## Main Conclusion

The new evidence strengthens the earlier interpretation:

- these rules are still `transport-and-scattering` systems, not Life-like birth/death systems
- `clean interfaces remain sharp`
- `single carriers stay exact ballistic objects`
- the genuinely nontrivial behavior is confined to `a very small set of pair encounters`

So the interesting part of the search space is becoming clearer: if there is richer complexity
inside these families, it will likely come from repeated sparse collisions of mobile carriers,
not from interface turbulence or spontaneous object generation.

## Shared Features

All three rules have the following common properties.

- A single isolated particle remains a rigid translating carrier.
- A single isolated hole remains a rigid translating carrier.
- Vertical and horizontal interface seeds evolve into exact period-2 translating textures.
- Measured interface width remains `0` throughout the long runs.
  - There is no interface roughening in these probes.
- No rule created extra particles or holes from these localized tests.

This is strong evidence against a Life-like interpretation.  The systems are conservative,
ballistic, and highly structured.

## Rule 55897

### Carrier transport

- isolated particle velocity: `northwest`
- isolated hole velocity: `east`

### Interfaces

- vertical interface: drift `east`, exact period `2`
- horizontal stripes: drift `north`, exact period `2`
- diagonal stripes: drift `north`, exact period `2`
- interface width stays exactly `0`

So all tested large-scale structured states remain perfectly sharp and simply translate or
alternate.

### Pair collisions

Particle-particle interactions are rare but real.

- `2 / 12` tested particle offsets deviate from free transport
- colliding offsets:
  - `(1, 1)`
  - `(1, 2)`
- `0 / 12` tested hole offsets deviate from free transport

The interaction is mild, not explosive:

- the final state differs from free transport by only about `2` cells
- cluster count remains `2`
- no new localized structures are created

Interpretation:

`55897` is a particle-biased scattering rule.  Particles can shear each other in a few
near-diagonal encounters, while holes behave as essentially free carriers in the tested cases.

## Rule 24795

### Carrier transport

- isolated particle velocity: `southwest`
- isolated hole velocity: `north`

### Interfaces

- vertical interface: drift `east`, exact period `2`
- horizontal stripes: drift `north`, exact period `2`
- diagonal stripes: drift `north`, exact period `6`
- interface width stays exactly `0`

This is still not roughening behavior, but the period-6 diagonal stripe regime remains the most
interesting large-scale structured behavior seen among the three rules.

### Pair collisions

This rule shows the most balanced nontriviality.

- `1 / 12` tested particle offsets deviate from free transport
- `1 / 12` tested hole offsets deviate from free transport
- particle collision offset:
  - `(-2, 1)`
- hole collision offset:
  - `(2, 0)`

Again the interaction is localized and mild:

- final deviation from free transport is only about `2` cells
- cluster count remains `2`
- no new long-lived structures appear immediately

Interpretation:

`24795` remains the strongest candidate because it has:

- the strongest damage-spreading scores from the earlier study,
- the unique period-6 diagonal-stripe regime,
- and nontrivial pair interactions for both particles and holes.

It still does not show rich object ecology, but it is the best current candidate for reusable
carrier-scattering dynamics.

## Rule 2370

### Carrier transport

- isolated particle velocity: `east`
- isolated hole velocity: `west`

### Interfaces

- vertical interface: drift `east`, exact period `2`
- horizontal stripes: drift `north`, exact period `2`
- diagonal stripes: `static`, exact period `1`
- interface width stays exactly `0`

This is the cleanest and most regular of the three rules.

### Pair collisions

- `1 / 12` tested particle offsets deviate from free transport
- `1 / 12` tested hole offsets deviate from free transport
- both occur for the same offset:
  - `(1, 1)`

The interaction is again extremely mild:

- final deviation from free transport is about `2` cells
- cluster count remains `2`
- no persistent compound structure is generated

Interpretation:

`2370` is best viewed as a disciplined east/west conveyor rule with one narrow diagonal encounter
mode.  Compared with `24795` and `55897`, it looks much closer to a nearly integrable transport
system.

## Comparative Ranking

The mechanistic evidence keeps the same ranking as before:

1. `24795`
2. `55897`
3. `2370`

Why:

- `24795` has both particle and hole scattering, plus the period-6 diagonal-stripe regime.
- `55897` has only particle-side scattering in the tested offsets.
- `2370` has the most regular macroscopic behavior and the narrowest collision set.

## What This Means For The Search

The next promising direction is no longer broad random-state scoring.  The important question is:

`Can repeated localized carrier collisions build long-lived compound objects?`

The right next tests are therefore:

- larger and denser pair-collision catalogs
- three-carrier collision experiments
- repeated scatter-on-scatter constructions
- searches for moving bound states and repeatable scatterers

At this stage, the evidence points away from Life-like growth and toward a more conservative
particle-interaction picture.
