# Object Atlas Study of Rules 55897, 24795, and 2370

All bare numeric rule labels in this note are legacy snapshot indices from
`expanded_property_panel_nonzero.bin`. The permanent stable identifiers are:
- `55897` -> `sid:812b7dae7aa7`
- `24795` -> `sid:029b09cea0b5`
- `2370` -> `sid:f076fc6ffb58`

This note summarizes a deeper object-level study of the three shortlisted rules:

- `55897`
- `24795`
- `2370`

The goal was to test whether these rules support any genuinely compact moving or oscillating
objects, or whether their dynamics are better described as a sparse transport-and-scattering gas.

Raw outputs:

- [object_atlas_study.json](/Users/witoldbolt/Documents/solvw/omp/object_atlas_study.json)
- [object_atlas_24795_size5.json](/Users/witoldbolt/Documents/solvw/omp/object_atlas_24795_size5.json)

## Study Design

Two complementary probes were used.

### 1. Exhaustive small-motif atlas

For each rule, I enumerated all connected finite patterns under 8-neighbor connectivity with:

- size `1`
- size `2`
- size `3`
- size `4`

and embedded them in:

- zero background for particle motifs
- one background for hole motifs

The total motif count per rule at `max-size=4` was:

- `135` connected shapes
- `270` background-aware motif runs per rule

Each motif was simulated for `128` steps and compared against independent free transport implied
by the isolated particle or isolated hole velocity.

### 2. Three-carrier interaction atlas

For each rule, I also tested a small catalog of 3-carrier initial conditions:

- `36` particle triples in zero background
- `36` hole triples in one background

again comparing against the free independent-transport reference.

### Extra check for the strongest candidate

Because `24795` remained the strongest rule in earlier studies, I extended its motif atlas to
connected shapes up to size `5`:

- total connected shapes: `613`
- total background-aware motif runs: `1226`

This was intended to check whether compact object dynamics begin to appear only at slightly
larger support.

## Classification Language

Each motif or triple was classified heuristically into one of the following outcomes:

- `free_transporter`
  The pattern follows free ballistic transport with no detected interaction.
- `separated_carriers`
  The pattern diverges from the initial compact shape and resolves into multiple separated
  carriers.
- `localized_scatter`
  The pattern interacts nontrivially but remains spatially localized without forming a compact
  periodic object.
- `phase_shift_scatter`
  The pattern stays very close to free transport but with a mild shape or phase correction.
- `compact_bound_state` or `compact_oscillator`
  A compact periodic or moving-frame periodic object.

The crucial result is that the last category did not occur.

## Main Result

Across all connected motifs up to size `4` for all three rules, and up to size `5` for Rule
`24795`, I found:

- `0` compact bound states
- `0` compact oscillators
- `0` compact triple-collision bound states

This is the strongest evidence so far that these rules do **not** hide a small glider/oscillator
ecology of the sort one would expect in Life-like complex rules.

## Rule-by-Rule Summary

### Rule 55897

Motif class counts up to size `4`:

- `free_transporter`: `74`
- `separated_carriers`: `33`
- `localized_scatter`: `106`
- `phase_shift_scatter`: `57`

Triple class counts:

- `free_transporter`: `50`
- `separated_carriers`: `15`
- `localized_scatter`: `7`

Interpretation:

- Many motifs interact, but the interactions stay mild.
- No compact objects survive.
- The strongest behavior is particle-side scattering; hole-side dynamics are closer to free
  transport.

This reinforces the earlier picture of `55897` as a directional scattering rule, but not a rich
object-based one.

### Rule 24795

Motif class counts up to size `4`:

- `free_transporter`: `66`
- `separated_carriers`: `33`
- `localized_scatter`: `123`
- `phase_shift_scatter`: `48`

Triple class counts:

- `free_transporter`: `41`
- `separated_carriers`: `20`
- `localized_scatter`: `10`
- `phase_shift_scatter`: `1`

Interpretation:

- `24795` remains the strongest scattering rule.
- It shows the largest amount of genuine localized interaction among the three.
- Even so, these interactions still do not stabilize into compact periodic objects.

### Rule 2370

Motif class counts up to size `4`:

- `free_transporter`: `33`
- `separated_carriers`: `66`
- `localized_scatter`: `155`
- `phase_shift_scatter`: `16`

Triple class counts:

- `free_transporter`: `45`
- `separated_carriers`: `22`
- `localized_scatter`: `4`
- `phase_shift_scatter`: `1`

Interpretation:

- `2370` continues to look like the most regular and conveyor-like rule.
- Many motifs do interact, but the typical outcome is resolution into simple carrier sets rather
  than any reusable compound object.

## Size-5 Extension for Rule 24795

The larger atlas for `24795` gives:

- `free_transporter`: `126`
- `separated_carriers`: `107`
- `localized_scatter`: `498`
- `phase_shift_scatter`: `495`
- `compact_bound_state`: `0`
- `compact_oscillator`: `0`

This matters a lot.  It means that the absence of compact objects at size `<=4` is not just a
small-cutoff artifact for the strongest candidate.  Even after expanding to `1226` background-aware
motif runs up to connected size `5`, no compact periodic object family emerged.

## What We Understand Now

Taken together with the earlier random-state, interface, and pair-collision studies, the current
picture is:

1. These rules are conservative carrier systems.
2. Single particles and holes behave as exact ballistic objects.
3. Clean interfaces do not roughen.
4. Small motifs often interact, but the interactions are mild.
5. The usual outcomes are:
   - free transport,
   - splitting into separated carriers,
   - localized scatter with small phase corrections.
6. There is currently no evidence of a compact object ecology.

So the best current description is:

`anisotropic conservative scattering gases`

rather than:

`Life-like complex rules with gliders and oscillators`

## Best Current Ranking

The ranking remains:

1. `24795`
2. `55897`
3. `2370`

but the gap is now better understood:

- `24795` is the strongest because it has the richest localized scattering layer.
- `55897` is similar but somewhat more one-sided and more advection-dominated.
- `2370` is structurally the simplest.

## Next Possible Step

The next meaningful extension would no longer be a wider motif atlas.  The most useful remaining
probe is:

- `longer sparse collision programs`

meaning carefully designed repeated collision setups where multiple carriers are launched in
sequence to test whether scatterers can be reused and whether any delayed compound structure can
be assembled.

At this point, however, the burden of proof has shifted.  New experiments would need to find a
specific constructed object, because the generic small-object search is now strongly negative.
