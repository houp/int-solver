# Deep Analysis Of Shortlisted Candidates

All bare numeric rule labels in this note are legacy snapshot indices from
`expanded_property_panel_nonzero.bin`. The strongest recurring candidates discussed here map to:
- `24795` -> `sid:029b09cea0b5`
- `55897` -> `sid:812b7dae7aa7`
- `2370` -> `sid:f076fc6ffb58`
- `85932` -> `sid:adb15d42281e`
- `98759` -> `sid:ca6705f68694`

This note summarizes a deeper automated study of the shortlisted candidates from
[expanded_property_panel_exploration.json](/Users/witoldbolt/Documents/solvw/omp/expanded_property_panel_exploration.json).

Detailed machine-readable output:

- [candidate_deep_analysis.json](/Users/witoldbolt/Documents/solvw/omp/candidate_deep_analysis.json)

## Candidate Set

The analyzed rules were:

- `28354`
- `85932`
- `93021`
- `99452`
- `12435`
- `48212`
- `7685`
- `98759`
- `24795`
- `2370`
- `55897`
- `34521`

These were chosen as a mixture of:

- top-ranked rules from the first-pass dynamic screen,
- and rules that showed noticeably larger damage spreading than the rest.

## Test Battery

For each rule, the following scenarios were simulated:

- random density `0.12`, `128 x 128`, `256` steps
- random density `0.30`, `128 x 128`, `256` steps
- random density `0.50`, `128 x 128`, `256` steps
- checkerboard, `128 x 128`, `192` steps
- vertical half-plane interface, `128 x 128`, `192` steps
- horizontal stripes, `128 x 128`, `192` steps

For each scenario, the following measures were collected:

- late-time activity
- late-time entropy
- late-time autocorrelation with the initial state
- late-time one-bit damage spreading
- late-time `2 x 2` patch damage spreading
- exact tail period detection
- dominant best-fit lattice shift between consecutive late-time states

Small localized motifs were also tested:

- single particle
- horizontal particle pair
- diagonal particle pair
- `2 x 2` live block

## Main Conclusion

The shortlisted rules are still overwhelmingly **transport-dominated**, not Life-like.

Across the whole candidate set, the most consistent observations are:

- no candidate showed a nontrivial exact tail period on random initial conditions;
- almost every candidate had a single dominant lattice shift with fraction `1.0` in late time;
- the best-fit shift overlap was usually extremely close to `1.0`;
- the localized motif tests mostly showed rigid drift plus mild support broadening, not
  self-maintaining oscillators or expanding complicated debris fields.

So the current best candidates are better understood as **anisotropic particle-transport and
collision rules** than as birth/death or glider-supporting rules in the Conway-Life sense.

## Structured-Seed Behavior

The structured seeds reveal how simple many of these rules really are.

### Period-2 Regime

For most of the diagonal-monotone rules:

- checkerboard becomes exact period `2`
- vertical interface becomes exact period `2`
- horizontal stripes become exact period `2`

This is a strong sign of simple advection or alternating transport layers, not deep recurrence.

### Static Interface Regime

Two orthogonal-monotone rules stood out:

- `93021`
- `34521`

For these rules:

- the vertical half-plane interface is static, exact period `1`
- or the stripe configuration is static, exact period `1`

This makes them look even more like directional one-dimensional transport systems embedded in 2D.

## Damage-Spreading Results

Most candidates had very small damage spreading on random density `0.12` and `0.30`.

The notable exceptions were:

- `24795`
- `55897`
- `2370`

These are the best current candidates for “nontrivial” behavior.

### Rule `24795`

Properties:

- `diagonal_monotone`
- isolated particle: `southwest`
- isolated hole: `north`

Most significant metrics:

- random `0.30`: late one-bit damage `0.0823`
- random `0.30`: late patch damage `0.1085`
- random `0.50`: late one-bit damage `0.0450`
- random `0.50`: late patch damage `0.0780`

Interpretation:

- among the analyzed rules, this is one of the clearest examples where collisions amplify local
  perturbations rather than merely transporting them;
- nevertheless, its late-time behavior is still dominated by a single global shift direction.

### Rule `55897`

Properties:

- `diagonal_monotone`
- isolated particle: `northwest`
- isolated hole: `east`

Most significant metrics:

- random `0.30`: late one-bit damage `0.1588`
- random `0.30`: late patch damage `0.1705`

Interpretation:

- this is the strongest perturbation amplifier in the analyzed set;
- however, the dominant late-time best-fit shift is still unique and stable, so this is best read
  as a transport rule with collision-sensitive branching, not as a fully disordered rule.

### Rule `2370`

Properties:

- `diagonal_monotone`
- isolated particle: `east`
- isolated hole: `west`

Most significant metrics:

- random `0.50`: late one-bit damage `0.0366`
- random `0.50`: late patch damage `0.0369`
- horizontal stripes: late one-bit damage `0.4920`
- horizontal stripes: late patch damage `0.4945`

Interpretation:

- this rule reacts strongly to stripe-like defects;
- it appears to be especially sensitive to transport-layer perturbations rather than to generic
  sparse disorder.

## Localized Motif Results

The localized motif tests reinforce the same picture.

For all candidates:

- a single particle remains a single particle;
- the isolated-particle motion agrees with the previously detected one-step velocity class;
- a `2 x 2` block never exploded into a large cloud over the tested horizon.

The main nontrivial signal comes from the final bounding-box area:

- many rules kept horizontal or diagonal pairs compact;
- a few broadened them moderately;
- the largest observed final support areas were still small, e.g. around `9` to `15`.

This is again more consistent with directional scattering than with rich object ecology.

## Final Interpretation

Do these shortlisted rules look interesting?

- Yes, but mainly as **particle-transport and scattering systems**.
- No, not yet as clear candidates for Life-like localized complexity.

The strongest candidates for deeper manual visual inspection are:

1. `55897`
2. `24795`
3. `2370`

Why these three:

- they show the strongest perturbation growth;
- they are not trivial shifts;
- they still preserve enough transport structure to be interpretable.

If the goal is to find genuinely richer rules, the next screening round should emphasize:

- collision amplification more heavily than raw activity,
- sensitivity on structured seeds,
- and support broadening of small motifs,

rather than entropy/activity alone.
