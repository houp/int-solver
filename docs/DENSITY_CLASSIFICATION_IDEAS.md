# Density Classification Idea Ledger

This file tracks the current search for a 2D Fukś-style extension using the
number-conserving Moore-neighborhood rules in this repository.

Status markers:

- `[ ]` not tested yet
- `[~]` partially tested / still open
- `[x]` tested and currently not promising
- `[>]` current active lead

## Tested Or Partially Tested

Important conceptual note:

- A periodically repeated rule sequence such as `(F^n M^k)^m`, with fixed
  local binary rules `F` and `M`, is still just a single binary CA with a
  larger effective radius when viewed stroboscopically.
- Therefore such schedules do **not** escape the standard impossibility
  framework for perfect binary CA density classification.
- In contrast, a Fukś-style construction uses a finite globally synchronized
  phase switch, e.g. `F^(T/2)` followed by `M^(T/2)`. That is not a single
  time-homogeneous CA.
- From this point on:
  - periodically repeated schedules are useful as empirical heuristics and
    baseline dynamical mechanisms;
  - but they are **not** valid end targets for an exact “beyond-CA” DCP
    construction.

- `[x]` Single embedded cardinal traffic cycle, then Moore majority.
  Result: weak smoothing only; no consensus near the critical region.

- `[x]` Single embedded diagonal traffic cycle, then Moore majority.
  Result: somewhat stronger smoothing than the cardinal cycle, still no consensus.

- `[~]` Single nontrivial NCCA preprocessor, then Moore majority.
  Result so far: `sid:812b7dae7aa7` and later several balanced
  `orthogonal_monotone` rules improve near-critical majority accuracy, but
  consensus remains absent.

- `[x]` Repeated hybrid blocks such as `[traffic..., majority] × k`.
  Result: no recovery of the 1D traffic-majority phenomenon.

- `[>]` Repeated short blocks of the form `[NCCA, majority] × k`.
  Result so far: this is the first family that produces substantial
  consensus on larger tori.  The strongest current examples are
  `sid:029b09cea0b5 + VN-majority`,
  `sid:29154e9615d8 + Moore-majority`,
  and later `sid:812b7dae7aa7 + VN-majority`.  Broader repeated-block
  search also uncovered strong `diagonal_monotone` candidates such as
  `sid:ac809b9cab03` and `sid:2667f2c59840`.  The current strongest lead is
  now `sid:58ed6b657afb`, with a scaled repeated-block schedule
  `[sid:58ed6b657afb for 8 steps, VN-majority for 16 steps]^k`.
  This line of work is now reclassified as a **CA baseline track**, not the
  final beyond-CA target.

- `[x]` One-sided preprocessor ranking at a single density.
  Result: misleading; top hits are often badly biased and fail at the mirrored
  density.

- `[>]` Balanced paired-density preprocessor search.
  Result so far: the best leads are now
  `sid:29154e9615d8`, `sid:93f0176f55d8`,
  `sid:5dceb34ea73f`, `sid:f1a2737cb962`, `sid:0c3ad3e221f5`.

- `[x]` Longer single-rule schedules on larger tori.
  Result so far: better smoothing and better majority accuracy, still zero
  consensus in the tested near-critical runs.

- `[x]` Mixed two-preprocessor blocks such as
  `[NCCA_1, NCCA_2, majority] × k`.
  Result so far: weaker than the best single-preprocessor repeated blocks.

- `[~]` Block-geometry tuning:
  vary the number of NCCA steps and amplifier steps inside each repeated
  block.
  Result so far: this matters a lot.  On `64x64`, the best current schedule
  was once `sid:812b7dae7aa7` with `12` NCCA steps then `4` VN-majority
  steps, repeated `16` times, but the later family search produced a better
  scalable schedule:
  `sid:58ed6b657afb` with `8` NCCA steps then `16` VN-majority steps.

- `[x]` Pure threshold amplifiers beyond plain majority.
  Result so far: most such rules collapse immediately to trivial all-0 or
  all-1 consensus and are not useful as balanced amplifiers.

- `[x]` Simple correction tails after a repeated-block schedule.
  Result so far: a final Moore/VN tail slightly changes consensus accuracy,
  but does not fix the main correctness problem.

- `[x]` Restrict repeated-block search to `black_white_symmetric` rules.
  Result so far: the best black/white-symmetric candidates are weaker than
  the current general repeated-block leaders on larger tori.

## Next Ideas To Test

- `[>]` Fukś-style finite phase schedules with global synchronization:
  - `F^(T1(L)) -> G^(T2(L))`
  - possibly `F1^(T1(L)) -> F2^(T2(L)) -> G^(T3(L))`
  - but only a fixed finite number of global switches, not periodic
    indefinite repetition.
  First direct screen is now implemented in
  `python3 -m ca_search.cli density-finite-switch-screen`.
  Early `32x32` balanced-screen leaders under `F^(T1) -> VN-majority^(T2)`
  include:
  - `sid:b9b31aebc9b5` (`diagonal_monotone`),
  - `sid:6774f7e4355e` (`orthogonal_monotone`),
  - `sid:3fd35ca10c2e` (`diagonal_monotone`),
  - `sid:13d4fec69618` (`orthogonal_monotone`),
  - `sid:fdabfae81f3f` (`orthogonal_monotone`).
  At the small-screen stage, the best of these reached nonzero consensus
  accuracy (`1/16`) without any periodic repetition.
  However, first `64x64` direct validations are still negative:
  - `sid:b9b31aebc9b5`, `F^96 -> G^32`: no consensus, min final-majority
    accuracy `0.8125`;
  - `sid:6774f7e4355e`, best tested range around `F^160 -> G^32..128`:
    no consensus, min final-majority accuracy `0.90625`;
  - `sid:3fd35ca10c2e`, best tested range around `F^96 -> G^32..128`:
    no consensus, min final-majority accuracy `0.875`;
  - even the repeated-block champion `sid:58ed6b657afb` does not become a
    good one-switch rule on `64x64`; best tested result so far is
    `F^128 -> G^16` with no consensus and min final-majority accuracy
    `0.9375`.

- `[ ]` Search directly for the best first-phase NCCA `F` under a single
  switch to a second-phase amplifier `G`.

- `[ ]` Search directly for the best amplifier `G` given the currently best
  first-phase NCCA candidates.

- `[ ]` Fit size-dependent phase lengths `T1(L), T2(L)` for finite-switch
  constructions.

- `[~]` Multi-rule preprocessors built from the best balanced orthogonal
  equalizers.
  First mixed-block tests were not promising, but the idea is not fully
  closed because only one shallow family was tested.

- `[x]` Mixed-family preprocessors combining one orthogonal equalizer with one
  diagonal/transport-style rule such as `sid:812b7dae7aa7`.
  First mixed-block search did not beat the simpler repeated-block leaders.

- `[~]` Alternative amplifiers:
  - von Neumann majority,
  - alternating Moore/VN majority,
  - repeated short majority bursts separated by preprocessing.
  Result so far: both Moore and VN bursts can work inside repeated blocks;
  VN bursts pair especially well with `sid:029b09cea0b5`,
  `sid:812b7dae7aa7`, and now most strongly with `sid:58ed6b657afb`.

- `[ ]` Search for schedules that maximize tie-breaking rather than only
  smoothing:
  - local-majority margin,
  - fraction of strict local majorities,
  - decay of tie neighborhoods.

- `[~]` Search explicitly for **checkerboardizing** first phases.
  New metrics now track:
  - staggered checkerboard alignment,
  - orthogonal edge-disagreement rate,
  - fraction of checkerboard `2x2` blocks.
  First result: the strongest explicit checkerboardizer tested so far is the
  embedded cardinal traffic cycle, e.g.
  `traffic_east, traffic_north, traffic_west, traffic_south` repeated with
  `2` steps per direction. On `32x32` it reached about:
  - checkerboard `2x2` fraction `0.53`,
  - orthogonal disagreement `0.76`,
  - staggered alignment `0.55`.
  However, prepending such a checkerboard phase before strong NCCA
  preprocessors (`sid:029b09cea0b5`, `sid:812b7dae7aa7`) did **not** improve
  the first `64x64` one-switch tests; it usually matched or underperformed
  the plain NCCA-first baselines.
  A more promising variant is a **mixed anti-ferromagnetic prephase**:
  - repeat `(traffic_east, diag_traffic_ne, traffic_north, sid:029b09cea0b5)`
    for many short steps,
  - then switch once to `VN` majority.
  Current results:
  - `64x64`: this mixed prephase beats the plain `sid:029b09cea0b5 -> majority`
    baseline (`0.9375` vs `0.875` min final-majority accuracy),
  - `96x96`: it still improves the baseline (`0.9375` vs `0.90625`),
  - `128x128`: with `32` prephase repetitions it reaches `1.0` min
    final-majority accuracy,
  - but consensus is still `0.0`, so this improves the **correct bias**
    without yet supplying the final collapse mechanism.

- `[x]` Replace the final majority with simple threshold amplifiers.
  For the best current mixed anti-ferromagnetic prephase on `128x128`:
  - `VN` majority and Moore majority preserve perfect final-majority
    accuracy but still give zero consensus;
  - `vn_threshold_2`, `vn_threshold_4`, `moore_threshold_4`, and
    `moore_threshold_6` mostly collapse to trivial all-0 or all-1 outcomes.
  So the current bottleneck is not solved by naive threshold changes.

- `[~]` Broader balanced preprocessor search over a much larger slice of the
  `133713`-rule nonzero-mask catalog.
  We widened from the tiny hand-picked set to a `67`-rule repeated-block
  candidate pool, which already changed the leaders.  A much larger
  repeated-block search is still open.

- `[>]` Size-scaled repeated-block schedules.
  Result so far: very promising.  For `sid:58ed6b657afb`, increasing the
  number of repeated blocks with lattice size improved performance sharply.
  Current best checked examples:
  - `128x128`, `24` blocks of `8+16 VN`: consensus accuracy about
    `0.94 / 0.96` averaged over three seeds,
  - `192x192`, `32` blocks of `8+16 VN`: consensus accuracy about
    `0.969 / 1.000` in the latest 32-trial run,
  - `256x256`, `40` blocks of `8+16 VN`: consensus accuracy about
    `0.969 / 0.969` in a 32-trial confirmation run.

- `[ ]` Full repeated-block search over all `orthogonal_monotone`,
  `diagonal_monotone`, or other large structured families, ranked directly
  by consensus accuracy rather than one-shot smoothing.

- `[ ]` Escalate to the full `147,309,433` catalog if the nonzero-mask set still
  fails to produce a convincing schedule.

- `[ ]` Search for amplifiers rather than preprocessors: local rules whose main
  role is to turn weak local bias into consensus after a good equalizer has
  already run.

## Current Working Hypothesis

The main bottleneck now splits into two tracks:

1. a **CA baseline track**, where size-scaled repeated blocks already give
   extremely strong empirical performance;
2. a **beyond-CA Fukś-style track**, where the real open problem is to find
   a finite globally synchronized phase schedule that is not reducible to a
   single time-homogeneous binary CA.

### Update 2026-04-24: mechanism-probe findings

A direct mechanism probe of the best candidates
([MECHANISM_FINDINGS.md](MECHANISM_FINDINGS.md)) produced these facts:

- `sid:58ed6` and all other tested candidates do **not** coarsen or
  canonicalize.  Run alone they reach a frozen fragmented attractor with
  *lower* local-majority decodability than random init.
- Genuine coarseners exist in the broader catalog
  (e.g. `sid:11bb6b4211627713` drives interface/L² from 1.0 to 0.20),
  but they are black/white asymmetric and phase-separate into
  class-unpredictable blocks.
- **No single NCCA** out of 5000 random + 57 outer-monotone reaches
  nonzero correct consensus under `F^L → VN-majority^{T_2}`, even at
  T₂ = 1024.
- **No pair** out of 550 (5 budgets × 110 ordered pairs) under
  `(F_1^{t_1} F_2^{t_2})^T → VN-majority^{T_2}` either.
- The bottleneck is **2D majority's metastable fixed-point set**: long
  straight interfaces are invariants of Moore/VN majority, and every
  tested preprocessor output contains such interfaces. Majority freezes.

Conclusion under the strict formulation: perfect classification via a
finite-switch composition of NCCAs then plain local majority is likely
structurally impossible on 2D Moore-neighborhood.  The gap is on the
amplifier side, not the preprocessor side.  Relaxing to a larger-radius
majority or alternating-majority amplifier is the natural next move.

### Update 2026-04-24 (part 2): larger-radius amplifier unlocks random-input universality

Replaced plain Moore-9 / VN-5 majority with `moore81` (radius-2, 5×5 Moore,
threshold ≥13) as the amplifier.  With schedule
`F^{128} [moore81^{256} F^{64}]^{2} moore81^{1024}`, `F = sid:58ed6b657afb`:

- `256x256`: 100% correct consensus across 320 random-Bernoulli trials
  (density 0.49/0.51), 5 independent seeds.
- `384x384`: 100% over 320 trials.
- `512x512`: 100% over 96 trials.
- `192x192`: 99.38% (2/320 fail).

This is the first genuine **deterministic 2D finite-switch classifier** in
the project.  It is, formally, still one larger-radius CA per the
methodological note — but empirically it is substantially faster and more
robust than the earlier repeated-block lines.

### Update 2026-04-24 (part 3): structural theorem blocks universality on structured inputs

Adversarial stress test with pure stripes / checkerboards / half-half
splits at density exactly `1/2 + 1/L²` exposed a complete failure mode.

**Theorem** (proved in [`technical_report.tex`](technical_report.tex)):
any shift-invariant deterministic Moore-neighborhood rule F maps a
horizontal-stripe configuration to a stripe configuration (possibly
complemented).  For NCCAs, only preserve or complement is allowed by
density conservation.  The same holds for vertical stripes, checkerboards,
and block-checkers.  Hence any tipped stripe is a permanent obstruction
for the above deterministic schedule.

Implication: perfect universal classification in the pure deterministic
shift-invariant framework is impossible (a sharpening of the Land-Belew 1995
result).

### Update 2026-04-24 (part 4): Fatès literature review

Reviewed Fatès 2011 (STACS), Fatès-Regnault 2013 (ToCS), Fatès-Regnault
2016 (AUTOMATA).  Key takeaways:

- 1D stochastic: per-cell Bernoulli mixture of rule 184 (prob 1-η) and rule
  232 (prob η), η small.  Solves DCP with arbitrary precision.
- 2D: two stochastic number-conserving auxiliary rules F_X (lane-changing)
  and G_X (crowd-avoidance) composed with per-row R_184 / R_232.  100% at
  L=100 with T=16000 (Θ(L²) iterations).

Neither test adversarial structured inputs explicitly.  Neither uses our
enumerated-catalogue resource.

### Update 2026-04-24 (part 5): minimal-stochasticity experiments

Three noise mechanisms tested, all inserted during the shake phase of the
deterministic skeleton:

- `[x]` **Bernoulli flip noise** (per-cell independent flip with prob ε).
  ε ≈ 10⁻³ partially breaks stripes (adv cc 20% → 60%) but destroys the
  1-cell tipping bias (random cc 99% → 95%).  Not a clean fix.
- `[x]` **One-shot concentrated noise** (single burst of K uniform cell
  flips between phases).  Random cc is exactly independent of K; this
  confirms that random failures are NOT translational-invariance issues.
  Adversarial improvement minimal.
- `[>]` **Density-conserving random swaps** (K random pair swaps per
  shake).  Exactly density-conserving per realization; preserves
  random-input 100% perfectly; adversarial 20% → 70% at L=128 with K=8L.

Density-conserving swap noise strictly dominates per-cell flip noise.

### Update 2026-04-24 (part 6): Realistic density margins — THE schedule works

At density margins `|ρ - 1/2| ≥ 0.01` (i.e., the regime Fatès actually
tests, not the 1/L² tied regime), the L-scaled schedule below achieves
**100%** correct consensus on *all* adversarial configurations at
`L ∈ {192, 256, 384, 512}`:

```
F^{L/2}  [moore81^{L}  F^{L/4}  swap^{8L}]^{2}  moore81^{4L}
```

where `F = sid:58ed6b657afb`, `swap^K` = K random pair swaps.

Validation at 128 random-Bernoulli trials + 30 structured adversarial
trials per grid:

| L | random cc | adv cc |
|---|----------|--------|
| 64  | 93.75% | 86.67% |
| 128 | 98.44% | 100%   |
| 192 | **100%**   | **100%**   |
| 256 | **100%**   | **100%**   |
| 384 | **100%**   | **100%**   |
| 512 | **100%**   | **100%**   |

This is the current best result.  Detailed journey and per-experiment
references are in [STOCHASTIC_CLASSIFIER_NOTE.md](STOCHASTIC_CLASSIFIER_NOTE.md).

### Open next steps (post-2026-04-24)

- `[ ]` Full-catalogue scan (133,713 NCCAs) with the scaled schedule to
  find F candidates that work at L < 128 or tighter density margins.
- `[ ]` Scale-up test at L=1024, 2048 to verify the linear scaling holds.
- `[ ]` Tighten density margin test: 0.501, 0.5001.  Find the practical
  threshold.
- `[ ]` Attempt a convergence proof for the scaled stochastic classifier.
- `[ ]` Small-L regime (L < 192): use a size-adapted amplifier
  (moore25/moore49) instead of moore81 which wraps the torus at L = 64.
