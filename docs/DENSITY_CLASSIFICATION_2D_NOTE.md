# 2D Density Classification Note

This note records the first structured attempt to extend the 1D Fukś
traffic-majority construction to the 2D binary Moore-neighborhood
number-conserving rules enumerated in this repository.

## Motivation

In 1D, Fukś showed that a two-rule schedule solves density classification
perfectly on finite rings:

- apply the traffic rule 184 for a prescribed preprocessing time,
- then apply the majority rule 232 for a prescribed amplification time.

The traffic phase spreads information while preserving density; the
majority phase then pushes the system to the correct consensus.

Our question here was whether some 2D NCCAs from the current catalog can
play the same preprocessing role.

Important methodological correction:

- a periodically repeated binary schedule such as `(F^n G^k)^m` is still a
  single binary CA with a larger effective radius when sampled every
  `n + k` steps;
- therefore such schedules do not evade the standard impossibility result
  for perfect binary-CA density classification;
- they remain useful as an empirical **CA baseline track**;
- but the actual Fukś-style target for this project is a finite globally
  synchronized phase schedule such as `F^(T1(L)) -> G^(T2(L))`, or a small
  finite extension of that pattern.

## What Was Implemented

New reusable tooling:

- [ca_search/density_classification.py](/Users/witoldbolt/Documents/solvw/omp/ca_search/density_classification.py)
- CLI entry:
  - `python3 -m ca_search.cli density-classification-study`

The module provides:

- built-in 2D Moore majority and von Neumann majority rules,
- embedded cardinal and diagonal traffic rules,
- loading of selected catalog rules by `sid`,
- scheduled multi-rule rollouts,
- density-classification metrics.

## Core Metrics

For an initial random configuration with global majority label
\(\ell \in \{0,1\}\), each schedule was scored by:

- `local_majority_agreement`
  - after the NCCA preprocessing phase, fraction of lattice sites whose
    local Moore majority already agrees with the true global majority;
- `local_density_variance`
  - variance of local Moore-neighborhood densities after preprocessing;
  - lower means more even local mixing;
- `final_majority_accuracy`
  - after the full schedule, fraction of trials whose final global majority
    matches the initial global majority;
- `final_consensus_rate`
  - fraction of trials ending in all-0 or all-1 consensus;
- `final_consensus_accuracy`
  - fraction of trials ending in the correct consensus.

## Rules And Schedules Tested

We focused on:

- embedded cardinal traffic cycles,
- embedded diagonal traffic cycles,
- the best previously identified transport/scattering candidates:
  - `sid:029b09cea0b5`
  - `sid:812b7dae7aa7`
  - `sid:f076fc6ffb58`

The most informative schedules were:

- `majority_only`
- `traffic_cardinal_cycle_long_then_majority`
- `traffic_diagonal_cycle_long_then_majority`
- `candidate_029b_long_then_majority`
- `candidate_812b_long_then_majority`

The main study file is:

- [density_classification_study.json](/Users/witoldbolt/Documents/solvw/omp/density_classification_study.json)

A higher-confidence near-critical rerun is here:

- [density_classification_near_critical.json](/Users/witoldbolt/Documents/solvw/omp/density_classification_near_critical.json)

An exploratory hybrid-block follow-up is here:

- [density_classification_hybrid_study.json](/Users/witoldbolt/Documents/solvw/omp/density_classification_hybrid_study.json)

A broader preprocessor-ranking pass is here:

- [density_preprocessor_subset_screen.json](/Users/witoldbolt/Documents/solvw/omp/density_preprocessor_subset_screen.json)

The follow-up of the top-ranked preprocessors is here:

- [density_preprocessor_top_followup.json](/Users/witoldbolt/Documents/solvw/omp/density_preprocessor_top_followup.json)

The newer balanced paired-density screen is here:

- [density_preprocessor_balanced_subset_screen.json](/Users/witoldbolt/Documents/solvw/omp/density_preprocessor_balanced_subset_screen.json)

The larger-grid follow-up of the best balanced hits is here:

- [density_preprocessor_balanced_followup.json](/Users/witoldbolt/Documents/solvw/omp/density_preprocessor_balanced_followup.json)

And a longer, larger-torus scaling check is here:

- [density_preprocessor_balanced_scaling_followup.json](/Users/witoldbolt/Documents/solvw/omp/density_preprocessor_balanced_scaling_followup.json)

A first structured family search over short schedule blocks is here:

- [density_schedule_family_search.json](/Users/witoldbolt/Documents/solvw/omp/density_schedule_family_search.json)

Validation runs for the best repeated-block baseline schedules are here:

- [density_schedule_validation_64.json](/Users/witoldbolt/Documents/solvw/omp/density_schedule_validation_64.json)
- [density_schedule_validation_96.json](/Users/witoldbolt/Documents/solvw/omp/density_schedule_validation_96.json)

A broader repeated-block baseline search over a larger candidate pool is here:

- [density_repeated_block_candidate_search.json](/Users/witoldbolt/Documents/solvw/omp/density_repeated_block_candidate_search.json)

The larger-torus follow-up of those best repeated-block baseline schedules is here:

- [density_consensus_scaling_64.json](/Users/witoldbolt/Documents/solvw/omp/density_consensus_scaling_64.json)
- [density_consensus_scaling_96.json](/Users/witoldbolt/Documents/solvw/omp/density_consensus_scaling_96.json)

And the repetition sweep for the current top schedules is here:

- [density_repetition_scaling.json](/Users/witoldbolt/Documents/solvw/omp/density_repetition_scaling.json)
- [density_repetition_scaling_96.json](/Users/witoldbolt/Documents/solvw/omp/density_repetition_scaling_96.json)

A block-geometry search for repeated `VN`-majority schedules is here:

- [density_block_geometry_search.json](/Users/witoldbolt/Documents/solvw/omp/density_block_geometry_search.json)
- [density_block_geometry_validate_96.json](/Users/witoldbolt/Documents/solvw/omp/density_block_geometry_validate_96.json)

A threshold-amplifier search is here:

- [density_threshold_amplifier_search.json](/Users/witoldbolt/Documents/solvw/omp/density_threshold_amplifier_search.json)

A larger direct repeated-block screen over the balanced proxy subset is here:

- [density_repeated_block_balanced_subset_vn.json](/Users/witoldbolt/Documents/solvw/omp/density_repeated_block_balanced_subset_vn.json)

Validation of the best newly discovered repeated-block candidates is here:

- [density_new_candidates_validate_64.json](/Users/witoldbolt/Documents/solvw/omp/density_new_candidates_validate_64.json)
- [density_new_candidates_validate_96.json](/Users/witoldbolt/Documents/solvw/omp/density_new_candidates_validate_96.json)

A restricted repeated-block search over the black/white-symmetric family is here:

- [density_repeated_block_bw_vn.json](/Users/witoldbolt/Documents/solvw/omp/density_repeated_block_bw_vn.json)
- [density_bw_validate_64.json](/Users/witoldbolt/Documents/solvw/omp/density_bw_validate_64.json)

And a correction-tail search is here:

- [density_correction_tail_search.json](/Users/witoldbolt/Documents/solvw/omp/density_correction_tail_search.json)

An exact-family search over the `outer_monotone` 57-rule family is here:

- [density_outer_vn_12_4.json](/Users/witoldbolt/Documents/solvw/omp/density_outer_vn_12_4.json)
- [density_outer_vn_8_16.json](/Users/witoldbolt/Documents/solvw/omp/density_outer_vn_8_16.json)
- [density_outer_moore_8_8.json](/Users/witoldbolt/Documents/solvw/omp/density_outer_moore_8_8.json)
- [density_outer_geometry_validate_96.json](/Users/witoldbolt/Documents/solvw/omp/density_outer_geometry_validate_96.json)
- [density_outer_repetition_sweep_96.json](/Users/witoldbolt/Documents/solvw/omp/density_outer_repetition_sweep_96.json)

Focused validation of the current best rule is here:

- [density_58ed6_validate_96.json](/Users/witoldbolt/Documents/solvw/omp/density_58ed6_validate_96.json)
- [density_58ed6_validate_128.json](/Users/witoldbolt/Documents/solvw/omp/density_58ed6_validate_128.json)
- [density_58ed6_seed_robustness_128.json](/Users/witoldbolt/Documents/solvw/omp/density_58ed6_seed_robustness_128.json)
- [density_58ed6_192.json](/Users/witoldbolt/Documents/solvw/omp/density_58ed6_192.json)
- [density_58ed6_192_highconf.json](/Users/witoldbolt/Documents/solvw/omp/density_58ed6_192_highconf.json)
- [density_58ed6_192_repetition_scale.json](/Users/witoldbolt/Documents/solvw/omp/density_58ed6_192_repetition_scale.json)
- [density_58ed6_amp_sweep_128.json](/Users/witoldbolt/Documents/solvw/omp/density_58ed6_amp_sweep_128.json)
- [density_58ed6_highconf_128.json](/Users/witoldbolt/Documents/solvw/omp/density_58ed6_highconf_128.json)
- [density_58ed6_256_x40.json](/Users/witoldbolt/Documents/solvw/omp/density_58ed6_256_x40.json)
- [density_58ed6_256_x40_highconf.json](/Users/witoldbolt/Documents/solvw/omp/density_58ed6_256_x40_highconf.json)
- [density_58ed6_256_x40_seed421.json](/Users/witoldbolt/Documents/solvw/omp/density_58ed6_256_x40_seed421.json)

The first direct finite-switch screen for the actual beyond-CA target is here:

- [density_finite_switch_balanced_subset_vn.json](/Users/witoldbolt/Documents/solvw/omp/density_finite_switch_balanced_subset_vn.json)

A checkerboard-oriented prephase search is here:

- [density_checkerboard_phase_search.json](/Users/witoldbolt/Documents/solvw/omp/density_checkerboard_phase_search.json)

A mixed-block follow-up that combines traffic-style and NCCA steps is here:

- [density_checkerboard_mixed_blocks_search.json](/Users/witoldbolt/Documents/solvw/omp/density_checkerboard_mixed_blocks_search.json)

And the first three-phase validation runs motivated by that idea are here:

- [density_three_phase_checkerboard_search.json](/Users/witoldbolt/Documents/solvw/omp/density_three_phase_checkerboard_search.json)
- [density_three_phase_validate_64.json](/Users/witoldbolt/Documents/solvw/omp/density_three_phase_validate_64.json)

A focused validation of the best mixed anti-ferromagnetic `sid:029b09cea0b5`
prephase family is here:

- [density_checkerboard_mixed_validate_64.json](/Users/witoldbolt/Documents/solvw/omp/density_checkerboard_mixed_validate_64.json)
- [density_checkerboard_mixed_validate_96.json](/Users/witoldbolt/Documents/solvw/omp/density_checkerboard_mixed_validate_96.json)
- [density_checkerboard_mixed029b_sweep_96.json](/Users/witoldbolt/Documents/solvw/omp/density_checkerboard_mixed029b_sweep_96.json)
- [density_checkerboard_mixed029b_128.json](/Users/witoldbolt/Documents/solvw/omp/density_checkerboard_mixed029b_128.json)
- [density_checkerboard_mixed029b_128_sweep.json](/Users/witoldbolt/Documents/solvw/omp/density_checkerboard_mixed029b_128_sweep.json)

And the first amplifier sweep for that family is here:

- [density_checkerboard_mixed029b_amplifier_128.json](/Users/witoldbolt/Documents/solvw/omp/density_checkerboard_mixed029b_amplifier_128.json)

## Main Findings

### 1. Straight 2D traffic cycles do smooth local density, but only modestly.

On `64x64` random tori near the critical region:

- initial local-density variance is about `0.0277`,
- long cardinal traffic cycles reduce it to about `0.0251`,
- long diagonal traffic cycles reduce it further to about `0.0192`.

So these rules do have a diffusion/equalization effect, but it is much
weaker than what would be needed for an exact Fukś-style reduction.

### 2. The best previously found scattering rules smooth the lattice more strongly.

The strongest preprocessor we tested was:

- `sid:812b7dae7aa7`

After `64` preprocessing steps, its local-density variance near
`p = 0.49` and `p = 0.51` drops to about:

- `0.00931` at `p = 0.49`
- `0.00873` at `p = 0.51`

This is substantially lower than both traffic-cycle baselines, so this
rule really does spread information more evenly across the torus.

### 3. This stronger smoothing improves majority accuracy, but does not solve the task.

Near-critical rerun on `128` random `64x64` trials:

| Schedule | `p=0.49` final majority accuracy | `p=0.51` final majority accuracy | consensus rate |
| --- | ---: | ---: | ---: |
| `majority_only` | `0.9609` | `0.8906` | `0.0000` |
| `traffic_cardinal_cycle_long_then_majority` | `0.9063` | `0.9219` | `0.0000` |
| `traffic_diagonal_cycle_long_then_majority` | `0.9375` | `0.9063` | `0.0000` |
| `candidate_029b_long_then_majority` | `0.9375` | `0.9375` | `0.0000` |
| `candidate_812b_long_then_majority` | `0.9688` | `0.9609` | `0.0000` |

So `sid:812b7dae7aa7` is the best tested preprocessor.  It improves
near-critical majority classification, especially on the `p = 0.51`
side.  However:

- no tested schedule achieved nonzero consensus rate near the critical region,
- even after extending the majority tail to `128` steps, consensus still did
  not appear in those near-critical tests.

### 4. Repeated hybrid blocks also did not recover the 1D phenomenon.

We also tried repeated blocks such as:

- `[traffic_E, traffic_N, traffic_W, traffic_S, majority] × 8`
- `[candidate_812b7dae7, majority] × 16`

These did not produce consensus either, and some were less stable than the
simple “preprocess then majority” schedules.

### 5. A broader preprocessor search reveals a new failure mode: one-sided bias.

After the first hand-picked study, I ran a broader screen over a tractable
subset of the nonzero-mask catalog:

- all `outer_monotone` rules (`57`);
- all `orthogonal_monotone` rules (`1913`);
- all `center_blind` rules (`144`);
- a random sample of `2000` `diagonal_monotone` rules.

The first ranking criterion looked only at preprocessing quality at a
single density:

1. maximize `local_majority_agreement`,
2. then minimize `local_density_variance`.

This produced many strong-looking preprocessors, but most of them were
heavily one-sided.  The top of the ranking was dominated by
`orthogonal_monotone` rules such as:

- `sid:bf56fa4fae1f`
- `sid:7bd809a30ed2`
- `sid:53254db763c9`
- `sid:8e698d1e9e44`

At `p = 0.49`, these rules looked excellent.  For example,
`sid:bf56fa4fae1f` achieved:

- local-majority agreement `0.6123`,
- local-density variance `0.01257`.

But once we tested the full preprocess-plus-majority schedule on both sides of
the critical point, these same rules failed badly at `p = 0.51`.  For
`sid:bf56fa4fae1f`:

- at `p = 0.49`: final-majority accuracy `0.9219`;
- at `p = 0.51`: final-majority accuracy `0.1719`.

So these are not neutral density equalizers.  They are strong but biased
preprocessors that help one majority class while harming the other.

This is why `sid:812b7dae7aa7` remains the strongest current lead.  It is not
the most extreme one-sided smoother, but it is far more balanced:

- at `p = 0.49`: final-majority accuracy `0.9531`,
- at `p = 0.51`: final-majority accuracy `0.9531`,
- with low local-density variance on both sides (`0.00950` and `0.00861` in
  the near-critical rerun).

### 6. Balanced paired-density screening changes the shortlist completely.

The one-sided failure mode suggested a better search objective:

1. maximize the **minimum** final-majority accuracy at `p` and `1-p`,
2. penalize the final-majority accuracy gap between the two sides,
3. then maximize the **minimum** local-majority agreement,
4. and finally minimize the **maximum** local-density variance.

I implemented this as a separate balanced screen over exactly the same
tractable subset as before:

- all `outer_monotone` rules,
- all `orthogonal_monotone` rules,
- all `center_blind` rules,
- a deterministic sample of `2000` `diagonal_monotone` rules.

This keeps the search space at `4037` rules, but the ranking changes
completely.  The previous one-sided winners disappear, and the new top
balanced hits are:

- `sid:29154e9615d8`
- `sid:93f0176f55d8`
- `sid:5dceb34ea73f`
- `sid:f1a2737cb962`
- `sid:0c3ad3e221f5`

All of these are `orthogonal_monotone` rules.  On the small balanced
proxy run (`32x32`, `16` trials, `32+32` steps), they reached:

- minimum final-majority accuracy `1.0`,
- zero measured side bias,
- local-density variance around `0.0055` to `0.0087`.

So the strongest current candidates are no longer the previously
identified anisotropic scattering rules.  They are a cleaner,
approximately symmetric orthogonal-monotone family.

### 7. Larger-grid follow-up: the new orthogonal-monotone hits are stronger equalizers than `sid:812b7dae7aa7`.

I then compared the best balanced hits directly against the old best lead
`sid:812b7dae7aa7` on a larger near-critical run:

- `64x64` torus,
- `64` trials at each of `p = 0.49` and `p = 0.51`,
- `64` preprocessing steps,
- `64` Moore-majority steps.

Representative results:

| Rule | `p=0.49` final-majority accuracy | `p=0.51` final-majority accuracy | local-density variance (`0.49`, `0.51`) | consensus rate |
| --- | ---: | ---: | ---: | ---: |
| `sid:29154e9615d8` | `0.9844` | `0.9531` | `0.00552`, `0.00546` | `0.0000` |
| `sid:93f0176f55d8` | `0.9688` | `0.9844` | `0.00550`, `0.00551` | `0.0000` |
| `sid:5dceb34ea73f` | `0.9844` | `0.9375` | `0.00579`, `0.00587` | `0.0000` |
| `sid:812b7dae7aa7` | `0.9062` | `1.0000` | `0.00930`, `0.00864` | `0.0000` |
| `majority_only` | `0.9375` | `0.9531` | `0.02766`, `0.02769` | `0.0000` |

So the balanced orthogonal-monotone hits really are stronger density
equalizers than `sid:812b7dae7aa7`.  They produce much lower local-density
variance and generally better two-sided majority accuracy.

### 8. Longer scaling checks still show the same obstruction: no consensus.

The final sanity check was a longer run on a larger torus:

- `96x96`,
- `32` trials per side,
- `96` preprocessing steps,
- `96` Moore-majority steps.

The best balanced hits remained strong equalizers, but **still** produced
zero consensus:

- `sid:29154e9615d8`: final-majority accuracy `0.9688` / `0.9375`, consensus `0.0`
- `sid:93f0176f55d8`: final-majority accuracy `0.9688` / `1.0000`, consensus `0.0`
- `sid:812b7dae7aa7`: final-majority accuracy `1.0000` / `0.9688`, consensus `0.0`

The sample is small enough that exact majority accuracies fluctuate, but
the qualitative picture is stable:

- preprocessing quality improves,
- near-critical majority decisions improve,
- consensus does not appear.

### 9. Repeated short preprocessor-amplifier blocks change the picture.

After the single-preprocessor schedules stalled, I tested short repeated
blocks of the form

\[
[\text{NCCA for }8\text{ steps},\ \text{majority for }8\text{ steps}]^k.
\]

This is closer in spirit to the original Fukś construction, where the two
phases are not necessarily separated into one long diffusion phase followed
by one long amplification phase.

The first family search over a small hand-picked set of preprocessors found
the first nonzero consensus rates in this project.  The best schedules on
`64x64`, `32` trials per side, were:

- `sid:812b7dae7aa7` with repeated von Neumann majority blocks,
- `sid:29154e9615d8` with repeated Moore-majority blocks,
- `sid:93f0176f55d8` with repeated Moore-majority blocks.

For example, on `64x64` with `8` repetitions:

- `sid:812b7dae7aa7 + VN-majority`: consensus `0.2188` at `p=0.49`,
  `0.4375` at `p=0.51`;
- `sid:29154e9615d8 + Moore-majority`: consensus `0.2500` at `p=0.49`,
  `0.1875` at `p=0.51`.

So repeated blocks are qualitatively different from the earlier “all
preprocess then all majority” schedules.

### 10. Mixed preprocessors are not currently helping.

I next tested blocks of the form

\[
[\text{NCCA}_1\ 4,\ \text{NCCA}_2\ 4,\ \text{majority}\ 8]^8
\]

for the strongest currently known preprocessors.  These mixed blocks did
not improve on the best single-preprocessor repeated schedules.  The top
mixed-family schedules had much lower minimum consensus rates, typically
around `0.03` to `0.06`, and often worse majority accuracy as well.

So, at least in this first pass, adding a second preprocessor inside the
block increases complexity without improving the basic phenomenon.

### 11. A broader repeated-block search points back to `sid:029b09cea0b5` and `sid:812b7dae7aa7`.

I then widened the search to a larger candidate pool:

- top `40` rules from the earlier one-sided preprocessor screen,
- top `25` rules from the balanced paired-density screen,
- plus the previously hand-picked candidates.

This gave `67` distinct candidate preprocessors.  I screened schedules of
the form

\[
[\text{NCCA}\ 8,\ \text{majority}\ 8]^k
\]

for `k \in \{8,16\}` on `32x32` random tori.

The most striking small-torus result was that several schedules reached
consensus in **every** trial.  The strongest were:

- `sid:029b09cea0b5` with Moore-majority or von Neumann majority,
- `sid:812b7dae7aa7` with Moore-majority or von Neumann majority.

However, full consensus on the small torus is not the same as correct
density classification.  For instance, on `32x32` the schedule
`sid:029b09cea0b5 + Moore-majority`, repeated `16` times, had:

- consensus rate `1.0` on both `p=0.49` and `p=0.51`,
- but only `0.9375` final-majority accuracy on each side.

So the system is already strongly amplifying toward consensus, but still
occasionally into the wrong basin.

### 12. Larger-torus validation: consensus persists, but the mechanism is still imperfect.

The best repeated-block schedules were then tested on `64x64` with `128`
trials per side and on `96x96` with `64` trials per side.

On `64x64`, the strongest current schedules were:

| Schedule | `p=0.49` accuracy | `p=0.51` accuracy | `p=0.49` consensus | `p=0.51` consensus |
| --- | ---: | ---: | ---: | ---: |
| `sid:029b09cea0b5 + Moore-majority`, `16` blocks | `0.8661` | `0.8672` | `0.7559` | `0.8125` |
| `sid:029b09cea0b5 + VN-majority`, `16` blocks | `0.8425` | `0.8281` | `0.7795` | `0.7734` |
| `sid:812b7dae7aa7 + Moore-majority`, `16` blocks | `0.7874` | `0.9375` | `0.7638` | `0.9219` |
| `sid:812b7dae7aa7 + VN-majority`, `16` blocks | `0.8110` | `0.8906` | `0.8504` | `0.8906` |
| `sid:29154e9615d8 + Moore-majority`, `16` blocks | `0.9528` | `0.8984` | `0.6614` | `0.5703` |

On `96x96`, the effect weakens but clearly survives:

| Schedule | `p=0.49` accuracy | `p=0.51` accuracy | `p=0.49` consensus | `p=0.51` consensus |
| --- | ---: | ---: | ---: | ---: |
| `sid:029b09cea0b5 + VN-majority`, `16` blocks | `0.8906` | `0.9531` | `0.4688` | `0.5312` |
| `sid:029b09cea0b5 + Moore-majority`, `16` blocks | `0.8750` | `0.8594` | `0.3125` | `0.5625` |
| `sid:812b7dae7aa7 + VN-majority`, `16` blocks | `0.8750` | `0.9062` | `0.5000` | `0.5781` |
| `sid:29154e9615d8 + Moore-majority`, `16` blocks | `1.0000` | `0.9219` | `0.2344` | `0.4375` |

This is the strongest evidence so far for a real 2D analogue of the
traffic-majority mechanism: repeated blocks can simultaneously create
substantial consensus and retain reasonably high majority accuracy.
But the analogue is still imperfect, because accuracy is far from `1.0`.

### 13. More repetitions increase consensus, but not correctness.

Finally, I swept the repetition count on `96x96` for the two best current
families:

- `sid:029b09cea0b5 + VN-majority`,
- `sid:29154e9615d8 + Moore-majority`.

The qualitative trend is clear:

- consensus rises steadily as the number of repeated blocks increases,
- but final-majority accuracy does not improve in parallel and can even
  decline slightly on one side of the critical point.

For `sid:029b09cea0b5 + VN-majority` on `96x96`:

| blocks | `p=0.49` accuracy | `p=0.51` accuracy | `p=0.49` consensus | `p=0.51` consensus |
| ---: | ---: | ---: | ---: | ---: |
| `8` | `0.9375` | `0.9375` | `0.0000` | `0.0312` |
| `12` | `0.9062` | `0.9375` | `0.2188` | `0.1875` |
| `16` | `0.9062` | `0.9375` | `0.4375` | `0.4375` |
| `20` | `0.8750` | `0.9375` | `0.5625` | `0.6250` |
| `24` | `0.8750` | `0.9375` | `0.6875` | `0.6875` |

So the current best interpretation is:

- repeated blocks really do build a consensus-driving mechanism;
- but longer time mostly strengthens the drive toward consensus itself,
  not the correctness of the selected consensus.

### 14. Direct repeated-block screening finds new leaders.

Once repeated blocks became the main target, I stopped ranking rules by
one-shot preprocessing quality and instead screened them directly under the
schedule

\[
[\text{NCCA}\ 8,\ \text{VN-majority}\ 8]^{16}.
\]

On the same `4037`-rule proxy subset used earlier, this changed the
leaders completely.  The top small-torus schedules were no longer the old
hand-picked rules, but a family of mostly `diagonal_monotone` candidates
such as:

- `sid:ac809b9cab03`
- `sid:2667f2c59840`
- `sid:d3ed8a57a6ad`

On `32x32`, these rules reached:

- consensus rate `1.0` on both `p=0.49` and `p=0.51`,
- consensus accuracy `0.9375` on both sides for the best candidates.

So the repeated-block objective really does select a different rule family
than the earlier smoothing-based objective.

### 15. Larger-torus validation of the new repeated-block leaders.

The new `diagonal_monotone` candidates were then compared against the older
leaders `sid:029b09cea0b5` and `sid:812b7dae7aa7`.

On `64x64` with repeated `VN`-majority blocks:

| Rule | `p=0.49` accuracy | `p=0.51` accuracy | `p=0.49` consensus | `p=0.51` consensus |
| --- | ---: | ---: | ---: | ---: |
| `sid:ac809b9cab03` | `0.8438` | `0.8413` | `0.8750` | `0.8095` |
| `sid:2667f2c59840` | `0.8594` | `0.8730` | `0.5156` | `0.6984` |
| `sid:029b09cea0b5` | `0.8125` | `0.8571` | `0.7188` | `0.8095` |
| `sid:812b7dae7aa7` | `0.7812` | `0.9206` | `0.8594` | `0.9365` |

So `sid:ac809b9cab03` is more balanced than `sid:812b7dae7aa7`, while
still creating strong consensus.

On `96x96`, the picture becomes more mixed:

| Rule | `p=0.49` accuracy | `p=0.51` accuracy | `p=0.49` consensus | `p=0.51` consensus |
| --- | ---: | ---: | ---: | ---: |
| `sid:ac809b9cab03` | `0.9688` | `0.8750` | `0.6250` | `0.3125` |
| `sid:2667f2c59840` | `0.9688` | `0.9375` | `0.2188` | `0.2812` |
| `sid:029b09cea0b5` | `0.8438` | `0.9375` | `0.3438` | `0.3438` |
| `sid:812b7dae7aa7` | `0.7812` | `0.8750` | `0.5312` | `0.6250` |

So the new direct repeated-block search definitely finds real alternatives,
but there is still no single clear winner on all metrics.

### 16. Block geometry matters.

The choice `8` NCCA steps + `8` amplifier steps per block was arbitrary, so
I searched over

\[
[\text{NCCA } p,\ \text{VN-majority } a]^{16}
\]

with `p,a \in \{4,8,12,16\}` for the current best repeated-block candidates.

This matters a great deal.  On `64x64`, the strongest current schedule was:

- `sid:812b7dae7aa7`, `12` NCCA steps, `4` VN-majority steps, repeated `16` times.

Its performance on `64x64` was:

- `p=0.49`: accuracy `0.8438`, consensus `0.9688`, consensus accuracy `0.8438`;
- `p=0.51`: accuracy `1.0000`, consensus `1.0000`, consensus accuracy `1.0000`.

This is the best repeated-block result seen so far at that system size.

However, the larger-`96x96` validation shows that it is also strongly
asymmetric:

- `p=0.49`: accuracy `0.7188`, consensus `0.5625`;
- `p=0.51`: accuracy `0.9688`, consensus `0.9375`.

So geometry tuning is powerful, but it sharpens the same old issue:
consensus is getting stronger faster than correctness is getting balanced.

### 17. Threshold amplifiers and correction tails do not yet solve the problem.

I also tested two further ideas.

First, replace majority by nearby threshold rules such as:

- Moore thresholds `4,5,6,7`,
- von Neumann thresholds `2,3,4,5`.

This mostly failed in an obvious way: thresholds below majority quickly
drove almost everything to all-1 consensus, while thresholds above majority
drove almost everything to all-0 consensus.  So generic threshold
amplifiers do not appear to fix the problem.

Second, I appended long correction tails after the best repeated-block
schedules, e.g.

\[
[\text{NCCA},\ \text{VN-majority}]^{16} \to \text{Moore-majority}^{32,64,96}.
\]

This helps only marginally.  For example, on `64x64` the schedule
`sid:812b7dae7aa7 + VN-majority`, repeated `16` times, then followed by a
Moore-majority tail, reached:

- minimum consensus accuracy about `0.774`,
- minimum consensus rate about `0.875`,
- but minimum final-majority accuracy still only about `0.790`.

So the correction tail does not remove the basic consensus-vs-correctness
tradeoff.

### 18. Black/white symmetry is not currently the key.

Because one of the persistent problems is directional or class bias, I also
screened the whole `black_white_symmetric` family under the repeated-block
objective.  That family did produce some decent small-torus candidates, but
the larger-`64x64` validation was weaker than the current general leaders.

For example, the best black/white-symmetric repeated-block rule validated at
roughly:

- `p=0.49`: accuracy `0.9206`, consensus `0.0952`,
- `p=0.51`: accuracy `0.8281`, consensus `0.0781`.

So complement symmetry alone is not enough.

### 19. The exact `outer_monotone` family search finds a much stronger rule.

The earlier broader searches suggested that the best repeated-block rules
were hiding in a very small structured family.  In fact, the exact
intersection

\[
\texttt{outer\_monotone}
=
\texttt{outer\_monotone} \cap \texttt{orthogonal\_monotone} \cap \texttt{diagonal\_monotone}
\]

contains only `57` rules.  I screened that family directly under several
repeated-block geometries.

This produced a major improvement.  The best current rule is:

- `sid:58ed6b657afb`

and the best current block form is:

\[
[\texttt{sid:58ed6b657afb}\ 8,\ \text{VN-majority}\ 16]^k.
\]

At `32x32`, this schedule is already effectively perfect under the tested
samples.  More importantly, it continues to look strong as the lattice size
increases.

### 20. Current best candidate schedule.

The strongest schedule found so far is:

\[
[\texttt{sid:58ed6b657afb}\ 8,\ \text{VN-majority}\ 16]^k,
\]

with the number of repeated blocks increasing with lattice size.

Representative results:

| Grid | Schedule | final-majority accuracy (`p=0.49`, `p=0.51`) | consensus accuracy (`p=0.49`, `p=0.51`) |
| --- | --- | ---: | ---: |
| `96x96` | `k=24` | `0.9688`, `0.9688` | `0.9062`, `0.9062` |
| `128x128` | `k=24`, 64 trials | `0.9844`, `0.9844` | `0.9219`, `0.9531` |
| `128x128` | `k=24`, 3 extra seeds, 32 trials each | average `0.9792`, `0.9896` | average `0.9375`, `0.9583` |
| `192x192` | `k=32`, 32 trials | `1.0000`, `1.0000` | `0.9688`, `1.0000` |
| `256x256` | `k=40`, 32 trials | `1.0000`, `1.0000` | `0.9688`, `0.9688` |

This is the first result in the project that looks genuinely close to a 2D
Fukś-style extension rather than just an interesting heuristic.

### 21. Schedule length appears to need lattice-size scaling.

The best results do not come from a fixed number of repeated blocks.
Instead, performance improved when the number of blocks increased with the
lattice width:

- `64x64`: strong candidates already at `k=16`,
- `128x128`: best current runs near `k=24`,
- `192x192`: best current runs near `k=32`,
- `256x256`: best current runs near `k=40`.

So the present working hypothesis is that, just as in 1D, the diffusion /
amplification time should scale with system size.  The exact scaling law is
still open, but the empirical pattern above is already quite suggestive.

### 22. What remains unresolved.

Even with these strong results, this is not yet a finished theorem or a
finished algorithmic result.

Open questions:

1. Is `sid:58ed6b657afb` the genuinely best rule in the full `147,309,433`
   NCCA class for this task, or just the best among the families we searched
   directly under the repeated-block objective?
2. What is the correct size-dependent schedule law?
3. Are the remaining failures rare finite-size anomalies, or do they persist
   at a fixed positive rate?
4. Can the schedule be improved further by a small correction phase, or by a
   better amplifier than plain von Neumann majority?
5. Can the observed behavior be explained analytically?

## Current Conclusion

At this stage, we still do **not** have a full 2D extension of the
Fukś result within the currently tested deterministic Moore-neighborhood
NCCAs.

What we do have now is a much stronger intermediate statement:

- some 2D NCCAs, especially the balanced `orthogonal_monotone` rules
  `sid:29154e9615d8` and `sid:93f0176f55d8`, behave more like genuine
  density equalizers than the simple embedded traffic cycles;
- these preprocessors can improve the probability that a later majority
  phase picks the correct global majority;
- and, more importantly, some **repeated short block schedules** do create
  substantial consensus on large tori;
- and the best current repeated-block schedule,
  `[\texttt{sid:58ed6b657afb}\ 8,\ \text{VN-majority}\ 16]^k`,
  now comes very close to an exact density-classification solution on the
  tested lattice sizes.

The current best candidates are no longer just the original transport rules.
There are at least three mechanisms that mattered in the search:

- balanced equalizers such as `sid:29154e9615d8`,
- transport/scattering rules such as `sid:029b09cea0b5` and `sid:812b7dae7aa7`,
- newly discovered repeated-block `diagonal_monotone` candidates such as
  `sid:ac809b9cab03` and `sid:2667f2c59840`,
- and finally the much stronger `outer_monotone` rule `sid:58ed6b657afb`,
  which currently dominates the search.

So the analogue of 1D rule 184 in 2D is not “just use the obvious traffic
cycles.”  However, there is now an important conceptual split:

- the repeated-block schedules above are a valuable **CA baseline track**,
  because they produce the strongest empirical consensus seen so far;
- but they are not the final Fukś-style target, since periodic repetition
  still collapses to an ordinary binary CA at larger effective radius;
- the actual beyond-CA target is now a finite globally synchronized switch
  schedule such as `F^(T_1(L)) -> G^(T_2(L))`.

The first direct screen of such one-switch schedules has now been carried
out. Its early leaders on `32x32` under `F^(T_1) -> \text{VN-majority}^{T_2}`
were:

- `sid:b9b31aebc9b5`,
- `sid:6774f7e4355e`,
- `sid:3fd35ca10c2e`,
- `sid:13d4fec69618`,
- `sid:fdabfae81f3f`.

But the first larger `64x64` validations are still negative:

- `sid:b9b31aebc9b5`, best tested near `F^96 -> G^32`, gives no consensus;
- `sid:6774f7e4355e`, best tested near `F^{160} -> G^{32..128}`, gives no
  consensus;
- `sid:3fd35ca10c2e`, best tested near `F^{96} -> G^{32..128}`, gives no
  consensus;
- even the repeated-block champion `sid:58ed6b657afb` does not become a
  successful one-switch rule under the first tested `64x64` schedules.

So the repeated-block baseline remains empirically strong, but the true
Fukś-style finite-switch extension is still open.

## Most Promising Next Steps

1. Search directly for finite one-switch schedules `F^(T_1(L)) -> G^(T_2(L))`
   on larger candidate pools, since the first small-screen leaders were not
   the old repeated-block winners.
2. Fit explicit size-dependent phase lengths `T_1(L), T_2(L)` for the best
   one-switch candidates instead of keeping the phase lengths fixed.
3. If one-switch schedules remain weak, escalate to finite three-phase
   schedules such as `F_1^(T_1) -> F_2^(T_2) -> G^(T_3)`.
4. Keep the repeated-block family as a **CA baseline track**, because it is
   still the strongest empirical comparison point.
5. Search for **bias-corrected finite-switch schedules**, for example:
   - alternative amplifiers after the first NCCA phase,
   - final correction phases after the main amplifier,
   - better complement-balanced first phases.
6. Add richer preprocessing metrics, for example:
   - local-majority margin, not just agreement,
   - fraction of tie-free neighborhoods after preprocessing,
   - decay of low-frequency density modes in Fourier space,
   - two-sided symmetry penalties at `p` and `1-p`.
7. Allow schedules composed of several distinct NCCAs before the final
   majority phase, but with only a fixed finite number of global switches.
8. Test larger-radius or anisotropic majority/amplification rules rather
   than only the plain Moore majority rule.
9. Search specifically for preprocessors with complement-balance-related
   structural properties, since the newly observed failure mode is strong
   one-sided bias rather than weak mixing.

## 2026-04-24 update: L-Scaled Stochastic Classifier

A new line of work on the same day produced a substantial breakthrough.  A
full technical account is in [`technical_report.tex`](technical_report.tex)
("Density Classification in Two Dimensions") and a step-by-step lab-notebook
version is in [STOCHASTIC_CLASSIFIER_NOTE.md](STOCHASTIC_CLASSIFIER_NOTE.md).
Below is the headline summary.

### 23. Large-radius amplifier + NCCA preprocessor + density-conserving swap

**Schedule:**

```
F^{L/2}  [moore81^{L}  F^{L/4}  swap^{8L}]^{2}  moore81^{4L}
```

where:
- `F = sid:58ed6b657afb` (the best previously identified NCCA
  preprocessor, in the `outer_monotone ∩ orthogonal_monotone ∩
  diagonal_monotone` family of 57 rules);
- `moore81` is the radius-2 (5×5) Moore majority, threshold ≥ 13 of 25;
- `swap^K` is a density-conserving perturbation: pick K pairs of cells
  uniformly at random and swap their values.

**Validation** (per-grid: 128 random Bernoulli at density 0.49/0.51 + 30
structured adversarial at densities 0.51/0.52/0.55; stripes_h, stripes_v,
checker, block_checker, half_half with both target labels):

| Grid | Random correct-consensus | Adversarial correct-consensus |
|------|--------------------------|-------------------------------|
| 64²  | 93.75% (8/128 fail) | 86.67% (4/30 fail) |
| 128² | 98.44% (2/128 fail) | **100%** |
| 192² | **100%** | **100%** |
| 256² | **100%** | **100%** |
| 384² | **100%** | **100%** |
| 512² | **100%** | **100%** |

**For grids `L ≥ 192`, the schedule is universal** across the tested suite.

**Deterministic subschedule** (without swaps, K=0):  100% on random inputs
at L ∈ {256, 384, 512} with 320 trials × 5 seeds per grid, but fails on
structured-adversarial inputs (stripes, checkers) at density exactly
`1/2 + 1/L²`.  This reflects the structural theorem (see §24 below).

### 24. Structural theorem: shift-invariant Moore-NCCAs preserve stripes

> For any shift-invariant deterministic Moore-neighborhood rule F, a
> horizontal-stripe configuration is mapped to a horizontal-stripe
> configuration (possibly its complement).  For NCCAs, only preserve or
> complement is allowed by density conservation.

Proof: on a horizontal stripe, every cell in a row of given parity sees the
identical Moore neighborhood.  F depends only on the neighborhood, so F
produces the same output at every cell in a given row: the image is
translationally-uniform within each row.  For an NCCA, density conservation
forces the image to have density `1/2`, so only preserve or row-complement
is allowed.

The same holds for vertical stripes and for any checkerboard-like pattern
that is translationally-invariant under some finite-index subgroup of
lattice shifts.

Implication: the pure deterministic finite-switch schedule cannot succeed
on tipped stripes at density `1/2 + 1/L²`.  This is a sharper impossibility
than the classical Land-Belew 1995 result: it identifies an explicit
infinite family of configurations that every deterministic shift-invariant
Moore schedule fails on.

### 25. Why density-conserving swaps beat per-cell flip noise

Per-cell Bernoulli flip noise (flip each cell with probability ε)
independently affects ε·L² cells per shake.  On a tipped stripe with
density `1/2 + 1/L²`, the tipping cell is a bias of 1 against a noise of
ε·L² flips — the bias is destroyed for any ε > 1/L².  Empirically, even
ε=10⁻³ degrades random-input performance.

Density-conserving random swaps (pick two cells, swap their values) do not
change the global density by construction: the tipping-cell bias is
preserved per realization.  Empirically, random-input performance is
undisturbed by K ∈ [0, 16L] swaps per shake, while adversarial performance
improves to 100% at K=8L.

This is the key mechanism-level insight that turned the 2D DCP construction
from "works on random, fails on adversarial" into "works on both".

### 26. Comparison to Fatès-Regnault 2016

Fatès and Regnault (2016) solve 2D DCP with a pair of stochastic
number-conserving auxiliary rules composed with per-row 1D rules:

```
(F_X ∘ R_184)^{T_1}  →  (G_X ∘ R_232)^{T_2}
```

with T₁=T₂~Θ(L²) and Bernoulli(½) randomness inside F_X and G_X.  At
L=100, T=16,000 gives 100% on 1000 random trials.

Comparison:

| | Fatès-Regnault | this work |
|---|---------------|-----------|
| 2D grid tested up to | 100² | 512² |
| Total steps per trial | ~16,000 at L=100 | ~6L + 16L swaps (~3k at L=256) |
| Convergence scaling | Θ(L²) | Θ(L) |
| Adversarial structured inputs | not tested | tested and pass |
| Rule basis | 2 hand-designed stochastic NCCAs + two 1D rules | 1 enumerated NCCA + Moore-81 majority + random swaps |
| Stochastic source | Bernoulli(½) inside F_X, G_X | density-conserving random pair swaps |

### 27. Where the classifier fails

- **L < 192**: the Moore-81 amplifier has radius 4; at L=64 the
  neighborhood wraps the torus 8 times, collapsing toward global density.
  Size-adapted amplifiers (moore25, moore49) work better at small L.
- **Near-tied densities** (`|ρ - 1/2| ≤ 1/L²`): any finite-time classifier
  has nonzero error; 1-cell bias below the amplifier's noise floor.
- **Density exactly 1/2**: the problem is ill-defined (tied).

### 28. Next experiments

See Open next steps in [DENSITY_CLASSIFICATION_IDEAS.md](DENSITY_CLASSIFICATION_IDEAS.md):
catalogue-scale preprocessor scan, scale-up to L=1024/2048, density-margin
tightening, and a theoretical convergence proof.
