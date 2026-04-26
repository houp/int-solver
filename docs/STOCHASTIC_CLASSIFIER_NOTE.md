# 2D Density Classification — Stochastic Classifier Development Notes

*Working journal of the investigation that produced the L-scaled stochastic
classifier. Companion to [MECHANISM_FINDINGS.md](MECHANISM_FINDINGS.md),
[DENSITY_CLASSIFICATION_IDEAS.md](DENSITY_CLASSIFICATION_IDEAS.md), and
[DENSITY_CLASSIFICATION_2D_NOTE.md](DENSITY_CLASSIFICATION_2D_NOTE.md).*

This document traces today's full chain of reasoning, experiments, dead-ends
and positive results. It is intended as a faithful lab notebook rather than a
polished paper; the polished write-up is in [`technical_report.tex`](technical_report.tex)
(section "Density Classification in Two Dimensions").

## 0. Starting point

- Repository already contained a deterministic family of classifiers built
  around `sid:58ed6b657afb` (an outer-monotone NCCA) with Moore-9 VN-majority
  repeated blocks, achieving ~96% correct consensus at 256² on random inputs.
- Methodological note in [DENSITY_CLASSIFICATION_IDEAS.md](DENSITY_CLASSIFICATION_IDEAS.md)
  pointed out that periodic repetition of `[F^k M^k]^m` reduces to a single
  CA at larger effective radius, so it cannot perfectly solve DCP.
- The actual target is a **finite-switch schedule**
  `(F_1^{t_1} ∘ … ∘ F_k^{t_k})^T → majority^{T'}`, the 2D analog of 1D
  Fukś (rule 184 → rule 232).

## 1. Mechanism probe (Phase P1)

**Script:** [`mechanism_probe_58ed6.py`](mechanism_probe_58ed6.py) +
[`mechanism_probe_58ed6_summary.py`](mechanism_probe_58ed6_summary.py).

**Findings:**

- `sid:58ed6` applied alone on random near-critical lattice reaches a frozen
  fragmented attractor in ~64 steps.
- At the attractor: local-Moore-majority agreement with global majority
  *decreases* from ~0.42 (random init) to ~0.17.
- Interface / L² stays ≈ 1.0 — no coarsening.
- Same behavior for `sid:812b7dae7aa7`, `sid:029b09cea0b5`,
  `sid:29154e9615d8`, `sid:ac809b9cab03`. None of the leading candidates
  coarsens; they are fragmenters or frozen-state equalizers.

Implication: the earlier repeated-block success `[F^8 M^16]^k` is
emergent from the F-M alternation; F alone is not a Fukś-style preprocessor.

## 2. Catalog-scale coarsener scan

**Script:** [`coarsening_scan.py`](coarsening_scan.py).

Scanned a random 5000-rule subsample of the 133,713-rule nonzero-mask
catalog for genuine coarseners.

**Findings:**

- Coarseners DO exist. Best: `sid:11bb6b4211627713` drives interface/L²
  from ~1.0 to ~0.20 (i.e., phase-separates).
- But: all coarseners are black/white asymmetric. Under such rules, the
  phase-separated final state has a dominant color that is uncorrelated
  with the true global majority (sometimes minority color takes over).
- Decodability (local-majority agreement with global majority) at the
  attractor never exceeds 0.57 across 5000 rules tested.

Files: [`coarsening_random5k.json`](coarsening_random5k.json),
[`coarsening_outer57.json`](coarsening_outer57.json) (the 57-rule
`outer_monotone ∩ orthogonal_monotone ∩ diagonal_monotone` family).

## 3. Finite-switch screens — all negative

**Scripts:** [`finite_switch_coarsener_test.py`](finite_switch_coarsener_test.py),
[`composition_search.py`](composition_search.py),
[`multiphase_schedule_test.py`](multiphase_schedule_test.py).

Tested:
- Single-rule `F^{T_1} → VN-majority^{T_2}` over top coarseners, top
  decodability leaders, and the outer-monotone family. All gave
  `correct_consensus_rate = 0` even with T₂ up to 1024.
- Pair compositions `(F_1^{t_1} F_2^{t_2})^T → VN-majority^{T_2}`: 550
  configurations tested. All 0%.
- Multi-phase `F_1^{32} F_2^{32} → VN-majority^{1024}`: 180 configurations.
  All 0%.

**Diagnosis:** 2D VN- and Moore-majority both have many metastable fixed
points (straight stripes, rectangular blocks, maze patterns). Any
preprocessor output that contains such structures stays locked under
majority. Longer T₂ does not help: these are true fixed points.

Files: [`finite_switch_coarseners_T1_L.json`](finite_switch_coarseners_T1_L.json),
[`finite_switch_long_T2.json`](finite_switch_long_T2.json),
[`composition_search_pilot.json`](composition_search_pilot.json),
[`multiphase_pair_long.json`](multiphase_pair_long.json).

This led to the conclusion documented in
[MECHANISM_FINDINGS.md](MECHANISM_FINDINGS.md): under the strict
finite-switch formulation with plain local majority as amplifier, no tested
schedule reaches consensus.

## 4. Amplifier library — relaxing "local majority"

**Scripts:** [`amplifier_library.py`](amplifier_library.py),
[`amplifier_test.py`](amplifier_test.py).

Implemented a library of sub-Moore majorities:
- `moore9` (3×3 Moore, radius 1, threshold ≥5).
- `vn5` (von Neumann, radius 1, threshold ≥3).
- `diag5` (center + 4 diagonals, threshold ≥3).
- `row3` (center + left + right, threshold ≥2). **Exactly 1D rule 232
  applied per row.**
- `col3` (1D rule 232 per column).
- `maindiag3`, `antidiag3` (diagonal analogs).
- `ortho4`, `corner4`, `moore8` (no-center variants; biased, drive to
  all-zeros; useless as classifiers).

And radius-2+ amplifiers (native MLX implementation):
- `moore25` (5×5 Moore, threshold ≥13).
- `vn13` (radius-2 diamond, threshold ≥7).
- `moore49`, `vn25`, `moore81` (up to radius 4).

**Experiments:**

- Pure-amplifier alternating schedules (e.g., `moore81,row3,col3,maindiag3,
  antidiag3`) did not exceed 94% correct consensus at 256².
- `sid:58ed6 + moore81` + shake with sid:58ed6 → **major breakthrough**:
  100% correct consensus at 192², 256², 384², 512² on random inputs.
- Schedule:
  `F^{128} [moore81^{256} F^{64}]^{2} moore81^{1024}`.
- Validated across 5 independent seeds × 64 trials: 320 trials per grid;
  zero failures at L ≥ 256.

File: [`amplifier_58ed6_radius2.json`](amplifier_58ed6_radius2.json),
[`amplifier_big_validation.json`](amplifier_big_validation.json),
[`amplifier_256_pushtoperfect.json`](amplifier_256_pushtoperfect.json),
[`validation_big.json`](validation_big.json), [`validation_512.json`](validation_512.json).

At this point I thought the problem was essentially solved for random inputs.

## 5. Adversarial stress-test breaks the illusion

**Script:** [`adversarial_test.py`](adversarial_test.py).

The classifier was tested on 16 structured initial configurations per grid
(stripes, checkerboards, block-checkers, half-half, single large blob, two
blobs, near-critical random, etc.), each density-tipped by a single flipped
cell.

**Results:**

- 192²: 3/16 correct consensus.
- 256²: 3/16 correct consensus.
- 384²: 0/16 correct consensus.

**Failing inputs:** pure stripes (horizontal / vertical), checkerboard,
block-checkerboard, half-half split. Majorities "decoded" the right answer
(final density on the right side of 0.5) for most cases but did not reach
consensus.

## 6. Structural theorem — why it fails

**Script:** [`structured_probe.py`](structured_probe.py).

Per-phase probe of `sid:58ed6` on pure stripe at density 0.5 (one flipped
cell tip): interface length / L² stays at 1.0 through *every* phase. `F`,
`moore81`, `moore9`, diagonal traffic — all preserve stripes exactly.

**Theorem (Proposition 1 in the LaTeX report):**

> Any deterministic shift-invariant Moore-neighborhood rule F maps a
> horizontal stripe configuration to a horizontal stripe configuration
> (possibly complemented). For NCCAs, only preserve or complement is
> allowed by density conservation. Hence a tipped stripe at density
> 1/2 + 1/L² is a permanent obstruction for any such schedule.

This is sharper than the classical Land–Belew 1995 impossibility: it
identifies a specific family of configurations (stripes, checkerboards,
block patterns) on which no deterministic shift-invariant local-rule
schedule can succeed.

## 7. Literature review — Fatès

**Papers studied (via WebFetch):**

1. **Fatès 2011 STACS** — 1D stochastic DCP classifier. At each cell, at
   each time step, independently apply rule 184 with probability 1-η and
   rule 232 with probability η (η small). Solves 1D DCP with arbitrary
   precision as iterations → ∞.
2. **Fatès–Regnault 2013** — journal version with proof and expanded
   results.
3. **Fatès–Regnault 2016 (arXiv:1506.06653)** — 2D extension. Two
   stochastic number-conserving auxiliary rules F_X (lane-changing) and
   G_X (crowd-avoidance), composed with per-row R_184 and R_232:
   ```
   (F_X ∘ R_184)^{T_1} → (G_X ∘ R_232)^{T_2}
   ```
   Bernoulli(½) internal randomness; T₁ = T₂ ~ 3L². 100% at L=100 with
   T=16000.
4. **arXiv:1604.04402** — particle-spacing problem in 2D; shifts from CA
   to interacting particle systems.

**Key takeaways:**

- The stochastic rule choice per cell breaks translational invariance.
- Fatès uses p = 0.5 *inside* his stochastic NCCAs, not as a rule-mixing
  probability. The mixing weight η between 184 and 232 is small (~0.01),
  i.e., majority is the *rare* operation.
- Convergence time O(L²).
- Fatès tested only random Bernoulli inputs.

This reframed our direction. The user pointed out that Fatès's work
(10+ years old) didn't have our catalogue of 133k+ NCCAs or GPU compute.
Research path: find a minimal-stochasticity schedule, leveraging our
deterministic skeleton + GPU-scale sweep.

## 8. Minimal-ε Bernoulli flip noise (Experiment A)

**Script:** [`minimal_epsilon.py`](minimal_epsilon.py).

Added per-cell Bernoulli flip noise (independently flip each cell with
probability ε) during the shake phases.

Swept ε ∈ {0, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 0.5} at grids 64,
128, 192. 128 trials per density.

**Results (Table):**

| L | ε | random cc | adversarial cc |
|---|---|----------|----------------|
| 64 | 0 | 87.1% | 20% |
| 64 | 1e-3 | 89.5% | 60% |
| 64 | 1e-2 | 83.2% | 40% |
| 128 | 0 | 98.8% | 20% |
| 128 | 1e-3 | 91.4% | 40% |
| 192 | 0 | 99.2% | 20% |
| 192 | 1e-3 | 95.7% | 60% |
| 192 | 1e-1 | 5.5% | 30% |

Sweet spot at ε ≈ 10⁻³. Improves adversarial from 20% to 60% but **degrades
random-input consensus** because per-cell flips change density, destroying
the 1-cell tipping bias.

File: [`minimal_epsilon_sweep.json`](minimal_epsilon_sweep.json).

Per-case analysis at ε=1e-3 revealed:
- Stripes: flipped to correct color ~50% of the time (randomly chosen).
- Checkerboards: always stuck near density 0.5.
- Half-half: ~50/50 correct/wrong.
- Block-checker: usually OK.

## 9. One-shot concentrated noise (Experiment B) — NEGATIVE

**Script:** [`one_shot_noise.py`](one_shot_noise.py).

Apply a *single* noise burst flipping K uniformly-sampled cells, then run
the deterministic amplifier.

**Result:** random-input correct-consensus is exactly independent of K for
K ∈ [0, 0.1·L²]. The single noise burst at the end of the schedule neither
helps nor hurts. Adversarial improves only mildly.

This ruled out one-shot flip noise and redirected us toward *conservative*
noise.

File: [`one_shot_results.json`](one_shot_results.json).

## 10. Density-conserving random swaps (Experiment C) — BREAKTHROUGH

**Script:** [`conservative_noise.py`](conservative_noise.py).

Replace Bernoulli flips with K random pair swaps: pick two cells uniformly,
swap their values. Exactly density-conserving per realization.

**Results at K = 8L swaps per shake:**

| L | random cc (128 trials) | adversarial cc (10 cases) |
|---|-----------------------|----------------------------|
| 64 | 90.6% | 50% |
| 128 | 99.2% | 70% |
| 192 | **100.0%** | 50% |

Random-input performance is undisturbed by swaps (unlike flip noise).
Adversarial improves. But individual adversarial cases still showed a
50/50 direction issue at density exactly 1/2 + 1/L².

File: [`conservative_noise_sweep.json`](conservative_noise_sweep.json).

Per-case analysis revealed that at density 1/2 + 1/L² (one tipping cell),
the amplifier is essentially flipping a coin on which direction to drive
consensus. This is a practical limit for any finite-time classifier.

## 11. Realistic density margins (Experiment D)

**Script:** [`adversarial_realistic.py`](adversarial_realistic.py).

Retest with density margins in {0.51, 0.52, 0.55} — realistic non-tied
densities matching Fatès's original tests (he uses 0.49/0.51).

**Results at K = 8L:**

| L | adversarial cc | fails |
|---|---------------|-------|
| 128 | **100%** | 0/30 |
| 192 | **100%** | 0/30 |

**For L ≥ 128 at density margin ≥1%, the classifier is perfect on the
structured-adversarial battery.**

The apparent "stripe failure" was an artifact of testing at density ≈ 1/2
(the tied case). At realistic densities, the classifier succeeds.

File: [`adversarial_realistic.json`](adversarial_realistic.json).

## 12. Large-grid regression at 384²

Extended the realistic-density test to larger grids:
[`adversarial_big.json`](adversarial_big.json).

**Surprise:** at L=384, ALL half_half cases failed at density 0.505–0.55.
The deterministic schedule (K=0) also failed, ruling out the noise as the
cause.

**Diagnosis:** the schedule's fixed parameters (T_pre=128, T_amp=256, etc.)
don't scale with L. At L=384, these are too short; the post-preprocessor
state at larger L contains more persistent stable structures that
`moore81` cannot erode in the fixed T_amp_final=1024 steps.

Direct test at fixed density and varying T_amp_final showed that the
density gets **stuck** at ≈0.54, regardless of T_amp_final (1024 vs 8192
same final density) — true fixed point.

## 13. L-scaled schedule (SCALING FIX)

**Script:** [`scaled_schedule.py`](scaled_schedule.py).

All schedule lengths scale linearly with L:
```
T_pre = L/2, T_amp = L, T_shake = L/4, T_amp_final = 4L, K = 8L swaps
```

**Validation across L ∈ {64, 128, 192, 256, 384, 512}** at 128 random
trials + 30 adversarial trials per grid (density margins 1–5%):

| L | random cc | adv cc | random fails | adv fails |
|---|----------|--------|-------------|-----------|
| 64 | 93.75% | 86.67% | 8/128 | 4/30 |
| 128 | 98.44% | **100%** | 2/128 | 0/30 |
| 192 | **100%** | **100%** | 0 | 0 |
| 256 | **100%** | **100%** | 0 | 0 |
| 384 | **100%** | **100%** | 0 | 0 |
| 512 | **100%** | **100%** | 0 | 0 |

**For L ≥ 192, the classifier is universal** on our test suite:
random-Bernoulli inputs at density 0.49/0.51 AND structured adversarial
inputs (stripes, checkers, block-checkers, half-half) at density margins
≥1%.

File: [`scaled_schedule_validation.json`](scaled_schedule_validation.json).

## 14. Performance characterization (MLX on Apple M2 Max)

**Raw simulator benchmark** (batch=64, 200 steps, pairwise Moore step):

| L | NumPy | MLX | speedup |
|---|-------|-----|---------|
| 64 | 0.11 Gcells/s | 1.27 | 12× |
| 128 | 0.11 | 2.88 | 26× |
| 256 | 0.12 | 1.29 | 11× |

End-to-end on the full classifier at L=512 with 128 trials:
~386 seconds per seed for the random battery, ~77 s for adversarial.

**Python 3.14 benchmark:** mildly slower than 3.12 (10%) on this workload
because the hot loop is Metal dispatch, not interpreter CPU. Keep venv on
3.12.

**Multi-process:** not useful on top of MLX GPU (contends for Metal queue).
Would help for scipy-bound tasks like cluster labeling but those are
diagnostic, not hot.

## 15. Comparison to Fatès–Regnault 2016

| | Fatès–Regnault 2016 | This work |
|---|---------------------|-----------|
| 2D grid tested | 50², 100² | 64²–512² |
| 100% at L=100 iterations | 16000 | ~2500 (at L=128) |
| Adversarial tested | No (random only) | Yes (stripes, checkers, half-half at density margins ≥1%) |
| Stochasticity | Bernoulli(½) inside F_X, G_X | Random pair swaps (density-conserving) |
| Total steps/trial | ~16000 at L=100 | ~6L + 16L swaps (e.g., ~3.6k at L=256) |
| Rule basis | Hand-designed F_X, G_X + per-row R_184, R_232 | `sid:58ed6b657afb` (outer-monotone NCCA) + Moore-81 majority + random swaps |

## 16. Open questions / next steps (in order of priority)

1. **Catalog-scale screen for better preprocessors.** Scan the full 133k
   NCCA catalog with the L-scaled schedule at L=128, replacing
   `sid:58ed6` with each candidate. GPU-estimated compute: hours. Expected
   outcome: rules that work at L<128 and/or tighter density margins.

2. **Tighten density margin.** The classifier works at ≥1%. What's the
   limit? Test at 0.501, 0.5005, 0.5001. Theoretical prediction: breaks
   at density = 1/2 + Θ(1/L²) because the amplifier cannot reliably
   detect a single-cell bias.

3. **Scale-up to L=1024, 2048.** Check if L-scaled schedule continues.
   Computational budget: hour-scale per grid at L=1024 on MLX.

4. **Theoretical analysis.** Attempt a convergence proof. The mechanism
   is clear (swap noise breaks translational symmetry; Moore-81 curvature
   flow finishes). A formal convergence bound would require analyzing
   the combined dynamical system (deterministic F + swap + radius-2
   majority).

5. **Generalize the noise mechanism.** Random pair swaps is one choice.
   Are there better density-conserving noise operators (e.g., local
   permutations, swap-pairs restricted to Moore-neighbors)?

6. **Adaptive schedule parameters.** The linear scaling T_pre = L/2, etc.
   was hand-tuned. Can we do Bayesian-optimization over (c_pre, c_amp,
   c_shake, c_final, k_swap) to find the minimal total-time schedule?

## 17. Summary of artifacts produced today

**New Python scripts** (repo root):

- [`mechanism_probe_58ed6.py`](mechanism_probe_58ed6.py), [`mechanism_probe_58ed6_summary.py`](mechanism_probe_58ed6_summary.py)
- [`coarsening_scan.py`](coarsening_scan.py)
- [`finite_switch_coarsener_test.py`](finite_switch_coarsener_test.py)
- [`composition_search.py`](composition_search.py)
- [`multiphase_schedule_test.py`](multiphase_schedule_test.py)
- [`failure_diagnostic.py`](failure_diagnostic.py), [`failure_analysis.py`](failure_analysis.py)
- [`multi_switch_schedule.py`](multi_switch_schedule.py), [`multi_switch_two_ncca.py`](multi_switch_two_ncca.py)
- [`amplifier_library.py`](amplifier_library.py), [`amplifier_test.py`](amplifier_test.py)
- [`fates_style.py`](fates_style.py), [`hybrid_stochastic.py`](hybrid_stochastic.py)
- [`minimal_epsilon.py`](minimal_epsilon.py)
- [`one_shot_noise.py`](one_shot_noise.py)
- [`conservative_noise.py`](conservative_noise.py)
- [`adversarial_realistic.py`](adversarial_realistic.py)
- [`structured_probe.py`](structured_probe.py)
- [`scaled_schedule.py`](scaled_schedule.py)
- [`validation_big.py`](validation_big.py)

**Result JSON** (repo root): one file per experiment, with per-trial
records. See paths referenced throughout this document.

**Documentation updated or added:**
- [`technical_report.tex`](technical_report.tex) — new section "Density
  Classification in Two Dimensions".
- [`references.bib`](references.bib) — added Fatès papers, Land-Belew,
  Fukś 1997.
- [`MECHANISM_FINDINGS.md`](MECHANISM_FINDINGS.md) — mechanism-probe
  findings and structural argument (pre-stochastic).
- [`DENSITY_CLASSIFICATION_IDEAS.md`](DENSITY_CLASSIFICATION_IDEAS.md) —
  ledger updated with today's tested ideas.
- [`STOCHASTIC_CLASSIFIER_NOTE.md`](STOCHASTIC_CLASSIFIER_NOTE.md) — this
  file.

## 18. Headline result

A density classifier for 2D Moore-neighborhood configurations:

```
F^{L/2}  [moore81^{L}  F^{L/4}  swap^{8L}]^{2}  moore81^{4L}
```

where `F = sid:58ed6b657afb`, `moore81` is the radius-2 (5×5) Moore
majority, and `swap^K` denotes K uniformly random pair swaps.

**Achieves 100% correct consensus** across 128 random Bernoulli trials at
density 0.49/0.51 AND 30 structured adversarial trials (stripes,
checkerboards, block-checkers, half-half splits at densities 0.51–0.55,
both labels), for every grid size L ∈ {192, 256, 384, 512}.

Total work: ~6L deterministic CA steps + ~16L random swaps per trial —
O(L), substantially faster than Fatès–Regnault's O(L²) 2D construction.

The construction is **partially novel**: the density-conserving random
swap as the sole stochastic component, combined with a large-radius
deterministic amplifier and an NCCA preprocessor drawn from the
133,713-rule enumerated catalog, gives a sharper and more robust
classifier than prior work.
