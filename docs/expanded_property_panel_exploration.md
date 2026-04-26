# Expanded Nonzero-Mask Exploration

This note summarizes a first-pass dynamic screening of the updated nonzero-mask catalog obtained
from the current 35-property panel used in the manuscript.

## Updated Catalog Size

- Base family: number-conserving binary Moore-neighborhood rules
- Base count: `147,309,433`
- Additional property families in the current panel: `35`
- Rules with nonzero property mask in the expanded panel: `133,713`

This replaces the older `7,217` count, which only corresponded to the earlier 19-property panel.

Artifacts:

- binary catalog: [expanded_property_panel_nonzero.bin](/Users/witoldbolt/Documents/solvw/omp/expanded_property_panel_nonzero.bin)
- metadata: [expanded_property_panel_nonzero.json](/Users/witoldbolt/Documents/solvw/omp/expanded_property_panel_nonzero.json)
- first-pass dynamic ranking: [expanded_property_panel_exploration.json](/Users/witoldbolt/Documents/solvw/omp/expanded_property_panel_exploration.json)

## Screening Method

The first-pass ranking was intentionally simple:

1. exclude analytically trivial rules detected exactly by LUT structure:
   - identity,
   - rigid shifts,
   - embedded von Neumann non-shift traffic rules,
   - embedded diagonal-von-Neumann non-shift traffic rules;
2. simulate every remaining rule on a `32 x 32` periodic grid for `48` steps with the MLX backend;
3. use two initial-condition ensembles:
   - random density `p = 0.5`,
   - sparse density `p = 0.12`;
4. score each rule by a weighted combination of:
   - late-time activity on the random seed,
   - late-time activity on the sparse seed,
   - final binary entropy on the random seed,
   - final binary entropy on the sparse seed.

This is only a coarse “interestingness” screen. It favors sustained motion and disorder, not
necessarily genuine computational complexity.

## First Conclusions

The strongest conclusion from this first pass is negative:

- the expanded nonzero-mask catalog is still dominated by transport-like rules rather than obvious
  Life-like local birth/death phenomena.

The strongest positive conclusion is:

- there do exist nontrivial high-activity rules in the catalog that are not just identity, rigid
  shifts, or the already-understood embedded one-dimensional families.

However, the top-ranked rules are not spread across many property families. They are concentrated in:

- `diagonal_monotone`
- `orthogonal_monotone`
- one mixed case in `outer_monotone + orthogonal_monotone + diagonal_monotone`

In particular, the top `30` rules from this screen contain:

- `24` rules from `diagonal_monotone`
- `6` rules from `orthogonal_monotone`
- `1` rule from the mixed outer/orthogonal/diagonal monotone intersection

This strongly suggests that the current score is mostly discovering anisotropic transport rules
with persistent motion and collisions, not symmetry-rich or totalistic-like rules.

## Interpreting The Top Rules

The top candidates almost all have clear isolated-particle / isolated-hole drift:

- southwest vs northeast,
- south vs north,
- east vs west,
- southeast vs northwest,
- northeast vs southwest.

So the current high scorers look like directional interacting-particle systems. That is still
interesting, but it is a different kind of interestingness than Conway-Life-style localized
birth/survival logic.

## Damage-Spreading Check

A second check was run on the top `30` ranked rules:

- `64 x 64` grid
- random density `p = 0.30`
- compare one trajectory against a one-bit perturbation for `64` steps

Most top-ranked rules had very small late-time Hamming separation, typically around
`0.0002` to `0.003`. That is consistent with stable ballistic transport.

A few candidates stood out with noticeably larger perturbation growth:

- catalog id `24795`
- catalog id `2370`
- catalog id `55897`

All three belong to `diagonal_monotone`, and their late-time damage levels were much larger than
the rest of the top-ranked set. They are therefore the first concrete candidates worth a deeper
manual visual inspection.

## Bottom-Line Answer

Do we see any interesting rules in the expanded nonzero-mask catalog?

- Yes, in the weak exploratory sense: there are nontrivial transport-and-collision rules beyond the
  analytically trivial families, and a few of them do amplify perturbations noticeably.
- Not yet in the strong Life-like sense: this first pass does not show clear evidence of
  localized, self-maintaining, birth/death-style complexity.

The current best candidates for deeper study are the few diagonal-monotone rules with both:

- very high sustained activity,
- and non-negligible damage spreading.
