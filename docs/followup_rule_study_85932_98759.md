# Follow-Up Study of Rules 85932 and 98759

All bare numeric rule labels in this note are legacy snapshot indices from
`expanded_property_panel_nonzero.bin`. The permanent stable identifiers are:
- `85932` -> `sid:adb15d42281e`
- `98759` -> `sid:ca6705f68694`
- `24795` -> `sid:029b09cea0b5`
- `55897` -> `sid:812b7dae7aa7`
- `2370` -> `sid:f076fc6ffb58`

This note summarizes a deeper follow-up study of two additional shortlisted rules:

- `85932`
- `98759`

These were the strongest remaining second-tier candidates after the earlier analysis of:

- `24795`
- `55897`
- `2370`

Raw outputs:

- [focused_rule_study_85932_98759.json](/Users/witoldbolt/Documents/solvw/omp/focused_rule_study_85932_98759.json)
- [mechanistic_rule_study_85932_98759.json](/Users/witoldbolt/Documents/solvw/omp/mechanistic_rule_study_85932_98759.json)
- [object_atlas_study_85932_98759.json](/Users/witoldbolt/Documents/solvw/omp/object_atlas_study_85932_98759.json)

## Main Conclusion

Neither rule rises to the level of the first-tier candidates.

- `85932` is mildly nontrivial, but still strongly transport-dominated.
- `98759` is even simpler and looks close to a rigid diagonal conveyor with a few localized
  scattering motifs.

Both rules are therefore weaker candidates for genuinely interesting dynamics than:

1. `24795`
2. `55897`
3. `2370`

## Rule 85932

Properties:

- `diagonal_monotone`
- isolated particle velocity: `south`
- isolated hole velocity: `northwest`

### Focused dynamics

On the larger `256 x 256`, `512`-step runs:

- low- and medium-density random states remain almost perfectly ballistic
- even at density `0.30`, late patch damage is only `0.0004`
- at density `0.50`, late patch damage rises only to `0.0076`
- dominant late-time drift is always `south`

Structured seeds:

- checkerboard: exact period `2`
- vertical interface: exact period `2`
- horizontal stripes: exact period `2`
- diagonal stripes: exact period `2`

So the large-scale dynamics are simple periodic transport, not complex irregular behavior.

### Mechanistic probes

- particle pair collisions: `1 / 12`
- hole pair collisions: `2 / 12`
- interface width remains `0` in all tested large-scale seeds

This means the rule does have some localized collision response, but only a narrow one.

### Object atlas

Connected motifs up to size `4`:

- `free_transporter`: `55`
- `separated_carriers`: `92`
- `localized_scatter`: `95`
- `phase_shift_scatter`: `28`
- `compact_bound_state`: `0`
- `compact_oscillator`: `0`

Three-carrier atlas:

- `free_transporter`: `38`
- `separated_carriers`: `30`
- `localized_scatter`: `4`
- `compact_*`: `0`

Interpretation:

`85932` is a legitimate scattering rule, but only a mild one.  The most common nontrivial
outcome is that compact motifs simply resolve into separated carriers.

## Rule 98759

Properties:

- `diagonal_monotone`
- isolated particle velocity: `northeast`
- isolated hole velocity: `southwest`

### Focused dynamics

This rule turns out to be much simpler than the earlier shortlist score suggested.

On the larger `256 x 256`, `512`-step runs:

- random densities `0.05`, `0.12`, and `0.30` show almost no damage growth
- even at density `0.50`, late patch damage is only `0.0025`
- dominant late-time drift is always `northeast`

Structured seeds:

- checkerboard: exact fixed point
- diagonal stripes: exact fixed point
- vertical interface: exact period `2`
- horizontal stripes: exact period `2`

So the earlier hint of “interestingness” came largely from structured-seed sensitivity in the
coarser screening, not from rich long-horizon dynamics.

### Mechanistic probes

- particle pair collisions: `0 / 12`
- hole pair collisions: `0 / 12`
- interface width remains `0`

This is the clearest signal that `98759` is not a serious complexity candidate.

### Object atlas

Connected motifs up to size `4`:

- `free_transporter`: `80`
- `separated_carriers`: `22`
- `localized_scatter`: `156`
- `phase_shift_scatter`: `12`
- `compact_bound_state`: `0`
- `compact_oscillator`: `0`

Three-carrier atlas:

- `free_transporter`: `64`
- `separated_carriers`: `5`
- `localized_scatter`: `3`
- `compact_*`: `0`

Interpretation:

Despite many motifs being classified as `localized_scatter`, the rule has:

- no detected pair-collision events in the representative probe set,
- no compact objects,
- and very weak long-horizon damage spreading.

So `98759` is best described as a very regular transport rule with some small motif-dependent
deformations, not as a genuinely rich scattering system.

## Why The Earlier Shortlist Overrated Them

The earlier broad shortlist score mixed together:

- activity,
- coarse damage metrics,
- and sensitivity on structured seeds.

That is useful for funneling, but it can overvalue rules that:

- react strongly to a checkerboard or interface,
- yet remain almost perfectly ballistic under longer random-state runs,
- and fail to produce any compact interacting objects.

That is exactly what happened here, especially for `98759`.

## Updated Ranking

The current best-known ordering is:

1. `24795`
2. `55897`
3. `2370`
4. `85932`
5. `98759`

with a clear drop after the first three.

## Practical Takeaway

The two new rules were worth checking, but the deeper analysis is mostly negative.

- `85932` remains a mild second-tier scattering rule.
- `98759` can probably be retired from the interesting-candidate pool.

If we continue looking for more promising NCCAs, it is probably better to move to other parts of
the catalog than to keep drilling deeper into these two.
