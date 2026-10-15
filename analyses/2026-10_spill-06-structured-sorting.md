# Spill 06–07: Structure-Guided Sorting, and Whether It Moves the Camps

**Date:** 2026-10-15
**Scripts:** `scripts/spill_06_structured_sorting.py`, `scripts/spill_06b_pool_panels.py`,
`scripts/spill_07_sorting_simulation.py`, `scripts/spill_07b_benchmark_check.py`
**Logs:** `logs/spill_06_structured_sorting.log`, `logs/spill_06_structured_sorting_2020.log`,
`logs/spill_06b_pool_panels.log`, `logs/spill_07_sorting_simulation.log`,
`logs/spill_07b_benchmark_check.log`
**Data:** GSS 2006, 2008 and 2010 three-wave panels and the 2016–20 panel (people); GSS
cross-sections 2004–14 and 2012–18 (networks)

## Question

The spillover work (`2026-10_spill-01-homosex-pilot.md`) found that camps change along their
own belief structure, but that the change is carried less by individuals changing their minds
than by people moving between camps. This tests that mechanism directly: **do people leave a
camp when the way their beliefs fit together matches the other camp's network better than
their own?**

The question is about *arrangement*, not *position*. The standard account of partisan and
ideological sorting (Levendusky 2009; Mason 2018) is that people move to the camp whose
positions match theirs, and the cross-pressures account (Hillygus & Shields 2008) is that
people holding positions from both sides defect. Neither says anything about which beliefs go
together.

## Design

Fixed before the first run (tracker, 2026-10-15).

- **People.** Wave-1 liberals and conservatives in the three panels with political views also
  observed at wave 3: 2,284 people. 774 (34%) had left their wave-1 camp by wave 3, mostly to
  moderate.
- **Networks.** Each camp's network comes from the GSS cross-sections 2004–14 **with the panel's
  own wave-1 respondents removed**, so nobody is scored against a network their answers shaped.
  POLVIEWS and PARTYID are left out of the networks and the fit. 112 items.
- **Fit.** A person's fit to a network is Σ w_jk z_j z_k / Σ |w_jk| over the pairs of items
  they answered, with answers standardised on the cross-section. It is positive when the
  person's beliefs pair up the way the network's edges do. Flipping every answer leaves it
  unchanged, so it scores which beliefs go together, not where the person stands.
- **Predictor.** Relative fit = fit to the other camp's network minus fit to their own.
- **Model.** Logit for leaving the camp, controlling for mean fit to the two networks, issue
  position towards the other camp, cross-pressure (share of items on the other camp's side),
  ideological strength, camp, panel, age, degree and sex. HC1 errors.
- **Verdict rule.** Relative fit > 0 with p < 0.05 in the full model, for at least one
  estimator.

## Results

| Estimator | Camp only: OR per SD (p) | Full model: OR per SD (p) | Permutation p |
|---|---|---|---|
| Partial correlations | 1.60 (< 0.001) | **1.14 (0.036)** | 0.020 |
| Raw correlations | 1.73 (< 0.001) | **1.25 (0.001)** | 0.010 |

**Supported through both estimators.** People whose belief pattern fits the other camp's
network are more likely to leave their own camp, over and above their issue positions, their
cross-pressure and the strength of their identification. The permutation null — networks
estimated on shuffled camp labels, so they differ only by noise — confirms the effect does not
come from the fit measure itself.

The other predictors behave as the sorting literature expects. Issue position towards the
other camp is the strongest (OR 2.41 per SD), and strong identification protects against
leaving (OR 0.68). Cross-pressure adds nothing once position is in the model. Relative fit
correlates +0.36 with position, so the two are related but distinct.

The effect is the same size in both camps (partial OR 1.16 for liberals and 1.15 for
conservatives; raw 1.26 and 1.27), and each half is less significant on its own, as expected
with half the sample.

## Replication: the 2016–20 panel

Specified before running (tracker, 2026-10-15): the same model and verdict rule on the 2016 and
2018 respondents re-interviewed in 2020, with networks from the 2012–18 cross-sections (these
respondents removed) and two extra controls, origin year and web vs phone in 2020. The 2020
wave had no in-person interviews, which adds noise to apparent switching.

| | 2006–14 panels | 2016–20 panel |
|---|---|---|
| People (left camp) | 2,284 (774) | 1,115 (323) |
| Raw: OR per SD (p) | 1.245 (0.001) | 1.244 (0.072) |
| Partial: OR per SD (p) | 1.135 (0.036) | 0.965 (0.74) |
| Issue position: OR per SD | 2.41 | 4.06 |

**By the pre-set rule the replication fails.** But the two estimators part company. The raw
estimate is the same to three decimal places and misses significance only because the sample
is half the size (permutation p = 0.055). The partial estimate does not replicate at all.
Position-based sorting is much stronger in the later panel, and relative fit correlates more
with position there (+0.47 on raw correlations, against +0.38), which leaves less independent
variation for arrangement to explain.

**Pooled across eras (exploratory, `spill_06b_pool_panels.py`):**

| Estimator | Pooled OR per SD [95% CI] | p | Difference between eras |
|---|---|---|---|
| Raw | 1.245 [1.111, 1.394] | 0.0002 | p = 0.998 |
| Partial | 1.091 [0.984, 1.209] | 0.097 | p = 0.18 |

## Reading

This is the first result in the downstream-effects work that survives a design where sorting
cannot fake it — because here sorting is the outcome. Across two panel eras it holds for fit
measured on **raw correlations** — what co-occurs in each camp — with an identical effect size,
and it does not hold reliably for fit measured on **partial correlations**, the camp's direct
links. So the claim it supports is about the camp's pattern of co-occurrence, not its network of
direct associations. That is also the estimator the project's own framework assigns to claims
about effects in the world (P1.2: raw correlations for what co-occurs, partial for direct
structure) — but that alignment was noticed after the results, and the partial failure should be
reported as a result in its own right. It says that **the arrangement of a camp's beliefs selects
its members**: people stay where their beliefs fit together the way the
camp's do, and leave when they fit another camp's wiring better, whatever their positions.
That is a population-level consequence of belief structure, and it is the mechanism the
spillover decomposition pointed to: camps shift along their own structure because of who joins
and leaves them.

It also gives specification leg 2 — arrangement, not magnitude — an empirical form at the level
of persons. Position-based sorting is well documented; arrangement-based sorting, as far as the
literature search found, is not.

## Simulation: does arrangement-based sorting move the camps?

`scripts/spill_07_sorting_simulation.py`, specified before running. If camps changed *only*
through who joins and leaves, with every person's beliefs frozen at wave 1, would they move the
way the cross-sections show — and how much of that is sorting by arrangement rather than by
position? Using the 2006–14 panels, the exit model above and entry models for wave-1
moderates, camp shifts were computed under four scenarios: the *actual* moves, the *full*
model, a *position-only* model with the relative-fit terms set to zero, and the difference
between those two, the **arrangement contribution**. Each was compared with the observed
cross-sectional camp change over the same windows, and checked for whether it moves each camp
along its own network more than along the other camp's (own minus swapped network lag).

| Networks | Scenario | Corr. with observed 4-year change [95%] | Own − swapped lag [95%] |
|---|---|---|---|
| partial | actual | −0.07 [−0.20, +0.11] | −0.05 [−0.07, +0.03] |
| partial | full | +0.05 [−0.07, +0.14] | −0.00 [−0.01, +0.01] |
| partial | arrangement | +0.07 [−0.17, +0.22] | **+0.06 [+0.01, +0.10]** |
| raw | arrangement | +0.08 [−0.05, +0.19] | +0.06 [−0.00, +0.14] |

**By the pre-set rule, not supported:** no scenario matches the observed four-year change. Two
things survive:

- **Arrangement-based sorting moves each camp along its own structure.** All sorting moves the
  camps along the structure they share (own and swapped lags both about 0.9, which is partly
  mechanical, since a composition shift follows the covariance of the beliefs). Only the
  arrangement part favours each camp's own network, clearly on partial correlations and just
  short on raw.
- **The four-year benchmark could not have worked** (`spill_07b_benchmark_check.py`,
  exploratory). Camp-differenced change in two windows with no shared respondents correlates at
  only +0.17, and two windows sharing the 2010 sample correlate at −0.57 — the signature of
  sampling noise, since 2010 enters one as an end point and the other as a start. Even the
  actual sorting does not match it.

Against a benchmark with real signal — the 27-year within-cohort camp change from spill_01's
condition B — the simulated sorting points the way the camps actually travelled:

| Networks | Full | Position-only | Arrangement | Arrangement beyond position (partial r) [95%] |
|---|---|---|---|---|
| partial | +0.49 | +0.47 | +0.54 | +0.34 [−0.04, +0.57] |
| raw | +0.48 | +0.45 | +0.49 | +0.26 [−0.04, +0.55] |

But the arrangement shift mostly points where position sorting points (r ≈ 0.7 between them).
What it adds beyond position leans the same way as the long-run change, with intervals that
just include zero. And the comparison sets four-year sorting against three decades of change,
so it speaks only to direction.

**Net:** the end-to-end link — arrangement-based sorting producing the camp-level co-movement —
is **consistent with the evidence but not demonstrated.** The person-level result (spill_06)
stands on its own; the simulation shows it moves camps along their own structure and in the
direction they have travelled, without isolating how much of the observed change it explains.

## Limits

- **Modest effect.** One SD of relative fit raises the odds of leaving by 14–25%, against 141%
  for issue position.
- **Most switches are to moderate.** The outcome is leaving a camp, not crossing to the other
  one; full crossings are too rare to model separately.
- **Measurement error in POLVIEWS** produces some apparent switching; ideological strength is
  controlled, which absorbs most of the boundary effect, but not all.
- **The pre-specified replication failed its rule.** The raw effect replicates in size and pools
  to p = 0.0002, but that pooling was not pre-specified.
- **Raw versus partial.** The raw estimate is larger. Raw correlations carry the camps' whole
  correlated blocks, which is legitimate for a claim about fit to what co-occurs in a camp, but
  the partial estimate is the cleaner test of fit to direct links.

## Next

- A sharper end-to-end test needs a benchmark with signal over the panel windows: the
  long-run cross-sections give direction only. Pooling more panel waves (wave 2 transitions,
  the 2016–20 panel) would give the simulation itself more precision.
- Robustness: switching between waves 1 and 2; full crossings only; panel weights.
- Position this against Levendusky (2009), Mason (2018) and Hillygus & Shields (2008).
