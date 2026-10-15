# Spill 01–05: HOMOSEX Spillover Pilot, Replications, Sorting Control, Panel Test

**Date:** 2026-10-15
**Script:** `scripts/spill_01_homosex_pilot.py`
**Logs:** `logs/spill_01_homosex_pilot.log`, `logs/spill_01_grass_pilot.log`,
`logs/spill_01_cappun_1993-1996_2006-2010_pilot.log`, `logs/spill_01_cappun_pilot.log`,
`logs/spill_02_focal_embeddedness.log`, `logs/spill_03_fuzziness_benchmark.log`,
`logs/spill_04_panel_test.log`, `logs/spill_04_panel_test_mean13.log`,
`logs/spill_05_change_decomposition.log`
**Figures:** `figures/spill_01_homosex_pilot.png`, `figures/spill_01_grass_pilot.png`
**Data:** GSS, base window 1987–91, late window 2014–18; 300 permutations and 300 bootstraps per condition

## Question

Does a group's own belief network predict how a shift in one belief spreads to its other
beliefs? This is the pilot for the lead paper's downstream-effects claim. Between 1987–91
and 2014–18 both liberals and conservatives liberalised sharply on HOMOSEX (by 0.87 and 0.48
pooled SDs within the cohorts present in both windows). If population belief structure shapes
belief change, each camp's other beliefs should move in proportion to their baseline edge
with HOMOSEX in that camp's network, and not in proportion to the other camp's edges.

## Design

The two groups are differenced, so that any item-level drift common to both cancels:

    D_g1,j − D_g2,j = β (w_g1,j · D_g1,H − w_g2,j · D_g2,H) + a + e_j

where `D_gj` is group g's change in the mean of item j (in pooled baseline SDs) and `w_gj` is
the baseline edge between HOMOSEX and j in group g's network. A positive β means movement
follows each group's own arrangement beyond what the groups share. `r_own` is the correlation
across items between the two sides of that equation; `r_swap` is the same with each group's
edges replaced by the other group's.

Outcomes exclude POLVIEWS and PARTYID, and the four other homosexuality items (MARHOMO,
SPKHOMO, COLHOMO, LIBHOMO), since spillover means movement in other beliefs. Both estimators
are run: regularized partial correlations (graphical LASSO, α = 0.2) and raw Pearson
correlations on pairs with at least 100 joint observations.

| Condition | Groups | Sample | Threat addressed |
|---|---|---|---|
| A | liberal vs conservative | all respondents | — |
| B | liberal vs conservative | born 1925–69 | cohort replacement |
| C | liberal vs conservative | B, late window reweighted to baseline composition | other composition change |
| D | college vs no degree | US-born, born 1925–65, reweighted | ideological sorting, immigration |

Condition D exists because A–C cannot rule out **sorting**. Self-identified liberals in
2014–18 are not the kind of people who called themselves liberal in 1987–91, and a change in a
camp's membership moves the mean of every item in proportion to its covariance with the
dimension along which membership changed. People cannot re-sort into education groups on
their attitudes.

The go criterion was fixed before the first run: in condition B, β > 0 with permutation
p < 0.05 for at least one estimator, and `r_own > r_swap` for that estimator. The exclusion of
the homosexuality items, condition D, the US-born restriction and the joint-observation floor
were added after a 12-replicate smoke run to close threats it exposed. The criterion was not
changed. A later fallback for ill-conditioned graphical LASSO fits (refitting on the nearest
positive definite correlation matrix) was added when the GRASS run failed on one network; both
items were rerun with it, and the HOMOSEX numbers moved by at most 0.001 in p.

## Results

| Cond. | Estimator | β | p (perm) | r_own | ρ_own (Spearman) | r_swap | p (own − swap) | Focal rank |
|---|---|---|---|---|---|---|---|---|
| A | partial | +4.21 | 0.003 | +0.39 | +0.46 | −0.29 | 0.007 | |
| A | raw | +0.94 | 0.003 | +0.56 | +0.57 | −0.18 | 0.003 | |
| B | partial | +4.79 | 0.003 | +0.36 | +0.46 | −0.28 | 0.007 | 1 of 47 |
| B | raw | +1.27 | 0.003 | +0.59 | +0.60 | −0.05 | 0.007 | 8 of 53 |
| C | partial | +3.70 | 0.003 | +0.28 | +0.38 | −0.23 | 0.023 | |
| C | raw | +1.17 | 0.007 | +0.49 | +0.51 | −0.24 | 0.003 | |
| D | partial | +1.74 | 0.013 | +0.24 | +0.28 | −0.23 | 0.013 | 2 of 40 |
| D | raw | +0.36 | 0.146 | +0.28 | +0.31 | −0.25 | 0.020 | 5 of 48 |

p = 0.003 is the floor at 300 permutations. "Focal rank" places HOMOSEX among every item that
shifted at least 0.1 SD in both groups, each used in turn as the focal item.

**The go criterion is met, through both estimators.**

1. **Each group's own network predicts its own movement, and the other group's network
   predicts it worse — in fact with the wrong sign.** `r_own` is positive and `r_swap`
   negative in every condition and estimator. The own-minus-swap gap clears the permutation
   null in all eight cells.
2. **It survives the cohort and composition controls.** B and C barely change the picture.
3. **It survives the sorting-proof contrast, at reduced size.** In D the partial-correlation
   version holds (β p = 0.013, gap p = 0.013); the raw β does not (p = 0.146), although its
   own-minus-swap gap does (p = 0.020).
4. **HOMOSEX is special, not just any item.** Using every other item that shifted as the
   focal item, HOMOSEX ranks first of 47 (partial, B) and second of 40 (partial, D). In D the
   median item scores close to zero (r_own +0.02 partial, +0.06 raw), so the effect is
   specific to the item that actually moved most.
5. **Not driven by a few items or by near-duplicates.** Spearman correlations are as large as
   Pearson, and adding the homosexuality items back barely changes `r_own`.

## What the pilot does not show

1. **The lib/con raw result is partly non-specific.** In B the median item used as focal
   scores r_own = +0.45 on raw correlations, close to HOMOSEX's +0.59. Something moves the
   whole correlated block differently in the two camps — sorting is the obvious candidate. The
   exploratory network-lag measure tells the same story: strong in A–C (r ≈ 0.6–0.7), null in
   D (p = 0.35 and 0.43). The HOMOSEX-specific effect survives D; the generic "every item
   follows its neighbours" pattern does not. **For the lib/con claim, sorting is the threat to
   answer next.**
2. **The strong form cannot be tested with HOMOSEX.** Only one item per contrast has a raw
   edge with HOMOSEX of opposite sign in the two groups (|r| ≥ 0.05 in both). In B it is
   POLMURDR, where the prediction holds for conservatives only. In D it is EQWLTH, where it
   holds for both: as HOMOSEX disapproval fell, college graduates moved towards
   redistribution and the non-college moved away from it, as their opposite-signed edges
   predict. That is suggestive of the educational realignment, but it is one item and the
   non-college edge is weak (r = −0.06). Opposite-direction movement needs a focal item, or a
   design, where sign flips are common.
3. **β mixes spillover and common causes.** A positive β means change follows each group's
   own arrangement. It does not say whether HOMOSEX caused the neighbours to move or a common
   driver moved items in proportion to their association with HOMOSEX. For the paper's
   population-level claim — two populations with different arrangements respond differently
   to the same shift — the distinction is secondary, but the prose must not say "caused".
4. **One focal item, one pair of windows.** Nothing here yet speaks to the ~2008 break.
5. **The bootstrap CIs understate uncertainty in the wrong direction.** Resampling adds noise
   to the estimated edges, which attenuates β, so several observed βs sit near the top of
   their bootstrap intervals. The permutation test is the primary inference.

## Replication: GRASS

Specified before running: identical design, windows, conditions and go criterion, with
marijuana legalisation as the focal item. Both camps bottom out on GRASS in 1987–91 and move
towards legalisation by 2014–18 (within-cohort shifts +0.99 liberal, +0.63 conservative). The
item set has no other marijuana items, so nothing is excluded as same-object.

| Cond. | Estimator | β | p (perm) | r_own | r_swap | p (own − swap) | Focal rank |
|---|---|---|---|---|---|---|---|
| B | partial | +3.36 | 0.003 | +0.17 | −0.11 | 0.066 | 18 of 47 |
| B | raw | +1.65 | 0.003 | +0.60 | −0.29 | 0.003 | 7 of 53 |
| C | partial | +2.14 | 0.050 | +0.12 | −0.08 | 0.156 | |
| C | raw | +1.35 | 0.003 | +0.53 | −0.32 | 0.003 | |
| D | partial | +0.68 | 0.282 | +0.08 | −0.08 | 0.226 | 13 of 40 |
| D | raw | +0.24 | 0.126 | +0.17 | −0.15 | 0.080 | 18 of 48 |

**GRASS passes the letter of the go criterion and fails its spirit.** Condition B clears the
criterion through both estimators. But every check that distinguishes a GRASS-specific
effect from the generic lib/con pattern fails: the partial own-minus-swap gap is not
significant, GRASS sits mid-pack among items used as focal (18th of 47), and the
sorting-proof education contrast is null. Its raw lib/con result (r_own +0.60) is no larger
than the median item's (+0.45), which is the non-specific block movement already flagged for
HOMOSEX.

Two lessons. First, **the pre-set criterion was too lenient**: the generic lib/con movement
passes it for most items, so passing says little. Future criteria must include specificity —
the focal-item rank and condition D. Second, **the evidence for group-specific spillover
currently rests on HOMOSEX alone.**

The strong form remains untestable here too: GRASS has five opposite-sign edges in B, and the
camp-item predictions match 6 of 10 times, which is chance.

## Exploratory: spillover against focal embeddedness

Specified after the GRASS result and reported as exploratory (`scripts/spill_02_focal_embeddedness.py`).
GRASS has almost no partial-correlation edges — at most 0.10 among liberals and near zero among
conservatives and the non-college — and spillover needs edges to travel along. Across every
item used as focal, r_own was correlated with the item's embeddedness: its mean absolute edge
weight (*strength*) and how differently the two groups wire it (*difference*).

| Condition | Estimator | r_own vs strength | r_own vs difference |
|---|---|---|---|
| B (lib/con) | partial | ρ = +0.34 (p = 0.021) | ρ = +0.52 (p < 0.001) |
| B (lib/con) | raw | ρ = +0.26 (p = 0.065) | ρ = +0.15 (p = 0.30) |
| D (education) | partial | ρ = +0.09 (p = 0.57) | ρ = +0.07 (p = 0.66) |
| D (education) | raw | ρ = +0.24 (p = 0.095) | ρ = +0.28 (p = 0.056) |

In the lib/con partial networks, spillover is strongest for the items the two camps wire most
differently, as the account predicts; HOMOSEX ranks first of 47 on that difference and GRASS
31st on strength. Two cautions. The relationship does not hold in the education contrast,
where only HOMOSEX and CAPPUN stand out. And part of it may be signal-to-noise rather than
theory, since a focal item with larger edge differences gives a less noisy predictor. CAPPUN
ranks near the top of r_own in B and D on partial correlations, which makes it a candidate
third focal item — chosen post hoc, so it would need its own pre-specified test.

## Sorting control

Specified before running (tracker, 2026-10-15). Self-identified liberals in 2014–18 are not the
people who called themselves liberal in 1987–91, and a change in who is in a camp moves its
averages with no spillover at all. Two new conditions:

- **E** — actual camps, each late window reweighted to its baseline mix of party (Dem/Ind/Rep)
  and denomination fundamentalism as well as demographics.
- **F, predicted camps — decisive.** A 1987–91 model predicts liberal vs conservative from traits
  fixed before adulthood (region, place type, religion and fundamentalism at 16, parents'
  degrees, own degree, cohort, race, sex). The same model assigns everyone in both windows; top
  third vs bottom third; US-born, born 1925–62. Membership never depends on self-identification.

Verdict rule: survives sorting if F gives β > 0, permutation p < 0.05 and r_own > r_swap for at
least one estimator.

| Focal | Cond. | Estimator | β p (perm) | r_own | r_swap |
|---|---|---|---|---|---|
| HOMOSEX | E | partial | 0.106 | +0.10 | −0.09 |
| HOMOSEX | E | raw | 0.110 | +0.30 | −0.26 |
| HOMOSEX | F | partial | 0.090 | +0.11 | −0.11 |
| HOMOSEX | F | raw | 0.269 | +0.14 | −0.01 |
| GRASS | F | partial | 0.020 | +0.14 | −0.14 |
| GRASS | F | raw | 0.189 | +0.10 | +0.08 |

**By the pre-set rule, the HOMOSEX result does not survive the sorting control.** In F it points
the right way (own network positive, other network negative) but at about a third of its
within-cohort size and short of significance. E, which reweights the late window on party,
weakens it similarly; the reweighting is heavy there (late liberal effective sample 551 of 955).
GRASS, oddly, passes the F rule on partial correlations, although it failed everything else.

**F is weak partly by construction.** The background model separates the camps only modestly
(AUC 0.66): the predicted-liberal group is 42% liberal and 24% conservative, the
predicted-conservative group 20% and 42%. Groups that fuzzy have small network differences, so
F has little power. F also shows sorting directly: over the period the predicted-conservative
group went from 42% to 51% conservative.

### Benchmark G: fuzziness or sorting?

Exploratory, designed after F's smoke-test estimates and before its full results
(`scripts/spill_03_fuzziness_benchmark.py`). G builds groups exactly as fuzzy as F's — the same
eligible respondents, sizes, reweighting and liberal/moderate/conservative mix — but from
*self-identified* respondents, so they remain exposed to re-sorting. If F's weakness were only
fuzziness, F and G would agree; if sorting drives the effect, G would exceed F. 30 random draws.

| Focal | Estimator | F r_own | G mean r_own [95% range] | Share of G draws above F |
|---|---|---|---|---|
| HOMOSEX | partial | +0.10 | +0.05 [−0.12, +0.24] | 30% |
| HOMOSEX | raw | +0.14 | +0.26 [+0.04, +0.52] | 73% |
| GRASS | partial | +0.15 | +0.05 [−0.20, +0.27] | 17% |
| GRASS | raw | +0.10 | +0.15 [−0.17, +0.42] | 57% |
| CAPPUN | partial | +0.16 | +0.07 [−0.12, +0.22] | 23% |
| CAPPUN | raw | −0.13 | +0.28 [+0.06, +0.52] | 100% |

The two estimators split cleanly.

- **Raw correlations: sorting inflates the result.** Sorting-exposed groups beat sorting-free
  groups for every item, completely so for CAPPUN, whose raw effect vanishes once sorting is
  removed. This fits the earlier sign that the raw lib/con result is non-specific: re-sorting
  shifts whole correlated blocks of beliefs together, and raw correlations encode those blocks.
- **Partial correlations: no sign that sorting drives it.** For every item the sorting-free F
  matches or beats the sorting-exposed G. But at this fuzziness the effect is small (r ≈ 0.1)
  and F cannot confirm it.

## Replication: CAPPUN (held-out windows)

CAPPUN surfaced in the GRASS run's focal placebo on the 1987–91/2014–18 windows, so a test on
those windows is not independent. The primary test used held-out windows — **1993–96 vs
2006–10**, entirely different respondents — under the strict rule (condition B; top-quartile
focal rank; significant condition D).

| Windows | Cond. | Estimator | β p (perm) | r_own | r_swap | Focal rank |
|---|---|---|---|---|---|---|
| held-out | B | partial | 0.233 | +0.09 | −0.03 | 13 of 21 |
| held-out | B | raw | 0.106 | +0.38 | +0.15 | 6 of 26 |
| held-out | D | raw | 0.007 | +0.48 | −0.49 | 2 of 19 |
| held-out | F | partial | 0.990 | −0.24 | +0.24 | |
| selection | B | partial | 0.003 | +0.35 | −0.01 | 1 of 46 |
| selection | B | raw | 0.003 | +0.68 | +0.06 | 5 of 53 |

**CAPPUN fails its independent test.** On the held-out windows condition B is not significant,
the focal rank is mid-pack, and F goes the wrong way. Only the education contrast holds. On the
selection windows it looks as strong as HOMOSEX, which is exactly why the held-out test was
needed. One mitigation: the held-out shifts are smaller (liberals −0.33 SD, conservatives
−0.16) and fewer items moved, so that test has less power.

## Where this leaves the claim

1. **HOMOSEX passes the strict rule** (condition B, top-quartile rank, significant D) on partial
   correlations. No other item does.
2. **The raw-correlation version is partly sorting.** The headline raw numbers (r_own ≈ 0.6)
   should not be quoted as spillover.
3. **The partial-correlation version is not sorting-driven but is underpowered** once sorting is
   removed with these fuzzy groups.
4. **The first fully independent test of a new item failed.**

Net: group-specific spillover is **suggestive for HOMOSEX on partial correlations and not yet
established**. The decisive next test needs groups that are both sorting-proof and sharply
ideological.

A numerical caveat: graphical LASSO at the project's settings (max_iter = 100) often stops at its
iteration limit, so partial-correlation estimates wobble slightly between processes (F HOMOSEX
r_own was +0.111 in one run and +0.100 in another). Conclusions are unaffected at this
precision, but final estimates should use a converged fit.

## Panel test: the same people, camps fixed

`scripts/spill_04_panel_test.py`, specified before running. The GSS 2006, 2008 and 2010 panels
follow the same respondents over three waves; pooling respondents present at waves 1 and 3
gives 2,323 people in the two camps (2006→10, 2008→12, 2010→14). A person's camp is fixed, so
nobody can re-sort. Networks come from wave-1 answers and change is each person's wave-3 minus
wave-1 answer. The smoke run exposed regression to the mean — camps chosen on a noisy wave-1
POLVIEWS drift towards the centre — so a corrected rule (camp from the mean of wave-1 and
wave-3 POLVIEWS, still fixed per person) was added and is reported alongside the
pre-specified wave-1 rule.

| Camp rule | Focal | Shift lib / con (SD) | Estimator | β p (perm) | r_own | r_swap |
|---|---|---|---|---|---|---|
| wave 1 | HOMOSEX | −0.015 / −0.102 | partial | 0.457 | +0.02 | +0.11 |
| wave 1 | HOMOSEX | | raw | 0.319 | +0.09 | +0.16 |
| mean of waves 1, 3 | HOMOSEX | −0.034 / −0.088 | partial | 0.747 | −0.05 | −0.02 |
| mean of waves 1, 3 | HOMOSEX | | raw | 0.413 | +0.05 | +0.04 |

**Null under both rules, for all three items.** But the decisive fact is the size of the shifts:
over four years liberals moved −0.03 SD on HOMOSEX and conservatives −0.09. There is almost no
within-person shift to spread, so the panel cannot detect spillover even if it exists. What it
does show is that individuals change slowly.

## Decomposition: what moves a camp?

`scripts/spill_05_change_decomposition.py`, exploratory. For each four-year panel window, a
camp's cross-sectional change in the focal item is split into cohort replacement (total minus
within-cohort change), individual change (the panel's own respondents) and re-sorting between
camps (within-cohort minus individual). Raw [−1, 1] units; each cell carries sampling error of
roughly ±0.04.

| Focal | Camp | Total | Replacement | Individual | Re-sorting |
|---|---|---|---|---|---|
| HOMOSEX | lib (mean of 3 windows) | −0.085 | −0.004 | −0.032 | −0.049 |
| HOMOSEX | con | −0.134 | −0.007 | −0.083 | −0.045 |
| GRASS | lib | +0.194 | −0.010 | +0.209 | −0.006 |
| GRASS | con | +0.136 | +0.009 | +0.109 | +0.018 |

Over four years cohort replacement contributes almost nothing. For conservatives most of the
HOMOSEX change is individuals changing their minds. **For liberals more of it is re-sorting** —
people entering or leaving the camp — than individual change, although the windows disagree
(2008–12 runs the other way). GRASS change is carried by individuals in both camps.

## Where the evidence now points

The cross-sectional pattern is real: a camp's own arrangement predicts which of its beliefs
move together, beyond what the two camps share. But the evidence that this is *spillover* — a
shift in one belief spreading to others inside people's heads — has fallen away at each step
that isolates individuals. It shrinks when re-sorting is removed (F, G), and the panel shows
individuals change too slowly to test it. The decomposition suggests why: for liberals, a large
share of camp change on HOMOSEX is re-sorting.

The reading this supports — **an interpretation, not a tested result** — is that camps change
along their own belief structure substantially because of *who joins and leaves them*. People
whose beliefs fit a camp's bundle move into it, so the camp's averages shift along the lines
its network describes. That is still a population-level consequence of belief structure, which
is the paper's claim, but the mechanism is composition rather than diffusion within persons,
and "spillover" is the wrong word for it. It is directly testable in the panels: do the people
who switch camps hold beliefs that fit the destination camp's network?

## Next

- **Test the sorting mechanism directly in the panels:** whether switchers' wave-1 beliefs fit
  the network of the camp they move to, better than the one they leave.
- **Raise power without sorting:** pool the test across focal items rather than one at a time,
  using sorting-proof groups (education, predicted camps).
- **Converged graphical LASSO** (higher max_iter) for any estimate that goes into the paper.

- **A design that can test the strong form.** Either a focal item whose edges flip between
  camps, or the network-lag version restricted to sign-flipped edges.
- **Rolling windows**, to connect spillover to the ~2008 break in divergence.
- Engage Turner-Zwinkels & Brandt (2022) as the direct precedent and Fishman & Davis (2022)
  as the null to answer; see `notes/2026-10-15-pre-emption-reads.md`.
