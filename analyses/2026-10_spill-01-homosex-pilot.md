# Spill 01: HOMOSEX Spillover Pilot

**Date:** 2026-10-15
**Script:** `scripts/spill_01_homosex_pilot.py`
**Log:** `logs/spill_01_homosex_pilot.log`
**Figure:** `figures/spill_01_homosex_pilot.png`
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
changed.

## Results

| Cond. | Estimator | β | p (perm) | r_own | ρ_own (Spearman) | r_swap | p (own − swap) | Focal rank |
|---|---|---|---|---|---|---|---|---|
| A | partial | +4.21 | 0.003 | +0.39 | +0.46 | −0.29 | 0.007 | |
| A | raw | +0.94 | 0.003 | +0.56 | +0.57 | −0.18 | 0.003 | |
| B | partial | +4.79 | 0.003 | +0.36 | +0.46 | −0.28 | 0.007 | 1 of 47 |
| B | raw | +1.27 | 0.003 | +0.59 | +0.60 | −0.05 | 0.007 | 8 of 53 |
| C | partial | +3.70 | 0.003 | +0.28 | +0.38 | −0.23 | 0.024 | |
| C | raw | +1.17 | 0.007 | +0.49 | +0.51 | −0.24 | 0.003 | |
| D | partial | +1.74 | 0.014 | +0.24 | +0.28 | −0.23 | 0.014 | 2 of 40 |
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
   version holds (β p = 0.014, gap p = 0.014); the raw β does not (p = 0.146), although its
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

## Next

- **Sorting control for the lib/con contrast.** Candidates: reweight each camp's late window
  on religious tradition (FUND) and party identification as well as demographics; add
  moderates as a third group; or treat re-identification directly by bounding how much of
  each camp's movement a plausible flow of switchers could produce.
- **Second focal item.** GRASS (marijuana legalisation) also shifted massively and is a
  different attitude domain; replicating there separates a HOMOSEX finding from a general one.
- **A design that can test the strong form.** Either a focal item whose edges flip between
  camps, or the network-lag version restricted to sign-flipped edges.
- **Rolling windows**, to connect spillover to the ~2008 break in divergence.
- Engage Turner-Zwinkels & Brandt (2022) as the direct precedent and Fishman & Davis (2022)
  as the null to answer; see `notes/2026-10-15-pre-emption-reads.md`.
