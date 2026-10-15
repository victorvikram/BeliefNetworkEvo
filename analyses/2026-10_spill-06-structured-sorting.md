# Spill 06: Structure-Guided Sorting

**Date:** 2026-10-15
**Script:** `scripts/spill_06_structured_sorting.py`
**Log:** `logs/spill_06_structured_sorting.log`
**Data:** GSS 2006, 2008 and 2010 three-wave panels (people), GSS cross-sections 2004–14 (networks)

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

## Reading

This is the first result in the downstream-effects work that survives a design where sorting
cannot fake it — because here sorting is the outcome. It says that **the arrangement of a
camp's beliefs selects its members**: people stay where their beliefs fit together the way the
camp's do, and leave when they fit another camp's wiring better, whatever their positions.
That is a population-level consequence of belief structure, and it is the mechanism the
spillover decomposition pointed to: camps shift along their own structure because of who joins
and leaves them.

It also gives specification leg 2 — arrangement, not magnitude — an empirical form at the level
of persons. Position-based sorting is well documented; arrangement-based sorting, as far as the
literature search found, is not.

## Limits

- **Modest effect.** One SD of relative fit raises the odds of leaving by 14–25%, against 141%
  for issue position.
- **Most switches are to moderate.** The outcome is leaving a camp, not crossing to the other
  one; full crossings are too rare to model separately.
- **Measurement error in POLVIEWS** produces some apparent switching; ideological strength is
  controlled, which absorbs most of the boundary effect, but not all.
- **One set of panels, one era.** The 2016–20 panel is an independent replication waiting to be
  run.
- **Raw versus partial.** The raw estimate is larger. Raw correlations carry the camps' whole
  correlated blocks, which is legitimate for a claim about fit to what co-occurs in a camp, but
  the partial estimate is the cleaner test of fit to direct links.

## Next

- **Replicate in the 2016–20 panel**: an independent sample, a different era, and the period
  after the ~2008 break.
- Does arrangement-based sorting account for the camp-level co-movement the spillover tests
  found? Simulating camp change from the estimated switching model would answer it.
- Robustness: switching between waves 1 and 2; full crossings only; panel weights.
- Position this against Levendusky (2009), Mason (2018) and Hillygus & Shields (2008).
