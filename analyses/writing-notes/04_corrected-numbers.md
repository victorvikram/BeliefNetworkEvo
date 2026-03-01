# Corrected Numbers: Master Reference

All skeleton claims must use these HAC-corrected and robustness-checked values.
Primary specification: **non-overlapping windows (step=4, n=12)**. Secondary:
overlapping windows with HAC correction (step=2, n=21).

All results below use **fixed variables excluding POLVIEWS/PARTYID (N=62)**
unless otherwise noted.

---

## Section 3: Structural Divergence Trend

| Specification | Slope (/yr) | r | p | Source |
|---------------|-------------|------|---------|--------|
| Original (unfixed vars, OLS) | 0.00985 | 0.894 | <0.0001 | sound_03 |
| Fixed vars (64), OLS | 0.00435 | 0.691 | 0.0004 | sound_07 Check 1 |
| Fixed vars excl POLVIEWS (62), OLS | 0.00410 | 0.675 | 5.73e-04 | sound_07 Check 4 |
| **HAC-corrected (overlapping, 62 vars)** | **0.00417** | — | **8.08e-04** | sound_08 |
| **Non-overlapping (primary, 62 vars)** | **0.00440** | **0.655** | **0.0207** | sound_08 |

**Recommended skeleton language:** "Euclidean distance between liberal and
conservative partial correlation matrices increases at +0.0044 per year (non-overlapping
windows, p=0.021; HAC-corrected overlapping, p<0.001)."

**Structural break:** Acceleration detected at ~2008 (slope before=0.00112,
after=0.01840). Report as exploratory.

**Alpha sensitivity (fixed 64 vars):** Euclidean slope significant at all 5 alpha
levels tested (0.10-0.30), range [0.0035, 0.0055]. All p<0.01. (sound_07 Check 2)

---

## Section 4: Centrality Divergence

| Specification | Slope (/yr) | r | p | Source |
|---------------|-------------|------|---------|--------|
| Original (unfixed, with POLVIEWS) | -0.00576 | -0.864 | <0.0001 | sound_03 |
| Fixed vars (64), OLS | -0.00203 | -0.635 | 0.0015 | sound_07 Check 1 |
| Fixed vars excl POLVIEWS (62), OLS | -0.00144 | -0.461 | 0.0309 | sound_07 Check 4 |
| HAC-corrected (overlapping, 62 vars) | -0.00145 | — | 5.40e-04 | sound_08 |
| **Non-overlapping (primary, 62 vars)** | **-0.00102** | **-0.568** | **0.054** | sound_08 |

**Honest assessment:** The centrality rho trend is **WEAKENED** under correction.
HAC overlapping survives (p=5.4e-04), but the cleanest test (non-overlapping)
is marginal (p=0.054). The paper should:
1. Lead with Euclidean distance as the primary metric (robust)
2. Present centrality rho as supporting evidence with appropriate caveats
3. Frame centrality as "directionally consistent" rather than "highly significant"

**Full-matrix comparison (sound_07 Check 3):** Raw Pearson weighted-degree rho
trend is STRONGER than LASSO (slope=-0.0037, r=-0.94 vs -0.0020, r=-0.64).
This is important for the LASSO reframing (see Doc 05).

---

## Section 4: Hub Variables (POLVIEWS-excluded)

Top centrality movers from sound_07 Check 4 (excluding POLVIEWS/PARTYID):

| Variable | Slope | r | Description |
|----------|-------|-------|-------------|
| PORNLAW | 0.5644 | 0.649 | Pornography laws |
| POLHITOK | 0.4980 | 0.703 | Approve police hitting citizens |
| POLMURDR | 0.3171 | 0.577 | Approve police striking murder suspects |
| CONPRESS | 0.2998 | 0.536 | Confidence in the press |
| SPKHOMO | 0.2364 | 0.418 | Allow homosexual to speak |

FDR-surviving variables (centrality rank-difference slopes, sound_08 Part C):

| Variable | Slope | q-value |
|----------|-------|---------|
| POLHITOK | 0.4983 | 0.0051 |
| POLMURDR | 0.2739 | 0.0051 |
| SUICIDE1 | 0.1766 | 0.0051 |
| POLABUSE | -0.1825 | 0.0030 |

**Interpretation:** Morality/policing variables (PORNLAW, POLHITOK, POLMURDR)
are the fastest-diverging organizing hubs. This is more substantively interesting
than "POLVIEWS is central" because it is not circular.

---

## Section 2: Static Difference

| Metric | Value | Source |
|--------|-------|--------|
| Permutation Z-score | 6.81 | sound_02 |
| Permutation p | <0.001 | sound_02 |
| Significant edges (bootstrap) | 60 | sound_02 |
| Liberal edges | 405 | sound_02 |
| Conservative edges | 356 | sound_02 |
| Liberal clustering | 0.369 | sound_02 |
| Conservative clustering | 0.421 | sound_02 |

These are reference-period (2000-2010) numbers. The per-window permutation
tests (sound_09) confirm significance at all 22 windows (22/22 at p<0.05,
16/22 at p<0.001). Z-scores increase over time (2.8-4.7 early, 9.0-14.3 late).

---

## Section 1: Structural Balance

| Metric | Value | Source |
|--------|-------|--------|
| Observed balance | 99.0% (516/521 triads) | sound_12 |
| Null A (random signs) | 51.1% (std 2.2%) | sound_12 |
| Null B (degree-preserving) | 51.1% (std 4.1%) | sound_12 |
| Null C (signed config model) | 50.5% (std 4.3%) | sound_12 |
| All null p-values | <0.001 (1000 permutations each) | sound_12 |

**Recommended language:** "Observed structural balance (99%) significantly exceeds
all three null models, including the most stringent signed configuration model
(null mean: 51%, p<0.001)."

---

## Section 5: Conservative Heterogeneity

| Metric | Value | Source |
|--------|-------|--------|
| Belief constraint (PC1 R²), liberal | 0.216 | sound_06 |
| Belief constraint (PC1 R²), conservative | 0.184 | sound_06 |
| Constraint difference p | <0.001 | sound_06 |
| GMM clusters (liberal) | k=5 | sound_06 |
| GMM clusters (conservative) | k=3 | sound_06 |
| Cross-domain mean |r| (liberal) | 0.075 | sound_06 |
| Cross-domain mean |r| (conservative) | 0.068 | sound_06 |

These are from sound_06 and were not subject to temporal trend correction.
No changes needed.

---

## Additional Robustness (sound_11)

| Check | Key Result | Verdict |
|-------|-----------|---------|
| Exclude 2021-2022 | slope=0.00277, p=0.0025 | PASS |
| PARTYID alternative split | slope=0.00389, p=7.56e-04 | CONVERGENT |
| Survey weights (WTSSALL) | r(weighted,unweighted) lib=0.994, con=0.997 | PASS |
| Eigenvalue audit | min_eig=0.0205, no negative eigenvalues | PASS |
| POLVIEWS composition | stable (%lib 24-33%, %con 29-37%) | INFO |

---

## Honest Survival Assessment

| Claim | Survives? | Notes |
|-------|-----------|-------|
| Lib/con diverging (Euclidean) | **YES** | p=0.021 non-overlapping, p<0.001 HAC |
| Centrality misalignment | **WEAKENED** | p=0.054 non-overlapping, p<0.001 HAC |
| Structural balance | **YES** | All 3 null models rejected |
| Conservative heterogeneity | **YES** | Not a temporal trend claim |
| POLVIEWS is fastest-rising hub | **CONTAMINATED** | Excluded from primary; supplementary only |
| Morality/policing hubs diverging | **YES** | FDR q<0.01 for POLHITOK, POLMURDR |
| Divergence predates internet | **YES** | Pre-2004 windows significant |
| Divergence robust to alpha | **YES** | 5/5 alphas significant |
| Divergence robust to mode switch | **YES** | Excluding 2021-22: p=0.0025 |
| Divergence robust to PARTYID split | **YES** | p=7.56e-04 |

---

## Numbers That Changed Most from Original Skeleton

1. **Divergence slope:** 0.010/yr → 0.0044/yr (55% was variable-count inflation)
2. **Centrality rho trajectory:** "0.79→0.58 (p<0.0001)" → slope -0.0010, p=0.054 (marginal)
3. **POLVIEWS centrality:** Headline → Supplementary (circularity)
4. **Hub variables:** POLVIEWS, PARTYID, GRASS → PORNLAW, POLHITOK, POLMURDR
5. **Balance null:** ~51% (one null) → 50.5-51.1% (three nulls, all p<0.001)
