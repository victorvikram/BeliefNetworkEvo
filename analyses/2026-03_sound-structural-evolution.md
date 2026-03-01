# Sound Structural Evolution

**Script:** `scripts/sound_05_structural_evolution.py`
**Date:** 2026-03

## Question

Previous analyses established that liberal and conservative belief networks have been diverging for 48 years. But how stable is the *internal* structure of these networks over time? Specifically:

1. **Community stability:** Do belief clusters persist or reorganize across time windows?
2. **Hub migration:** Which variables gain or lose centrality, and does this differ between liberals and conservatives?

## Method

Rolling windows (4-year, 2-year step) using regularized partial Pearson correlations (alpha=0.2). All matrices restricted to a fixed set of ~58 variables (intersection across all windows) for cross-window comparability. Three network types computed per window: total, liberal (POLVIEWS < 0), conservative (POLVIEWS > 0), with sample-size matching for lib/con.

Community stability measured via Normalized Mutual Information (NMI) between consecutive windows. Hub migration measured via Spearman rank correlation of degree centrality between consecutive windows, plus linear regression of centrality on time.

## Results

### 1. Community Structure is Highly Stable

| Metric | Total | Liberal | Conservative |
|--------|-------|---------|-------------|
| Mean NMI (consecutive windows) | 0.946 | 0.911 | 0.914 |
| Stable core variables (>= 80% consistency) | 36/58 (62%) | — | — |

NMI values near 1.0 mean community assignments are highly reproducible across time windows. The total network is most stable; liberal and conservative networks show slightly more community reorganization but are still highly consistent.

62% of variables remain in the same community across at least 80% of windows. These stable cores represent the backbone of the belief network structure — variable groupings that persist across nearly five decades.

### 2. Hub Migration: POLVIEWS Rising, GRASS Declining

**Top centrality gainers** (degree centrality slope per year):
| Variable | Slope | r | p |
|----------|-------|---|---|
| POLVIEWS | +0.004019 | 0.864 | <0.001 |
| CONCLERG | +0.002321 | 0.782 | <0.001 |
| RELIG_None | +0.001281 | 0.660 | <0.001 |
| ABPOOR | +0.001268 | 0.623 | 0.001 |
| PARTYID | +0.001065 | 0.848 | <0.001 |

**Top centrality losers:**
| Variable | Slope | r | p |
|----------|-------|---|---|
| GRASS | -0.003036 | -0.637 | 0.001 |
| COLHOMO | -0.002256 | -0.866 | <0.001 |
| PORNLAW | -0.001938 | -0.706 | <0.001 |
| ABNOMORE | -0.001903 | -0.627 | 0.001 |
| LIBHOMO | -0.001736 | -0.784 | <0.001 |

POLVIEWS (political self-identification) has risen dramatically in centrality — becoming more structurally connected to other beliefs over time. Similarly, PARTYID and religious variables are gaining. Meanwhile, variables about marijuana (GRASS), tolerance for homosexuals in specific roles (COLHOMO, LIBHOMO), and pornography (PORNLAW) are losing centrality, possibly as these issues become less politically divisive.

### 3. Hub Rankings Are Stable Overall

| Metric | Total | Liberal | Conservative |
|--------|-------|---------|-------------|
| Mean Spearman rho (consecutive windows) | 0.942 | 0.919 | 0.885 |

Centrality rankings are highly stable window-to-window. Conservative networks show slightly less stability (0.885 vs 0.942 for total), suggesting more internal restructuring of which beliefs are most central for conservatives.

## Key Takeaways

1. **Belief communities are structural constants.** Despite 48 years of attitudinal change, the basic clustering of beliefs into domains (political, civil liberties, morality, etc.) is remarkably stable. The *content* of what people believe changes; the *organization* of those beliefs does not.

2. **Politicization is the dominant trend.** POLVIEWS has the fastest-growing centrality of any variable — beliefs are becoming increasingly organized around political identity. PARTYID shows the same trend.

3. **Culture war issues are losing structural importance.** Variables about homosexuality, marijuana, and pornography are becoming less central — not because they're unimportant, but because consensus is forming. When everyone agrees, the variable no longer differentiates.

4. **Religion is rising in centrality.** CONCLERG and RELIG_None are gaining connections, reflecting the increasing structural role of religiosity in organizing the full belief system.
