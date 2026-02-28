# Sound Divergence Anatomy

**Notebook:** `notebooks/sound_04_divergence_anatomy.ipynb`
**Date:** 2026-02

## Question

Sound 03 established that liberal and conservative belief networks have been diverging for 48 years (1974-2022), with Euclidean distance rising from 0.72 to 1.29 (slope=+0.010/yr, r=0.852, p<0.0001). But *what* is diverging? Which edges, which belief domains, which structural properties are driving this?

## Method

Same rolling-window framework as sound_03: 4-year windows, 2-year steps, regularized partial Pearson (alpha=0.2), sample-size matched. But instead of storing only summary statistics, we store the full lib and con correlation matrices at each of 22 windows, enabling edge-level and domain-level decomposition.

## Results

### 1. Edge Sign Disagreements: Almost None

| Period | Sign Disagreements per Window |
|--------|-------------------------------|
| 1974-1998 | 0 |
| 2000-2022 | 0-1 (max = 1) |
| Trend | slope=+0.000061/yr, r=0.418, p=0.053 |

**The divergence is not about sign reversals.** Across all 22 windows, there are almost never edges where liberals and conservatives see opposite-sign relationships. When divergence occurs, both groups agree on the *direction* of the relationship — they just disagree on its *strength*. This means the divergence is quantitative (magnitude), not qualitative (direction).

### 2. Dimensionality: Moving in Parallel, Not Diverging

Both groups show identical dimensionality trends:

| Metric | Liberal Trend | Conservative Trend |
|--------|--------------|-------------------|
| PC1 variance explained | -0.000148/yr (r=-0.789, p<0.0001) | -0.000141/yr (r=-0.791, p<0.0001) |
| Participation ratio | +0.701/yr (r=0.809, p<0.0001) | +0.688/yr (r=0.805, p<0.0001) |

Both liberal and conservative networks are becoming more multi-dimensional over time (PC1 variance declining from ~2.4% to ~1.6%, effective dimensionality rising from ~75 to ~110). But they're moving in lockstep — the dimensionality gap between them is negligible. The divergence is not about one group becoming simpler while the other becomes more complex.

### 3. Edge-Level Drivers: Widespread, Not Concentrated

| Metric | Count |
|--------|-------|
| Total edges tracked | 6,843 |
| Positive slope (diverging) | 936 (13.7%) |
| Negative slope (converging) | 775 (11.3%) |
| Significantly diverging (p<0.05) | 208 (3.0%) |
| Significantly converging (p<0.05) | 127 (1.9%) |

**Top 10 fastest-diverging edges:**

| Edge | Slope (/yr) | r | Mean |diff| |
|------|------------|---|------|
| RACDIF2 — RACDIF4 | +0.0033 | 0.850 | 0.065 |
| COURTS — WRKWAYUP | +0.0031 | 0.729 | 0.046 |
| DIDVOTELAST — PARTYID | +0.0025 | 0.854 | 0.027 |
| CONFED — CONLEGIS | +0.0025 | 0.477 | 0.073 |
| FEFAM — POLMURDR | +0.0025 | 0.802 | 0.023 |
| HELPOTH — THNKSELF | +0.0024 | 0.724 | 0.067 |
| RELIG_Catholic — RELIG_Protestant | +0.0024 | 0.858 | 0.178 |
| PRESLAST_DEMREP — RACDIF1 | +0.0023 | 0.799 | 0.020 |
| PARTYID — PRESLAST_DEMREP | +0.0023 | 0.529 | 0.069 |
| OBEY — PRAYER | +0.0022 | 0.862 | 0.044 |

The divergence is broadly distributed across many edges rather than concentrated in a few. The top diverging edges span multiple domains: racial attitudes (RACDIF2-RACDIF4), partisan alignment (DIDVOTELAST-PARTYID, PARTYID-PRESLAST_DEMREP), institutional trust (CONFED-CONLEGIS), child-rearing/morality crossovers (OBEY-PRAYER), and religion (Catholic-Protestant).

**Fastest converging edges** mostly involve LIBMSLM (allow Muslim clergy to speak) connecting to other variables — suggesting that the lib/con disagreement on Muslim civil liberties peaked and is now declining.

### 4. Domain Decomposition: Within-Domain Drives 73% of Divergence

| Domain (within) | Mean d² | % of Total |
|-----------------|---------|-----------|
| Political | 0.196 | 21.0% |
| Civil liberties | 0.106 | 11.4% |
| Morality/family | 0.085 | 9.2% |
| Institutions | 0.077 | 8.3% |
| Religion | 0.074 | 7.9% |
| Abortion | 0.053 | 5.6% |
| Spending | 0.052 | 5.6% |
| **All within-domain** | **0.677** | **72.8%** |
| **All between-domain** | **0.253** | **27.2%** |

**Top between-domain contributors:**

| Domain Pair | Mean d² | % of Total |
|-------------|---------|-----------|
| Civil liberties x Morality/family | 0.041 | 4.4% |
| Abortion x Morality/family | 0.039 | 4.2% |
| Political x Spending | 0.030 | 3.2% |

The divergence is overwhelmingly **within-domain** (73%). Liberals and conservatives are increasingly disagreeing on how beliefs *within* the same domain relate to each other — not on how different domains connect. The Political domain alone contributes 21% of total divergence.

Both within-domain and between-domain divergence are growing significantly over time:
- Within-domain: slope=+0.013/yr, r=0.815, p<0.0001
- Between-domain: slope=+0.006/yr, r=0.741, p=0.0001
- Within-domain share: no significant trend (p=0.79) — both are growing proportionally

The domain decomposition sums exactly match total d² (max error: 2e-15), confirming correct decomposition.

### 5. Modularity: Stable Gap, Not Diverging

| Metric | Liberal | Conservative | Trend (lib) | Trend (con) |
|--------|---------|-------------|-------------|-------------|
| Modularity (Q) | ~0.65 | ~0.72 | p=0.34 (NS) | p=0.22 (NS) |
| Communities (size>=3) | ~9 | ~9 | p=0.78 (NS) | p=0.78 (NS) |

Conservative networks are consistently more modular (Q ~0.72 vs ~0.65), but this gap is stable — neither group is becoming significantly more or less modular over time. The number of detected communities is similar (~9) and stable for both groups. The divergence is not about belief *packaging* — it's about the weights within those packages.

### 6. Centrality: Increasingly Disagreeing on What's Central

| Metric | Value |
|--------|-------|
| Spearman rho trend | slope=-0.006/yr, r=-0.784, **p<0.0001** |
| Rho in 1976 | 0.79 |
| Rho in 2020 | 0.58 |

**This is the strongest structural divergence signal.** The Spearman rank correlation of degree centrality between liberal and conservative networks has been declining steadily and significantly. In the 1970s, the two groups largely agreed on which beliefs are structurally central (rho~0.80). By 2020, that agreement has eroded substantially (rho~0.58).

**Variables with largest centrality disagreement (2018-2022):**

| Variable | Rank Diff (lib - con) | Interpretation |
|----------|-----------------------|----------------|
| PRAYER | -78.5 | Much more central for conservatives |
| SPANKING | -78.0 | Much more central for conservatives |
| NATARMS | -76.0 | Much more central for conservatives |
| NATCITY | +73.5 | Much more central for liberals |
| NATAID | +73.5 | Much more central for liberals |
| POLHITOK | +69.0 | Much more central for liberals |
| WRKWAYUP | -68.0 | Much more central for conservatives |
| GUNLAW | +67.0 | Much more central for liberals |

The pattern is substantively coherent: conservatives increasingly organize their belief networks around prayer, discipline/obedience (spanking, work ethic), and military spending, while liberals increasingly organize around urban policy (city spending), foreign aid, police accountability, and gun control.

## Key Takeaways

1. **The divergence is about magnitude, not direction.** Sign disagreements are essentially zero — both groups see the same positive/negative relationships, just at different strengths. This is quantitative divergence, not qualitative.

2. **Dimensionality is not diverging.** Both groups are becoming more multi-dimensional at the same rate. The divergence is not about structural complexity.

3. **Within-domain divergence dominates (73%).** The groups are increasingly disagreeing on how beliefs within the same domain relate to each other, not on how different domains connect. The Political domain alone drives 21% of total divergence.

4. **Modularity gap is stable.** Conservatives have consistently more modular networks, but this isn't changing over time. The divergence isn't about how beliefs are *packaged* into clusters.

5. **Centrality is the key structural divergence.** The groups increasingly disagree on which beliefs are structurally central (rho declining from 0.79 to 0.58, p<0.0001). Conservatives are centering their networks around prayer, discipline, and military; liberals around urban policy, foreign aid, and police accountability.

6. **The divergence is broadly distributed.** No single edge or small set of edges drives the divergence — it's spread across hundreds of edges, with the fastest-diverging spanning racial attitudes, partisan alignment, institutional trust, and morality/child-rearing crossovers.
