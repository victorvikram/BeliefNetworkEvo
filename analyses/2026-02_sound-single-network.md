# Sound Single Network Analysis

**Notebook:** `notebooks/sound_01_single_network.ipynb`
**Date:** 2026-02

## Question

What is the structure of the 2000-2010 belief network, and what can we actually conclude from it?

This analysis addresses three conceptual gaps in the baseline:
1. Is the network basically one-dimensional (liberal-conservative)?
2. What are the community structures?
3. Is 99% structural balance actually surprising?

## Reference Network

**Period:** 2000-2010 (biennial GSS years)
**Method:** Regularized partial Pearson correlation (graphical LASSO, alpha=0.2)
**Result:** 121 variables, 376 non-zero edges

## 1. Dimensionality Check (PCA)

Eigendecomposition of the pairwise-complete Pearson correlation matrix:

| Component | Variance Explained | Cumulative |
|-----------|-------------------|------------|
| PC1 | 10.0% | 10.0% |
| PC2 | 5.8% | 15.8% |
| PC3 | 3.5% | 19.3% |
| PC4 | 3.2% | 22.5% |
| PC5 | 2.5% | 25.0% |

- Components needed for 50% variance: **27**
- Components needed for 80% variance: **68**

**Verdict: The network is genuinely multi-dimensional.** PC1 explains only 10% of variance -- far below the 40% threshold that would indicate a one-dimensional structure. This means centrality and community findings are substantively interesting, not just proxies for a liberal-conservative axis.

**PC1 loadings** (civil liberties / moral traditionalism axis): HOMOSEX, LIBMSLM, SPKMSLM, SPKCOM, LIBCOM, ABSINGLE, ABNOMORE, ABANY

**PC2 loadings** (partisan / redistributive axis): PRESLAST_DEMREP, PARTYID, HELPBLK, EQWLTH, HELPPOOR, HELPNOT, NATRACE, AFFRMACT

The first two components capture distinct dimensions -- moral/cultural attitudes vs partisan/economic attitudes -- confirming that the belief space is not reducible to a single axis.

## 2. Network Statistics

| Metric | Value |
|--------|-------|
| Nodes | 121 |
| Edges | 376 |
| Density | 0.052 |
| Average degree | 6.21 |
| Clustering coefficient | 0.450 |
| Transitivity | 0.425 |

## 3. Community Detection (Louvain)

26 communities detected, 10 substantive (size >= 3):

| Community | Size | Domain | Key Variables |
|-----------|------|--------|---------------|
| 1 | 23 | Political / partisan | POLVIEWS, PARTYID, PRESLAST_DEMREP, EQWLTH, HELPBLK, WRKWAYUP, RACDIF1-4 |
| 2 | 18 | Civil liberties / tolerance | All SPK*, COL*, LIB* variables |
| 3 | 14 | Morality / family | HOMOSEX, PREMARSX, GRASS, PORNLAW, PRAYER, FEFAM, SPANKING |
| 4 | 12 | Confidence in institutions | All CON* variables (CONBUS, CONFED, CONLEGIS, etc.) |
| 5 | 10 | Spending priorities (social) | NATEDUC, NATENRGY, NATENVIR, NATHEAL, NATSCI, NATPARK |
| 6 | 10 | Abortion / end-of-life | AB*, LETDIE1, SUICIDE1, SUICIDE2 |
| 7 | 5 | Child-rearing values | OBEY, THNKSELF, WORKHARD, HELPOTH, POPULAR |
| 8 | 5 | Police use of force | POLABUSE, POLATTAK, POLESCAP, POLHITOK, POLMURDR |
| 9 | 5 | Religion | CONCLERG, POSTLIFE, RELIG_Protestant, RELIG_Catholic, RELIG_None |
| 10 | 3 | Social trust | FAIR, HELPFUL, TRUST |

Communities map cleanly onto interpretable belief domains. The largest community (political/partisan) includes both ideological self-placement and concrete policy attitudes about redistribution and racial inequality, suggesting these are tightly linked in the 2000s.

## 4. Centrality with Dimensionality Context

| Variable | Betweenness | Degree | |PC1 Loading| |
|----------|-------------|--------|--------------|
| OBEY | 0.240 | 19 | 0.135 |
| ABNOMORE | 0.227 | 14 | 0.167 |
| SPKCOM | 0.219 | 13 | 0.174 |
| POLVIEWS | 0.201 | 15 | 0.117 |
| HOMOSEX | 0.200 | 28 | 0.181 |
| PRESLAST_DEMREP | 0.171 | 22 | 0.096 |

**Correlation between |PC1 loading| and degree:** r = 0.654
**Correlation between |PC1 loading| and betweenness:** r = 0.415

Degree is moderately correlated with PC1 loading but far from redundant. Betweenness centrality is only weakly correlated (r=0.415), confirming that network position captures information beyond the dominant dimension.

**Bridge variables** (high betweenness, low PC1 loading):

| Variable | Betweenness | Degree | |PC1| | Interpretation |
|----------|-------------|--------|------|----------------|
| NATFARE | 0.108 | 12 | 0.044 | Welfare spending -- bridges partisan and moral clusters |
| NATRACE | 0.072 | 14 | 0.048 | Race spending -- bridges partisan and tolerance clusters |
| AFFRMACT | 0.069 | 8 | 0.008 | Affirmative action -- bridges multiple domains |
| CONFINAN | 0.053 | 7 | 0.020 | Confidence in finance -- institutional bridge |
| CONFED | 0.040 | 7 | 0.017 | Confidence in government -- institutional bridge |
| CONLEGIS | 0.038 | 10 | 0.011 | Confidence in legislature -- institutional bridge |

These variables are structurally important *because* they connect different belief domains, not because they align with a single ideological axis.

## 5. Structural Balance with Null Model

| Metric | Observed | Null (1000 shuffles) |
|--------|----------|---------------------|
| Balance ratio | **99.0%** | 51.2% (sd=2.2%) |
| Balanced triads | 516 / 521 | ~267 / 521 |
| p-value | | **< 0.001** |

**The 99% balance is highly significant.** The null model (randomly shuffling edge signs while preserving the graph structure and fraction of positive/negative edges) produces balance around 51% -- essentially chance level. The observed 99% is ~22 standard deviations above the null mean.

Edge sign distribution: 243 positive (64.6%), 133 negative (35.4%). Even with this positive skew, random sign assignment only produces ~51% balanced triads.

**Conclusion:** Structural balance in the belief network is a real phenomenon, not an artifact of having mostly positive edges or a particular graph structure.

## Key Takeaways

1. **The network is genuinely multi-dimensional** (PC1 = 10%, not 40%+). The two leading dimensions capture moral/cultural attitudes and partisan/economic attitudes separately.
2. **Communities map onto interpretable domains**: political, civil liberties, morality, institutional confidence, spending, abortion, child-rearing, policing, religion, and social trust.
3. **Centrality is not just PC1 loading** (r=0.415 for betweenness). The most interesting nodes are *bridge variables* (NATFARE, AFFRMACT, CONLEGIS) that connect different belief domains without strongly loading on the dominant dimension.
4. **99% structural balance is highly significant** (p < 0.001). This is not a trivial consequence of the graph structure -- it reflects genuine consistency in how beliefs relate to each other.
