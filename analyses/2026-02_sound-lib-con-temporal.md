# Sound Liberal vs Conservative: Temporal Trajectory

**Notebook:** `notebooks/sound_03_lib_con_temporal.ipynb`
**Date:** 2026-02

## Question

How do the structural differences between liberal and conservative belief networks evolve from 1974 to 2022?

## Method

- 4-year rolling windows, 2-year steps (23 windows, 22 successful)
- Regularized partial Pearson correlation (graphical LASSO, alpha=0.2)
- Sample-size matching at every window: the larger group is downsampled to match the smaller group's N
- Matched N ranges from 1,284 (1978-1982) to 3,054 (2018-2022)

## Key Finding: Liberal and Conservative Networks Are Diverging

The Euclidean distance between lib/con correlation matrices shows a significant upward trend:

| Metric | Value |
|--------|-------|
| Slope | +0.010 / year |
| Correlation (r) | 0.852 |
| p-value | < 0.0001 |

The two networks have been growing more structurally different for nearly 50 years.

## Trajectory by Period

| Period (midpoint) | Euclidean Distance | Lib Edges | Con Edges | Edge Gap | Pearson r |
|------|------|------|------|------|------|
| ~1976 | 0.72 | 273 | 218 | +55 | 0.903 |
| ~1982 | 0.83 | 319 | 256 | +63 | 0.890 |
| ~1988 | 0.85 | 370 | 306 | +64 | 0.901 |
| ~1994 | 1.02 | 326 | 293 | +33 | 0.864 |
| ~2000 | 1.01 | 331 | 304 | +27 | 0.866 |
| ~2006 | 1.03 | 452 | 375 | +77 | 0.864 |
| ~2012 | 1.10 | 508 | 349 | +159 | 0.845 |
| ~2020 | 1.29 | 545 | 439 | +106 | 0.805 |

## Structural Details

### Density
Liberals have a denser network in **every single window** across the full 48-year span. The gap is not constant -- it widens substantially after ~2004. In the most recent window (2018-2022), the liberal network has 545 edges vs 439 for conservatives.

### Clustering
In the 1970s-1980s, conservatives had slightly higher clustering (0.46 vs 0.42). By the 2010s-2020s, this advantage has eroded (0.40-0.42 vs 0.35-0.45, more variable). The "tighter conservative clusters" finding from the 2000-2010 snapshot is not a permanent feature -- it appeared in some periods but not others.

### Network Similarity
The Pearson correlation between lib and con edge weights declined from ~0.90 (1970s) to ~0.80 (2020s). The two networks increasingly disagree on *which* beliefs are connected, not just the strength of connections.

## Interpretation

The divergence aligns with the partisan sorting literature. Over this period:
- Liberals and conservatives are not just moving apart on individual beliefs -- they are reorganizing the *structure* of how their beliefs relate to each other
- The liberal network is becoming more interconnected (more edges, more cross-domain connections)
- The conservative network grows more slowly in connectivity
- The two networks increasingly disagree on which pairs of beliefs are directly linked

The acceleration after ~2004 is notable and coincides with increased political polarization documented in other research.

## Caveats

- Rolling windows with 2-year steps create substantial overlap between adjacent data points, which inflates the apparent smoothness of trends
- The trend test (linear regression on non-independent windows) overstates statistical significance -- the p-value should be interpreted cautiously
- Variable availability changes over time (early windows have ~80 common variables, later windows have ~118), which affects comparability
- The 2016-2020 window only contains 2 survey years, making it noisier (it was excluded)

## Key Takeaways

1. **Liberal and conservative belief networks have been diverging for 48 years** (1974-2022), with the gap accelerating after ~2004.
2. **Liberals have always had denser networks** -- this is not a recent phenomenon or a sample size artifact.
3. **Network similarity is declining** (Pearson r: 0.90 -> 0.80) -- the two groups increasingly disagree on which beliefs are connected.
4. **Conservative clustering advantage is not stable** -- it appears in some periods but not others.
