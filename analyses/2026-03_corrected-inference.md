# Corrected Statistical Inference

## Problem

All trend p-values in previous analyses are potentially invalid because:

1. **Overlapping windows**: 22 windows (step=2, width=4) share 50% of data between
   adjacent windows, violating the independence assumption and inflating OLS standard
   errors. As the statistician reviewer noted: "p < 0.0001 could become p ~ 0.02-0.08."

2. **Multiple comparisons**: ~4,200 implicit edge-level and variable-level tests have
   zero correction, inflating family-wise error rate.

3. **Acceleration claims**: No formal structural break test supports claims of
   "accelerating" divergence.

## Methods

### HAC Correction (Part A)

Newey-West HAC standard errors with maxlags=1 on all headline trend regressions
using overlapping windows (step=2, n=21). This accounts for the
serial autocorrelation induced by window overlap. Durbin-Watson statistics are
reported to verify autocorrelation structure.

### Non-Overlapping Validation (Part B)

Independent replication using step=4 windows (n=12). With no
window overlap, observations are independent and plain OLS is valid. Lower power due
to fewer observations, but any surviving trend is free from autocorrelation bias.

### FDR Correction (Part C)

Benjamini-Hochberg false discovery rate correction applied to all per-variable tests:
- Centrality rank-difference slopes (62 tests)
- Per-variable degree centrality slopes (lib and con, ~62 each)

### Structural Break Tests (Part D)

Grid-search segmented regression compared to linear fit via AIC. Tests whether
the Euclidean distance and centrality rho trends show a structural break (acceleration
or deceleration) rather than a simple linear trend.

### Configuration

All analyses use:
- Fixed variables: 62 (intersection across all windows, excluding POLVIEWS and PARTYID)
- Regularized partial Pearson correlation (alpha=0.2)
- Sample-matched liberal vs conservative groups
- Random state: 42

---

## Results

### Part A: HAC-Corrected Trends

| Metric | OLS slope | OLS p | HAC slope | HAC p | DW | Non-overlap slope | Non-overlap p |
|--------|-----------|-------|-----------|-------|----|-------------------|---------------|
| Euclidean distance | 0.00417 | 7.18e-04 | 0.00417 | 8.08e-04 | 1.477 | 0.00440 | 0.0207 |
| Centrality rho | -0.00145 | 0.0372 | -0.00145 | 5.40e-04 | 2.509 | -0.00102 | 0.0542 |
| Sign disagreement | 0.00000 | 1.0000 | 0.00000 | 1.0000 | N/A | 0.00000 | 1.0000 |
| Density (lib) | -0.00036 | 0.0369 | -0.00036 | 0.0579 | 0.514 | -0.00033 | 0.1293 |
| Density (con) | -0.00002 | 0.8227 | -0.00002 | 0.8263 | 0.812 | 0.00003 | 0.8180 |
| Clustering (lib) | 0.00011 | 0.6320 | 0.00011 | 0.5737 | 0.778 | 0.00004 | 0.9032 |
| Clustering (con) | 0.00089 | 1.08e-05 | 0.00089 | 1.67e-09 | 0.891 | 0.00073 | 0.0039 |


**Key findings:**
- Euclidean distance: OLS p=7.18e-04 -> HAC p=8.08e-04 (DW=1.477)
- Centrality rho: OLS p=0.0372 -> HAC p=5.40e-04 (DW=2.509)
- Sign disagreement: OLS p=1.0000 -> HAC p=1.0000 (DW=nan)

Durbin-Watson interpretation: DW << 2 confirms positive autocorrelation from window
overlap, justifying the HAC correction. Values closer to 2 for non-overlapping windows
would further confirm this.

### Part B: Non-Overlapping Validation

- Euclidean distance: slope=0.00440, p=0.0207, r=0.655
- Centrality rho: slope=-0.00102, p=0.0542, r=-0.568
- Sign disagreement: slope=0.000000, p=1.0000, r=0.000

Directional agreement: slopes from non-overlapping windows agree with overlapping-window slopes for Euclidean distance.

### Part C: FDR-Corrected Variable-Level Tests

**Centrality rank-difference slopes:**
- Variables tested: 62
- Significant (OLS p<0.05): 13
- Significant (HAC p<0.05): 10
- Significant (FDR q<0.05): 4

FDR-surviving variables (centrality rank-difference slopes):
  - POLHITOK: slope=0.4983, q=0.0051
  - POLMURDR: slope=0.2739, q=0.0051
  - SUICIDE1: slope=0.1766, q=0.0051
  - POLABUSE: slope=-0.1825, q=0.0030


### Part D: Structural Break Tests

- Euclidean distance: acceleration detected = Yes (breakpoint ~2008, slope before=0.00112, after=0.01840)
- Centrality rho: acceleration detected = No

---

## Verdict

| Claim | HAC (p<0.05) | Non-overlap (p<0.05) | Overall |
|-------|:---:|:---:|:---:|
| Lib/con diverging (Euclidean) | Yes | Yes | **SURVIVES** |
| Centrality misalignment | Yes | No | **WEAKENED** |

### Implications for the Paper

The Euclidean distance (divergence) trend survives both corrections. The centrality
rho trend is weakened. The paper should lead with Euclidean distance as the primary
metric and present centrality results with appropriate caveats about reduced
significance under HAC correction.

### Recommended Reporting

For the paper, we recommend:
1. **Primary specification**: Non-overlapping windows (step=4) as the cleanest test
2. **Secondary specification**: Overlapping windows with HAC correction (step=2, maxlags=1)
3. **Report Durbin-Watson**: Shows the autocorrelation structure explicitly
4. **FDR correction**: All per-variable results should use FDR-corrected q-values
5. **Structural breaks**: Report segmented regression results
