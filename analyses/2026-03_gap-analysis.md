# Gap Analysis: Paper Skeleton vs Evidence vs Reviewer Demands

Date: 2026-03-01

## How to Read This Document

Each gap is formatted as:

> **GAP [priority]-[number]: Title**
> - **Paper claims:** What the skeleton says
> - **Current evidence:** What scripts/analyses currently provide
> - **Reviewer concern:** What the panel flagged
> - **Gap:** What's missing
> - **Action:** Specific script/analysis/edit needed
> - **Effort:** S (< 1hr), M (1-4hr), L (4-8hr), XL (1+ day)

Priority: P0 = blocks submission, P1 = seriously weakens paper, P2 = improves paper,
P3 = nice to have / future work.

---

## P0: Submission Blockers

### GAP P0-1: Serial Autocorrelation — All Temporal p-Values Are Wrong

- **Paper claims:** "slope=+0.010/yr, p<0.0001" (Section 3), "centrality rho slope,
  p<0.0001" (Section 4), and dozens more temporal trend p-values throughout.
- **Current evidence:** All scripts use `scipy.stats.linregress()` which assumes
  independent observations. 22 overlapping windows share 50% of data.
- **Reviewer concern:** Statistician rated this the single most likely issue to change
  conclusions. Effective N may be 8-12 instead of 22. p-values could inflate by 10x.
- **Gap:** No corrected standard errors anywhere in the pipeline.
- **Action:** New script `sound_08_corrected_inference.py`:
  1. For every temporal trend regression in the pipeline (Euclidean distance, centrality
     rho, sign disagreements, dimensionality, modularity, centrality slopes), compute:
     - Newey-West HAC standard errors (maxlags=1) via `statsmodels`
     - Durbin-Watson statistic
     - Effective degrees of freedom
  2. Run parallel analysis with non-overlapping windows (step_size=4, ~11 windows)
  3. Report corrected p-values alongside originals
  4. If any headline result crosses p=0.05 after correction, flag immediately
- **Effort:** M (need `statsmodels` HAC; non-overlapping windows just re-run existing
  pipeline with different step_size)

### GAP P0-2: No Engagement with DellaPosta (2020)

- **Paper claims:** Novel contribution to understanding belief system evolution.
- **Current evidence:** Paper skeleton does not mention DellaPosta at all.
- **Reviewer concern:** CSS and PolSci reviewers both flagged this as near-fatal. Same
  data, same timespan, published in ASR. Any reviewer will ask "what does this add?"
- **Gap:** No literature positioning relative to the most direct predecessor.
- **Action:** Paper writing task (not a script). Requires:
  1. Explicit differentiation paragraph in introduction
  2. Dedicated subsection in related work
  3. Frame: DellaPosta showed whole-network consolidation; we show differential
     architecture between groups + centrality divergence (which he didn't measure)
  4. Also engage: Boutyline & Vaisey (2017), Baldassarri & Gelman (2008),
     Mason (2018), Kozlowski & Murphy (2021), Goldberg & Stein (2018),
     Baldassarri & Goldberg (2014), Fishman & Davis (2022)
- **Effort:** L (literature review + writing, not code)

### GAP P0-3: Ecological Fallacy Language Throughout

- **Paper claims:** "reasoning from different structural premises," "which issues serve
  as organizing hubs for the rest of the belief system," "cognitive consistency
  pressures operate at the population level."
- **Current evidence:** All analyses are population-level partial correlations.
- **Reviewer concern:** Philosopher, Complex Systems, and PolSci all flagged this.
  Population-level conditional dependencies ≠ individual cognitive architecture.
- **Gap:** Paper language implies individual-level cognition from aggregate data.
- **Action:** Paper writing task. Systematic audit of all interpretive claims:
  1. Use "belief network" for the statistical object; "belief system" only when
     discussing theory
  2. Replace "organize beliefs" with "beliefs co-vary" or "associational structure"
  3. Add explicit caveat paragraph referencing Brandt & Sleegers (2021) and
     Robinson (1950)
  4. Frame: "Population-level structure reflects the joint product of individual
     psychology, social influence, and elite cue-giving. Tracking its evolution is
     valuable even without decomposing these sources."
- **Effort:** M (systematic find-and-replace + framing paragraph)

### GAP P0-4: POLVIEWS/PARTYID Should Be Excluded from Primary Analysis

- **Paper claims:** "POLVIEWS rising in centrality fastest (+0.004/yr)" is presented as
  a headline finding (Section 4, Evidence Table row 4b).
- **Current evidence:** sound_07 Check 4 shows centrality divergence survives exclusion
  (p=0.031) but marginally. Top movers shift to PORNLAW, POLHITOK, POLMURDR.
- **Reviewer concern:** Four reviewers flagged circularity. The variable used to define
  groups should not appear as a node.
- **Gap:** The primary analysis includes POLVIEWS/PARTYID. The excluded version is a
  robustness check, not the main result.
- **Action:** Restructure the narrative:
  1. Make POLVIEWS/PARTYID-excluded the primary centrality analysis
  2. Move with-POLVIEWS results to supplementary
  3. Update paper skeleton Section 4: headline hub variables become PORNLAW, POLHITOK,
     POLMURDR, CONPRESS, TRUST (from sound_07 Check 4 output)
  4. Reinterpret: "morality and policing variables are the fastest-diverging organizing
     hubs" — actually a MORE interesting finding than "POLVIEWS is central"
  5. Rerun sound_05 structural evolution with vars_no_pol to get clean hub migration
     trajectories
- **Effort:** M (rerun sound_05 variant + paper restructuring)

---

## P1: Seriously Weakens Paper If Not Addressed

### GAP P1-1: Graphical LASSO Value-Add Is Empirically Undermined

- **Paper claims:** "Centrality — the identification of hub variables — requires
  sparsification to be well-defined, and the finding that lib/con centrality rankings
  are diverging is the paper's strongest and most novel result." Also: "This sparse
  structure enables graph-theoretic analyses that have no meaningful analog in the full
  correlation matrix."
- **Current evidence:** sound_07 Check 3 shows the full-matrix weighted-degree rho
  trend is STRONGER (slope=-0.0037, r=-0.94) than the sparse graph
  (slope=-0.0020, r=-0.64). The paper's own robustness check contradicts its framing.
- **Reviewer concern:** CSS reviewer: "If the finding is stronger without
  sparsification, what work is the LASSO doing?"
- **Gap:** The "revelation" framing is falsified by our own data.
- **Action:** Reframe the LASSO contribution in the paper (writing task):
  1. Acknowledge honestly that centrality divergence exists in raw correlations too
  2. Reframe LASSO value as: (a) interpretability — produces a graph where community
     detection, bridge identification, and betweenness are well-defined; (b) parsimony —
     separates direct from indirect associations; (c) principled thresholding — avoids
     arbitrary cutoffs
  3. The centrality divergence is the headline FINDING; the LASSO is the methodological
     framework for CHARACTERIZING it (communities, bridges, balance)
  4. Update the Methodological Argument section of the skeleton
  5. Consider presenting Level 1 (matrix) and Level 2 (graph) results side by side
     to show what the graph adds beyond detection
- **Effort:** M (writing + skeleton restructuring)

### GAP P1-2: Structural Balance Null Model Is Too Weak

- **Paper claims:** "99% balanced triads, p < 0.001 vs null ~51%"
- **Current evidence:** sound_01 compares against random sign assignment, which gives
  ~50% balance mechanically. With ~80% positive edges, a degree-preserving null
  would give much higher expected balance.
- **Reviewer concern:** Complex Systems reviewer: "comparing against the most
  permissive possible null." Need sign-preserving and degree-preserving nulls.
- **Gap:** The balance claim may be trivially true given the edge sign distribution.
- **Action:** New analysis (add to existing script or new `sound_09_balance_nulls.py`):
  1. Null model A: random sign assignment (current — keep as baseline)
  2. Null model B: edge-rewiring preserving degree sequence and sign ratio
  3. Null model C: configuration model preserving signed degree sequence
  4. Report balance against all three nulls
  5. If balance is still significantly elevated against null C, the finding is robust
  6. If not, reframe: "The high balance reflects the preponderance of positive partial
     correlations" — still interesting, but a simpler claim
- **Effort:** M

### GAP P1-3: No Confidence Intervals on Temporal Trajectories

- **Paper claims:** Euclidean distance rises from 0.72 to 1.29 (Section 3), centrality
  rho declines from 0.79 to 0.58 (Section 4).
- **Current evidence:** Point estimates only. No error bars on any temporal figure.
  No bootstrap CIs on per-window metrics.
- **Reviewer concern:** Statistician: "Without uncertainty quantification, the reader
  cannot assess whether the trends are precise or noisy."
- **Gap:** No uncertainty quantification anywhere in the temporal analysis.
- **Action:** Add bootstrap CIs to key temporal trajectories (can be in
  `sound_08_corrected_inference.py` or separate):
  1. For each window: resample respondents within each group (200+ bootstrap reps)
  2. Re-estimate graphical LASSO networks per bootstrap rep
  3. Compute Euclidean distance and centrality rho per rep
  4. Plot 95% CI bands on temporal trajectory figures
  5. Also: bootstrap CI on the 99% structural balance proportion
  Note: computationally expensive (22 windows x 200 reps x 2 groups = ~8,800 LASSO
  fits). May need to run overnight or reduce to key windows.
- **Effort:** L-XL (computation time is the bottleneck)

### GAP P1-4: No GSS Survey Weights

- **Paper claims:** Analyses use unweighted correlations throughout.
- **Current evidence:** No use of WTSSALL or WTSSNR anywhere in the pipeline.
- **Reviewer concern:** Survey methodologist: GSS documentation explicitly states
  weights must be used for 2004+. Unweighted correlations may overrepresent larger
  households.
- **Gap:** No weighted-correlation sensitivity check.
- **Action:** Sensitivity check script or addition to robustness:
  1. Check if WTSSALL is in the cleaned dataset (it may be excluded by DataConfig)
  2. If available, compute weighted Pearson correlations for the reference period
     (2000-2010) and compare to unweighted
  3. If substantively unchanged, report in one sentence
  4. If different, this becomes a major issue requiring pipeline changes
  Note: weighted graphical LASSO is non-trivial. The simplest approach is weighted
  pairwise correlations → check if the input matrix changes much. If the input barely
  changes, the LASSO output won't change either.
- **Effort:** M

### GAP P1-5: 2021 Mode Switch Not Addressed

- **Paper claims:** Temporal trends through 2022.
- **Current evidence:** Final windows include 2021-2022 (web-administered, ~17%
  response rate). No sensitivity check excluding these years.
- **Reviewer concern:** Survey methodologist: mode switch from face-to-face to web
  changes the covariance structure on sensitive items. This is exactly what we measure.
- **Gap:** No pre-2021 sensitivity check.
- **Action:** Rerun core temporal analysis excluding 2021 and 2022:
  1. Filter `cleaned_df` to YEAR <= 2018
  2. Rebuild rolling windows
  3. Compute divergence slope and centrality rho slope
  4. If trends hold through 2018, the 2021 contamination is moot for the trend claim
  5. Report this as a sensitivity check
- **Effort:** S-M (just filter the data and re-run existing functions)

### GAP P1-6: Multiple Comparisons Uncorrected

- **Paper claims:** "60 edges with bootstrap 95% CIs excluding zero" (Section 2),
  edge-level trend regressions, per-variable centrality slopes.
- **Current evidence:** ~4,200 implicit tests with no FDR correction.
- **Reviewer concern:** Statistician: "Finding several dozen 'significant' edges is
  entirely expected under the null with ~2,000 tests."
- **Gap:** Edge-level and variable-level findings have no multiple testing correction.
- **Action:**
  1. Apply Benjamini-Hochberg FDR correction to: (a) the 60 edge-level bootstrap CIs
     in sound_02, (b) edge-level trend regressions in sound_04, (c) per-variable
     centrality slopes in sound_05
  2. Report how many survive FDR q=0.05
  3. Distinguish pre-specified summary tests (overall divergence, centrality rho —
     few, robust) from exploratory edge/variable tests (many, need FDR)
- **Effort:** S-M (just apply `statsmodels.stats.multitest.multipletests` to existing
  p-value arrays)

### GAP P1-7: Permutation Test Has Technical Issues

- **Paper claims:** "Z=6.81, p<0.001" from permutation test.
- **Current evidence:** sound_02 runs 200 permutations (updated to 1000 based on
  agent catalog). Silent failure handling. Only run at reference period.
- **Reviewer concern:** Statistician: 200 iterations insufficient; failed permutations
  silently dropped biases the null; only one time period tested.
- **Gap:** Permutation test needs more iterations and transparency.
- **Action:**
  1. Increase to 10,000 permutations (or at least 1,000 — check current value)
  2. Report number of failed permutations
  3. Run permutation test at 3+ time periods (early, middle, late) to verify the
     lib/con difference is significant throughout the time series, not just at reference
  4. Consider Network Comparison Test (van Borkulo et al., 2022) as benchmark
- **Effort:** M (computationally expensive but straightforward)

---

## P2: Improves Paper Substantially

### GAP P2-1: No Alternative Ideology Split (PARTYID)

- **Paper claims:** All lib/con splits use POLVIEWS.
- **Current evidence:** Only POLVIEWS tested.
- **Reviewer concern:** PolSci, Survey, Philosopher: POLVIEWS meaning has shifted over
  48 years. Test PARTYID or a composite ideology index.
- **Gap:** Single operationalization of the key independent variable.
- **Action:** Rerun core temporal analysis with PARTYID split instead of POLVIEWS:
  1. Build rolling windows with `group_col="PARTYID"`,
     `group_conditions={"dem": "< 0", "rep": "> 0"}`
  2. Compute divergence slope and centrality rho slope
  3. If results converge with POLVIEWS results, the finding is more credible
  4. Report as sensitivity check
- **Effort:** M

### GAP P2-2: No Structural Break Test

- **Paper claims:** "acceleration after ~2004" (Section 3, informal claim).
- **Current evidence:** Only linear models fit. No piecewise or segmented regression.
- **Reviewer concern:** Statistician: "A linear model fit to a process with a
  structural break will report a moderate slope." The linear vs piecewise question is
  empirically testable.
- **Gap:** No formal test of whether the trend is continuous or has a breakpoint.
- **Action:** Add to `sound_08_corrected_inference.py`:
  1. Fit piecewise linear model with estimated breakpoint (Bai-Perron or simple grid
     search over candidate breakpoints)
  2. Compare linear vs piecewise via AIC/BIC
  3. Report: "The divergence trend is best described as [continuous/piecewise with
     breakpoint at YYYY]"
- **Effort:** M

### GAP P2-3: Eigenvalue Audit of Pairwise Correlation Matrices

- **Paper claims:** Uses graphical LASSO on pairwise-complete correlation matrices.
- **Current evidence:** No check that input matrices are positive semi-definite.
- **Reviewer concern:** Survey methodologist: pairwise-complete deletion with GSS
  ballot structure can produce non-PSD matrices. Graphical LASSO requires PD input.
- **Gap:** No PSD check or minimum eigenvalue reporting.
- **Action:** Add eigenvalue audit:
  1. For every window and group, compute eigenvalues of the pairwise correlation matrix
     BEFORE graphical LASSO
  2. Report minimum eigenvalue across all matrices
  3. If any are negative, apply nearest-PD correction (Higham) and report the correction
     magnitude
  4. Also report min/median/max pairwise-complete N per window
- **Effort:** S-M

### GAP P2-4: POLVIEWS Composition Over Time

- **Paper claims:** Lib/con divergence is structural, not compositional.
- **Current evidence:** No analysis of POLVIEWS distribution shifts over time.
- **Reviewer concern:** PolSci and Survey: if who calls themselves "liberal" changes,
  composition changes confound structural changes.
- **Gap:** No descriptive analysis of POLVIEWS distribution by decade.
- **Action:** Simple descriptive analysis:
  1. Stacked bar chart of POLVIEWS distribution by year/decade
  2. Report % liberal, % conservative, % moderate over time
  3. Show sample sizes per group per window
  4. If the liberal pool is shrinking (more selected), acknowledge this as a confounder
- **Effort:** S

### GAP P2-5: Name the Concept

- **Paper claims:** Describes structural divergence but doesn't give it a citable name.
- **Current evidence:** Paper skeleton uses "structural polarization" and "structural
  divergence" interchangeably, and also "different axes."
- **Reviewer concern:** CSS: "Give it a name. Define it formally. If reviewers can cite
  'structural misalignment (Author 2026)' as a named concept, the paper becomes a
  reference point." PolSci: "Dropping the word 'polarization' and using 'structural
  divergence' consistently" avoids confusion with the existing debate.
- **Gap:** No formal concept definition.
- **Action:** Paper writing task:
  1. Choose a term: "structural misalignment," "axial divergence," or "centrality
     divergence" — and define it formally
  2. Provide a one-paragraph operational definition distinguishing it from attitudinal
     polarization, affective polarization, and sorting
  3. Use the term consistently throughout
- **Effort:** S (writing only)

### GAP P2-6: No Discussion of Mechanism

- **Paper claims:** Descriptive findings about temporal trends.
- **Current evidence:** No discussion of why centrality hierarchies are diverging.
- **Reviewer concern:** PolSci: "For a top journal, reviewers will expect at least a
  discussion of which mechanisms are consistent with the temporal trajectory." CSS:
  "Goldberg and Stein (2018) associative diffusion model is directly relevant."
- **Gap:** No mechanism discussion.
- **Action:** Paper writing task:
  1. Discuss candidate mechanisms: elite cue-giving (Zaller), media sorting (Prior),
     associative diffusion (Goldberg & Stein), generational replacement
  2. Note which mechanisms are consistent/inconsistent with the 48-year trend (e.g.,
     predates internet → rules out social media as primary cause)
  3. Frame as: "We document the pattern; explaining it is for future work, but the
     temporal trajectory constrains the mechanism space."
- **Effort:** M (literature review + writing)

### GAP P2-7: Elevate Independence Finding to Co-Equal Contribution

- **Paper claims:** Section 5 presents coalition-vs-independence as supporting evidence.
- **Current evidence:** sound_06 provides full analysis. Currently positioned as
  secondary.
- **Reviewer concern:** CSS: "The independence finding is counterintuitive, clean, and
  has immediate implications. Elevate to co-equal status."
- **Gap:** Narrative structure buries the second-strongest finding.
- **Action:** Paper restructuring (writing task):
  1. Frame paper as having two main contributions: (a) centrality divergence,
     (b) independence-not-coalition
  2. Title could reference both: "...How Liberals and Conservatives Increasingly
     Disagree on Which Beliefs Matter — and Why Conservative Heterogeneity Is Not
     What You Think"
  3. Give Section 5 equal narrative weight
- **Effort:** M (writing/restructuring)

### GAP P2-8: Joint Graphical LASSO Not Cited or Addressed

- **Paper claims:** Estimates lib and con networks independently.
- **Current evidence:** Independent estimation + post-hoc comparison.
- **Reviewer concern:** Complex Systems: Joint Graphical LASSO (Danaher et al. 2014)
  estimates both networks simultaneously, directly models differential structure, and
  reduces false positives. Standard in genomics.
- **Gap:** Not cited, not justified.
- **Action:** At minimum, cite JGL literature and explain why independent estimation was
  chosen (computational simplicity across 22 windows x 2 groups; temporal dimension
  makes joint estimation across ~44 group-time combinations unwieldy). Ideally, run
  JGL on the reference period and compare.
- **Effort:** S (cite and justify) to L (implement JGL)

---

## P3: Nice to Have / Future Work

### GAP P3-1: Spearman Correlation Robustness Check
- Many GSS variables are ordinal with 2-4 categories. Pearson attenuated.
- Action: Rerun core analysis with `method=CorrelationMethod.SPEARMAN`
- Effort: S

### GAP P3-2: Group-Specific Alpha via BIC
- Same alpha=0.2 for both groups; could create artificial density differences.
- Action: Cross-validate alpha separately per group per window. Report density diff.
- Effort: M

### GAP P3-3: SignedLouvain or Leiden Instead of Louvain
- Louvain has resolution limit and doesn't handle signed networks well.
- Action: Replace with Leiden or SignedLouvain, compare community stability.
- Effort: M

### GAP P3-4: Downstream Prediction
- Does network structure at time t predict political outcomes at t+1?
- Action: Correlate centrality divergence with external polarization measures.
- Effort: L-XL

### GAP P3-5: Spin Glass / Symmetry-Breaking Formalization
- Frame belief system as frustrated spin glass; centrality divergence as bifurcation.
- Action: Formal model + simulation. Separate paper or extended discussion.
- Effort: XL

### GAP P3-6: Release Temporal Matrices as Benchmark Dataset
- 22 windows x 2 groups = 44 partial correlation matrices.
- Action: Package and release (after publication).
- Effort: S

### GAP P3-7: Bootstrap Edge Inclusion Probabilities
- Report which edges are stable across bootstrap resamples (>80% inclusion).
- Action: 500+ bootstrap LASSO fits for reference period.
- Effort: L (compute time)

### GAP P3-8: Moderates as Third Comparison Group
- Including moderates (POLVIEWS=0) would test if they track one group or neither.
- Action: Add `group_conditions={"lib": "< 0", "mod": "== 0", "con": "> 0"}`
- Effort: M

---

## Summary: Effort Budget

| Priority | Count | Total Effort |
|----------|-------|-------------|
| P0 (blockers) | 4 | ~2 M code + 2 L writing |
| P1 (serious) | 7 | ~3 M + 1 L-XL code, some writing |
| P2 (improves) | 8 | ~3 M + 3 S code, several writing |
| P3 (nice) | 8 | Mixed, mostly future work |

### Recommended Implementation Order

**Phase 1 — Fix the statistics (P0-1, P1-3, P1-6, P1-7):**
Single script `sound_08_corrected_inference.py` that:
- Recomputes all trend regressions with HAC standard errors
- Runs non-overlapping windows
- Adds bootstrap CIs to key trajectories
- Applies FDR correction to edge/variable-level tests
- Increases permutation iterations
- Reports Durbin-Watson statistics

**Phase 2 — Fix the framing (P0-2, P0-3, P0-4, P1-1):**
Paper skeleton revision:
- Engage DellaPosta and key literature
- Audit ecological-level language
- Make POLVIEWS-excluded analysis primary
- Reframe LASSO as interpretability tool

**Phase 3 — Additional robustness (P1-4, P1-5, P2-1, P2-3, P2-4, P1-2):**
Script `sound_09_additional_robustness.py` that:
- Weighted correlation check
- Exclude 2021-2022 check
- PARTYID alternative split
- Eigenvalue audit
- POLVIEWS composition over time
- Better structural balance null models

**Phase 4 — Paper strengthening (P2-5, P2-6, P2-7, P2-2, P2-8):**
Writing + one analysis:
- Name the concept
- Discuss mechanisms
- Elevate independence finding
- Structural break test
- Cite/justify JGL choice
