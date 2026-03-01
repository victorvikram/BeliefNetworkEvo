# LASSO Reframing

## The Problem

Sound_07 Check 3 shows that raw pairwise Pearson correlations produce a
*stronger* centrality divergence signal than the graphical LASSO:

| Method | Centrality rho slope | r | p |
|--------|---------------------|------|--------|
| Full-matrix weighted degree | -0.00374 | -0.943 | <0.0001 |
| Sparse graph degree centrality | -0.00203 | -0.635 | 0.0015 |

The paper's current framing — "This sparse structure enables graph-theoretic
analyses that have no meaningful analog in the full correlation matrix" — is
directly contradicted by our own data. The CSS reviewer flagged this: "If the
finding is stronger without sparsification, what work is the LASSO doing?"

---

## Honest Acknowledgment

**Required sentence for the paper:** "We note that the centrality divergence
signal is also present — and in fact stronger — in raw pairwise correlations
(weighted-degree rank correlation slope=-0.0037, r=-0.94) compared to the
sparse graph (slope=-0.0020, r=-0.64). The divergence finding does not
depend on sparsification."

**Supporting evidence from sound_10:** The correlation between LASSO-based
and raw Pearson distances across windows is r=0.910 — the methods agree on the
temporal trajectory. Raw Pearson z-scores are higher (10-28 vs 2.8-14.3)
because the dense matrix carries more signal (more edge weights contributing).

---

## Reframed LASSO Value

The LASSO is not needed to *detect* centrality divergence. Its value is in
*characterizing* the divergence — providing a sparse, interpretable graph
where structural analyses are well-defined. Four specific contributions:

### 1. Interpretability
A dense correlation matrix with ~1,891 non-zero edges (62 choose 2) is
uninterpretable as a graph. Every node connects to every other node.
The LASSO produces a sparse graph with ~200-400 edges where:
- Communities are well-defined (not every node in every community)
- Bridge variables are identifiable (high betweenness in a sparse topology)
- Hub-spoke structure is visible
- The graph can be *visualized* in a way that communicates structure

### 2. Parsimony — Direct vs Indirect Associations
The LASSO separates direct conditional dependencies from indirect associations
mediated through other variables. A raw correlation between PRAYER and GUNLAW
might be entirely mediated through POLVIEWS. The partial correlation isolates
the direct relationship, which is more relevant for causal interpretation.

### 3. Principled Thresholding
Without regularization, analyzing a correlation matrix as a graph requires
an arbitrary threshold (e.g., "edges with |r| > 0.1"). Different thresholds
produce different graphs. The graphical LASSO provides a principled,
data-driven sparsification via L1 penalization — no arbitrary cutoff needed.

### 4. Structural Metrics
Several analyses require a sparse, discrete-edge graph:
- **Structural balance** (99% balanced triads, p<0.001): a signed-graph
  property requiring discrete edges and a meaningful distinction between
  "connected" and "not connected"
- **Community detection** (Louvain): benefits from sparsification (though
  broadly similar results from factor analysis)
- **Betweenness centrality** and bridge identification: only meaningful
  when paths exist through a sparse topology

---

## Two-Level Framing (Strengthened)

The skeleton already introduces Level 1 (matrix) and Level 2 (graph) findings.
This framing should be strengthened and made the organizing principle:

### Level 1 Findings (Correlation Matrix — No LASSO Required)
- Multi-dimensionality (PCA: PC1=10%)
- Matrix distance between lib/con (Euclidean distance increasing, p=0.021)
- Temporal divergence trend
- Individual-level constraint and heterogeneity (sound_06)
- Domain decomposition of divergence (73% within-domain)
- Sign disagreements near zero
- Raw weighted-degree centrality divergence (slope=-0.0037)

**These are robust to method choice. They would survive with raw Pearson
correlations, partial correlations, or any reasonable correlation measure.**

### Level 2 Findings (Sparse Graph — LASSO Adds Value)
- Community structure (10 interpretable domains, domain reorganization)
- Bridge variables (NATFARE, AFFRMACT, CONLEGIS)
- Structural balance (99%, surviving three null models)
- Hub-spoke topology enabling centrality *interpretation* (which specific
  variables are most central, how communities reorganize)
- Density differences (405 vs 356 edges) — a graph property

**These add interpretive richness. They characterize the *nature* of the
divergence, not just its existence.**

### The Key Point
The centrality divergence **FINDING** exists at Level 1. The LASSO provides
Level 2 **CHARACTERIZATION** — it tells us *what kind* of divergence is
occurring (communities merging/splitting, specific bridges forming, balance
structure). The Level 2 characterization is the paper's interpretive
contribution; the Level 1 robustness is the paper's empirical foundation.

---

## Sound_10 Support

The raw Pearson robustness check (sound_10) provides the strongest support
for the LASSO reframing:

- All 22 windows significant at p<0.05 in BOTH raw Pearson and LASSO
- Correlation between raw and LASSO distance trajectories: r=0.910
- Raw Pearson trend: slope=0.034/yr, r=0.878, p<0.0001
- LASSO trend: slope=0.004/yr, r=0.675, p=0.0006

**The methods agree on the trajectory (r=0.910).** The LASSO reduces the
absolute distance (by zeroing weak edges) but preserves the temporal pattern.

---

## Skeleton Changes Required

### Thesis paragraph (line 9)

BEFORE: "which reveals the sparse architecture of direct belief-to-belief
relationships — distinct from the dense marginal correlation matrix where
everything correlates with everything. This sparse structure enables
graph-theoretic analyses (centrality, community detection, structural balance)
that have no meaningful analog in the full correlation matrix."

AFTER: "which separates direct conditional dependencies from indirect
associations. While the divergence trend itself is robust to method choice
(also appearing in raw correlations, r=0.91 trajectory agreement), the
sparse conditional dependency graph provides interpretive structure —
communities, bridges, and structural balance — that characterizes the
nature of the divergence."

### Methodological Argument section (lines 13-32)

Add after "Level 2" description:

> **Transparency note:** We emphasize that the headline divergence finding —
> increasing dissimilarity between liberal and conservative associational
> structures — does not depend on the graphical LASSO. Raw pairwise
> correlations produce the same trajectory (r=0.910 agreement) and a
> stronger centrality divergence signal (weighted-degree rho slope=-0.0037
> vs sparse degree slope=-0.0020). The graphical LASSO's contribution is
> not to *reveal* divergence but to *characterize* it: it produces a sparse
> graph where community structure, bridge variables, and structural balance
> are well-defined and interpretable. This parsimony-for-interpretability
> tradeoff is analogous to using LASSO regression rather than OLS — both
> can identify predictive variables, but LASSO selects a sparse set that is
> easier to interpret.

### Evidence Table (line 133)

Add a row or footnote:

> | — | Raw Pearson trajectory agreement | sound_07, sound_10 | r=0.910,
> raw slope=-0.0037 (stronger than sparse) | Matrix | Supp. |
