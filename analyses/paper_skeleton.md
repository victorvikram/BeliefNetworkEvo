# Paper Skeleton: Structural Divergence of Liberal and Conservative Belief Systems

## Title

**The Reorganization of American Ideology: How Liberals and Conservatives Increasingly Disagree on Which Beliefs Matter, 1974-2022**

## Thesis (one paragraph)

Using five decades of General Social Survey data (1974-2022, N=72,390), we show that liberal and conservative Americans do not simply disagree on the issues — they organize their beliefs according to fundamentally different structural logics. We estimate the conditional dependency structure among ~120 attitude variables using regularized partial correlations (graphical LASSO), which reveals the sparse architecture of direct belief-to-belief relationships — distinct from the dense marginal correlation matrix where everything correlates with everything. This sparse structure enables graph-theoretic analyses (centrality, community detection, structural balance) that have no meaningful analog in the full correlation matrix. The key finding is that the two groups increasingly disagree on which beliefs are *structurally central* — which issues serve as organizing hubs for the rest of the belief system (centrality rank correlation declining from 0.79 to 0.58, p<0.0001). This divergence has been growing continuously for 48 years (slope=+0.010/yr), is driven by reorganization rather than reversal (sign disagreements ~0), and is primarily within-domain (73%). Conservative belief systems are more modular, reflecting individual-level heterogeneity rather than factional coalition structure. These findings reframe polarization as a structural phenomenon: Americans are not just moving to opposite poles on a shared axis, but constructing different axes altogether.

---

## Methodological Argument: Why Conditional Dependency Structure (and When It Matters)

The paper uses two distinct levels of analysis. Being transparent about which findings depend on which level is essential for intellectual honesty.

**Level 1: Correlation matrices and belief vectors.** Several findings require only pairwise correlations or individual response profiles:
- Multi-dimensionality (PCA): PC1 = 10%, purely a vector-space finding
- Matrix distance between lib/con (Euclidean distance, Pearson r): matrix comparison, no graph needed
- Temporal divergence trend (slope = +0.010/yr): comparing matrices over time
- Individual-level constraint and heterogeneity (sound_06): raw belief vectors
- Domain decomposition of divergence: matrix algebra on partitioned correlation matrices

These are valid and informative analyses in their own right.

**Level 2: Sparse conditional dependency graph (graphical LASSO → network).** The graphical LASSO transforms the dense ~7,300-entry correlation matrix into a sparse graph of ~370 edges by removing indirect associations. This sparsification is not just visualization — it changes *what you can measure*:
- **Centrality** becomes meaningful: in the dense matrix, every variable has "degree" 120. In the sparse graph, degree and betweenness distinguish hub variables from peripheral ones. The centrality divergence finding (rho 0.79 → 0.58) is the paper's strongest result and has no analog in the full correlation matrix.
- **Bridge variables** (high betweenness, low PC1 loading) can only be identified in a sparse graph where path structure exists.
- **Structural balance** (99% balanced triads, p < 0.001) is a signed-graph property requiring discrete edges.
- **Community detection** benefits from sparsification (Louvain on a sparse graph vs factor analysis on a dense matrix), though both give broadly similar domain groupings.

**The honest framing:** The paper's contribution scales with how much each finding depends on the sparse conditional dependency structure. The headline result — centrality reorganization — is fully network-dependent. The supporting evidence (divergence trend, individual heterogeneity) uses simpler methods that strengthen the argument without requiring the network framework. The paper should present both levels clearly, not dress up matrix comparisons as network findings.

---

## Narrative Arc

### Section 1: Ideology is multi-dimensional, and its relational structure carries information beyond issue positions

**Claim:** The standard unidimensional model of ideology (liberal-conservative spectrum) fails to capture the structure of American belief systems. Beyond the high dimensionality of the belief space, the *relationships between beliefs* — which pairs of attitudes constrain each other — carry information that individual-level positions do not.

**Evidence:**
- PCA on the 2000-2010 correlation matrix: PC1 explains only 10% of variance; 27 components needed for 50%. PC1 = moral/cultural axis; PC2 = partisan/economic axis — these are distinct dimensions (sound_01)
- Graphical LASSO on the same data yields a sparse graph (~370 of ~7,300 possible edges), revealing direct conditional dependencies distinct from marginal correlations
- Community detection on the sparse graph recovers 10 interpretable belief domains: political, civil liberties, morality/family, institutional confidence, spending, abortion, child-rearing, police, religion, social trust (sound_01)
- Bridge variables (NATFARE, AFFRMACT, CONLEGIS) have high betweenness but low PC1 loading — they are structurally important because they *connect* domains, not because they load on an ideological axis. These are invisible in PCA but clear in the graph (sound_01)
- Structural balance: 99% of triads are balanced (p < 0.001 vs null), indicating that belief-to-belief consistency is a real structural property, not an artifact (sound_01)

**Implication:** The belief space is high-dimensional (PCA tells us this). But *which beliefs directly constrain which other beliefs* is a separate question that requires modeling conditional dependencies. The sparse graph that results has interpretable structure — communities, hubs, bridges — that the dense correlation matrix does not. The rest of the paper uses this framework where it adds value, and simpler methods where it doesn't.

**Figure role:** Establishes multi-dimensionality and the sparse dependency structure.

---

### Section 2: Liberals and conservatives have different dependency structures

**Claim:** When estimated separately, liberal and conservative conditional dependency structures are statistically distinguishable, with systematic differences in density, local clustering, and domain organization.

**Evidence:**
- Permutation test on correlation matrices: observed Euclidean distance = 0.890; null mean = 0.625, Z = 6.81, p < 0.001. *[Matrix-level; does not require graph framing]* (sound_02)
- 60 edges have bootstrap 95% CIs excluding zero. *[Matrix-level]* (sound_02)
- Liberal graph denser: 405 vs 356 edges, even after sample-size matching. *[Graph property — edge count in sparse graph]* (sound_02)
- Conservative graph more clustered: 0.421 vs 0.369. *[Graph property — requires sparse topology]* (sound_02)
- Community structure reorganizes: conservatives merge abortion with morality (19-member cluster); liberals keep them separate. Conservatives integrate partisan identity into the policy cluster; liberals separate them. *[Graph property — Louvain on sparse graph]* (sound_02)
- 20 variables switch communities between the two graphs (sound_02)

**Implication:** The two groups don't just hold different positions — they organize the conditional dependencies among their beliefs differently. The matrix-level permutation test establishes that the difference is real; the graph-level analyses (density, clustering, communities) characterize *how* the structures differ.

**Figure role:** Shows the permutation null distribution and highlights the key structural contrasts.

---

### Section 3: This structural divergence has been growing for 50 years

**Claim:** The lib/con structural difference is not static — it has been increasing monotonically since at least 1974, with acceleration after ~2004.

**Evidence:**
- Euclidean distance between correlation matrices: slope = +0.010/yr, r = 0.852, p < 0.0001, rising from 0.72 (1976) to 1.29 (2020). *[Matrix-level — this is a comparison of partial correlation matrices over time, not inherently a graph property]* (sound_03)
- Pearson r between edge weights declining from 0.90 to 0.80 — the two groups increasingly disagree on *which* pairs of beliefs are directly linked. *[Matrix-level]* (sound_03)
- Liberals have been denser (more non-zero partial correlations) in every single window across the full 48-year span. *[Graph property — counts edges in sparse graph]* (sound_03)
- Edge gap widens substantially after ~2004 (sound_03)

**Implication:** Structural polarization predates the internet era and the commonly cited acceleration points for attitudinal polarization. It is a long-run trend, not a recent rupture. Note: the core divergence trend is a matrix comparison and does not depend on the network framing — it would survive even without sparsification. The density trend does depend on the sparse graph.

**Figure role:** Time-series showing divergence trajectory with key historical markers.

---

### Section 4: The divergence is driven by reorganization of what's central, not reversal of what's connected

**Claim:** The mechanism of divergence is not that liberals and conservatives see opposite relationships between beliefs (they don't), but that they increasingly disagree on which beliefs are structurally central — which issues organize all the others.

**This is the section where the network framing is essential.** Centrality — the identification of hub variables that organize the rest of the belief system — is a graph property with no meaningful analog in the full correlation matrix or in individual belief vectors. It requires sparsification to be well-defined, and the finding that lib/con centrality rankings are diverging is the paper's strongest and most novel result.

**Evidence:**
- Sign disagreements essentially zero across all 22 time windows (max = 1 per window). The divergence is quantitative, not qualitative. *[Matrix-level]* (sound_04)
- Dimensionality trends are parallel: both groups becoming more multi-dimensional at the same rate. The divergence is not about complexity. *[Matrix-level]* (sound_04)
- Within-domain divergence accounts for 73% of total; both within and between growing proportionally. *[Matrix-level]* (sound_04)
- **Key finding (graph-dependent):** Centrality rank correlation declining from 0.79 to 0.58 (slope = -0.006/yr, p < 0.0001) — the strongest divergence signal. This is the finding that justifies the network approach: it reveals *what* is structurally changing beneath the matrix-level divergence trend (sound_04)
- In 2018-2022, conservatives center their graphs around PRAYER, SPANKING, NATARMS; liberals center around NATCITY, NATAID, POLHITOK, GUNLAW. *[Graph-dependent — requires degree centrality in sparse graph]* (sound_04)
- POLVIEWS rising in centrality fastest (+0.004/yr, r = 0.864); GRASS declining fastest (-0.003/yr). Politicization increasing, culture-war settlement decreasing. *[Graph-dependent]* (sound_05)
- Community structure is highly stable despite all this: NMI ~0.946 (total). The *containers* for belief stay constant; the *importance* of each container changes. *[Graph-dependent]* (sound_05)

**Implication:** Polarization is not about mirror-image opposition. Both groups agree on the *sign* of relationships (e.g., religiosity and social conservatism are positively correlated for everyone). What differs — and what is diverging — is which issues serve as organizing hubs. When conservatives organize around prayer and discipline while liberals organize around police accountability and urban policy, they are not just disagreeing — they are reasoning from different structural premises. This finding is invisible without the network representation.

**Figure role:** Two panels — centrality rank correlation declining over time, and a hub-migration plot showing which variables are rising/falling in centrality.

---

### Section 5: Conservative modularity reflects individual heterogeneity, not factionalism

**Claim:** Conservative belief systems are more modular not because the conservative coalition consists of distinct factions (religious right, libertarians, hawks), but because individual conservatives hold beliefs more independently across domains — a "big tent" of individually distinctive belief profiles.

**Note on method:** This section is primarily an individual-level analysis using raw belief vectors, not the network. The network observation (higher conservative modularity) motivates the question, but the evidence comes from comparing domain score distributions, constraint measures, and clustering of individual respondents. This is honest: the network flags the pattern, individual-level analysis explains it.

**Evidence:**
- Modularity gap is stable over time (~0.72 vs ~0.65), not itself diverging. *[Graph property — motivates the question]* (sound_04)
- Cross-domain correlations weaker for conservatives (mean |r| = 0.068 vs 0.075). *[Vector-level — domain score correlations]* (sound_06)
- Belief constraint (PC1 R-squared) significantly lower for conservatives: 0.184 vs 0.216, p < 0.001. *[Vector-level]* (sound_06)
- GMM clustering: conservatives need *fewer* clusters (k=3 vs k=5 for liberals) — opposite of the coalition prediction. *[Vector-level]* (sound_06)
- Conservative distributions wider and more platykurtic in 5/9 domains, especially Spending (var ratio 1.37), Civil liberties (1.31), Morality (1.21). *[Vector-level]* (sound_06)
- Distributions are flatter, not bimodal — continuous heterogeneity, not discrete factions. *[Vector-level]* (sound_06)
- All four tests converge on the Independence hypothesis (sound_06)

**Implication:** The "big tent" metaphor is structurally accurate. Conservative coalition coherence operates differently from liberal coherence: liberals are more constrained (knowing one position predicts others), while conservatives agree on fewer cross-domain linkages, leaving more room for individual variation. This has implications for party strategy, coalition management, and theories of ideological structure.

**Figure role:** Belief constraint distributions and/or domain correlation matrices showing the independence pattern.

---

## Evidence Table

| # | Claim | Analysis | Key Numbers | Method Level | Figure |
|---|-------|----------|-------------|--------------|--------|
| 1a | Ideology is multi-dimensional | sound_01 | PC1=10%, 27 components for 50% | Matrix (PCA) | Fig 1 |
| 1b | Sparse dependency structure has interpretable topology | sound_01 | 370 edges, 10 communities, bridge variables, 99% balance (p<0.001) | Graph | Fig 1 |
| 2 | Lib/con dependency structures are statistically distinguishable | sound_02 | Z=6.81 (p<0.001), 60 significant edges, 405 vs 356 edges, clustering 0.369 vs 0.421 | Matrix + Graph | Fig 2 |
| 3 | Divergence has been growing for 48 years | sound_03 | slope=+0.010/yr, r=0.852, distance 0.72→1.29, similarity 0.90→0.80 | Matrix (+ Graph for density) | Fig 3 |
| 4a | Divergence is magnitude, not direction | sound_04 | Sign disagreements ~0, dimensionality parallel, within-domain 73% | Matrix | Fig 4 |
| 4b | **Centrality disagreement is the key mechanism** | sound_04, sound_05 | **Centrality rho: 0.79→0.58 (p<0.0001)**, POLVIEWS +0.004/yr, GRASS -0.003/yr, NMI=0.946 | **Graph (essential)** | Fig 4 |
| 5 | Conservative modularity = individual heterogeneity | sound_06 | Constraint: 0.184 vs 0.216 (p<0.001), GMM k: 3 vs 5, var ratio 1.10 | Graph (question) + Vector (evidence) | Fig 5 |

---

## Key Figures (5)

### Figure 1: From Correlation Matrix to Sparse Dependency Structure

**What it shows:** (A) Scree plot demonstrating PC1 explains only 10% of variance — ideology is not a single axis. (B) Sparse dependency graph (graphical LASSO) with nodes colored by community membership, showing 10 interpretable belief domains. (C, optional) Comparison showing what the dense correlation matrix looks like vs the sparse graph — motivating why sparsification matters.

**Reader takeaway:** Belief systems are high-dimensional, and the conditional dependency structure among beliefs is sparse and interpretable. The graph reveals communities, hubs, and bridges that are invisible in PCA or the full correlation matrix.

**Source figures:** `sound_01_scree.png`, `sound_01_communities.png` (combine; panel C would be new)

---

### Figure 2: Liberal and Conservative Networks Differ Systematically

**What it shows:** (A) Permutation null distribution with observed distance marked (Z=6.81). (B) Side-by-side community structure comparison showing key domain reorganizations (e.g., abortion-morality merger in conservative network).

**Reader takeaway:** The structural difference is not noise — it is 6.8 standard deviations beyond chance. The two groups organize beliefs into different domain groupings.

**Source figures:** `sound_02_permutation.png`, `sound_02_communities.png` (combine into panels)

---

### Figure 3: 48 Years of Structural Divergence

**What it shows:** (A) Euclidean distance between lib/con networks over time (1974-2022) with regression line and confidence band. (B) Pearson correlation between edge weights declining over the same period. Optional: annotate with key political events (Reagan, Gingrich, Tea Party, Trump).

**Reader takeaway:** Liberal and conservative belief architectures have been growing apart steadily for five decades. This is a long-run structural trend, not a recent phenomenon.

**Source figures:** `sound_03_trajectories.png` (redesign to emphasize the divergence story; annotate with historical context)

---

### Figure 4: Reorganization, Not Reversal

**What it shows:** (A) Centrality rank correlation (Spearman rho) between lib and con networks declining from 0.79 to 0.58 over time. (B) Hub migration: variables gaining centrality (POLVIEWS, PARTYID, CONCLERG) and losing centrality (GRASS, COLHOMO, PORNLAW) over time. (C, optional) Sign disagreements near zero across all windows.

**Reader takeaway:** The two groups still agree on whether beliefs are positively or negatively related — they increasingly disagree on *which beliefs are central*. Political identity is becoming the organizing hub; settled culture-war issues are losing structural importance.

**Source figures:** `sound_04_centrality_rank.png`, `sound_05_centrality_gainers_losers.png` (combine; sign disagreements from `sound_04_sign_disagree.png` could be supplementary)

---

### Figure 5: The "Big Tent" — Individual Heterogeneity, Not Factional Coalitions

**What it shows:** (A) Distribution of individual-level belief constraint (PC1 R-squared) for liberals vs conservatives, showing conservatives are less constrained. (B) BIC curves for GMM clustering showing liberals need more clusters than conservatives (opposite of coalition prediction).

**Reader takeaway:** Conservative modularity reflects individual-level independence of belief domains, not factional subgroups. Each conservative holds a more idiosyncratic combination of positions.

**Source figures:** `sound_06_constraint.png`, `sound_06_clustering_bic.png` (combine into panels)

---

## Discussion Points

### For polarization theory
- Standard polarization measures (thermometer gaps, issue positions) capture *attitudinal* polarization. Structural divergence is a distinct phenomenon: the two groups could hold identical distributions on every issue and still have diverging belief architectures. We are measuring a different thing.
- The 48-year trend suggests structural polarization is a deeper, slower process than the attitudinal polarization spikes tied to specific political events.
- The centrality mechanism offers a new account of why compromise is difficult: when two groups organize their entire belief systems around different hub issues (prayer vs police accountability), they are not just disagreeing — they are reasoning from different structural premises.

### For political science
- The finding that POLVIEWS centrality is rising fastest connects to the "sorting" literature (Mason 2018): political identity is increasingly the organizing principle of the full belief system.
- Culture-war variables losing centrality (GRASS, COLHOMO) is consistent with the consensus-formation interpretation: once an issue is resolved, it no longer differentiates, so it loses network centrality.
- The "big tent" finding (conservative heterogeneity = individual-level, not factional) challenges standard typology approaches (Pew's political typologies, etc.) that assume discrete subgroups. The data support a continuous heterogeneity model.
- The denser liberal network may relate to asymmetric enforcement of ideological consistency within the two coalitions.

### For cognitive science / belief systems theory
- Near-perfect structural balance (99% balanced triads, p < 0.001) across the full network suggests that cognitive consistency pressures operate at the population level, not just within individuals.
- The stability of community structure (NMI ~0.946) despite changing content suggests that the *domains* of belief are cognitive primitives — they reflect how people naturally organize attitudes — while the *importance* of each domain is responsive to political context.
- The independence finding (conservative lower constraint) raises questions about the relationship between ideological constraint and cognitive style.

### For methodology
- Regularized partial correlations (graphical LASSO) serve a dual purpose: they handle multicollinearity across 120+ variables, and they produce sparse graphs where graph-theoretic metrics (centrality, clustering, balance) are well-defined. Without sparsification, these metrics are trivial or uninterpretable.
- The paper should be explicit about which findings require the graph representation and which could be obtained from simpler matrix or vector comparisons. Centrality divergence (the headline finding) is fully graph-dependent. The temporal divergence trend and individual heterogeneity analysis are not. This transparency strengthens rather than weakens the argument — it shows the network framing is used where it adds genuine insight, not as decoration.
- Sample-size matching at every comparison point prevents density artifacts from contaminating structural comparisons.
- The permutation + bootstrap framework provides rigorous significance testing for network comparisons without parametric assumptions.

---

## Supplementary Material (brief notes for later)

- Full variable list with descriptions, coding, and community assignments
- Sensitivity analyses: alpha parameter for regularization, window size, matching method
- All 10 community profiles at the reference period
- Edge-level difference tables (60 significant edges)
- Domain decomposition details and temporal trajectories
- Robustness: alternative community detection algorithms, alternative centrality measures
- Additional centrality-disagreement tables for each time period
