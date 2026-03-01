# Paper Skeleton: Structural Divergence of Liberal and Conservative Belief Networks

## Title

**The Reorganization of American Ideology: How Liberals and Conservatives Increasingly Disagree on Which Beliefs Matter, 1974-2022**

## Thesis (one paragraph)

Using five decades of General Social Survey data (1974-2022, N=72,390), we document *structural divergence* — the increasing dissimilarity between the population-level associational structures of self-identified liberal and conservative Americans. We estimate the conditional dependency structure among ~120 attitude variables using regularized partial correlations (graphical LASSO), which separates direct conditional dependencies from indirect associations. While the divergence trend itself is robust to method choice (also appearing in raw correlations, with r=0.91 trajectory agreement), the sparse conditional dependency graph provides interpretive structure — communities, bridges, and structural balance — that characterizes the nature of the divergence. The Euclidean distance between group-specific partial correlation matrices increases at +0.0044 per year (non-overlapping windows, p=0.021; HAC-corrected overlapping, p<0.001), driven by reorganization rather than reversal (sign disagreements ~0) and primarily within-domain (73%). The fastest-diverging hub variables are morality and policing items — attitudes toward pornography regulation (PORNLAW), police use of force (POLHITOK, POLMURDR), and the right to die (SUICIDE1) — not partisan identity markers. This paper makes two contributions. First, we show that structural divergence has been growing continuously for 48 years, with a possible acceleration around 2008, predating the internet era and consistent with slow-moving processes like elite polarization and associative diffusion. Second, we show that conservative belief networks are more modular than liberal ones — not because the conservative coalition consists of distinct factions, but because individual conservatives hold beliefs more independently across domains. These findings reframe polarization as a structural phenomenon: Americans are not just moving to opposite poles on shared issues — the population-level dependency structures for the two groups are increasingly organized around different hub variables.

---

## Related Work

### The constraint tradition
Converse (1964) introduced "constraint" as the degree to which positions on one issue predict positions on another, arguing that mass publics show far less constraint than elites. Subsequent work has debated whether constraint has increased. Baldassarri and Gelman (2008) showed that issue alignment increased among partisans but not the general public, using pairwise correlations between issue positions. Our work extends this tradition by modeling conditional dependencies rather than pairwise correlations, comparing the *structure* of these dependencies between groups rather than their average strength, and tracking structural evolution over 48 years.

### Belief network methods
Boutyline and Vaisey (2017) introduced belief network analysis to sociology, applying community detection to correlational structure among GSS attitudes. Their approach — treating the pattern of inter-attitude correlations as a network — is the methodological foundation for both DellaPosta (2020) and our work. We extend it by comparing group-specific networks over time. Brandt and Sleegers (2021) provided an important methodological critique, warning that population-level correlations should not be interpreted as individual cognitive architecture — a caveat we take seriously throughout (see Methods caveat).

### Pluralistic collapse
DellaPosta (2020) demonstrated that American beliefs are consolidating into fewer, more encompassing modules — a process he terms "pluralistic collapse." Using 219 GSS items and walktrap community detection, he showed that belief networks have fewer modules, higher modularity, and greater concentration over 1972-2016, even after removing or controlling for political ideology and party identification. His "oil spill" model describes polarization as the spread of alignment across previously unrelated issue domains.

Our contribution is complementary: DellaPosta analyzed one network (the whole population) and asked whether it is consolidating; we analyze two group-specific networks (liberal, conservative) and ask whether they are consolidating *differently*. His framework cannot detect centrality divergence — the finding that the two groups increasingly disagree on which beliefs are structurally central — because he does not compare group-specific structures. His metrics (module count, modularity, concentration) capture whole-network properties; ours (Euclidean distance between matrices, centrality rank correlation, hub migration) capture between-group structural dissimilarity. Where DellaPosta showed beliefs are becoming more packaged, we show they are becoming packaged around different hubs.

### Sorting, alignment, and polarization
Mason (2018) distinguished sorting (alignment of social identities with party) from polarization (movement to extreme positions). Our concept of "structural divergence" is distinct from both: two groups can hold identical position distributions on every issue and still exhibit structural divergence if the population-level dependency structures — which issues co-vary with which other issues — differ between groups. Kozlowski and Murphy (2021) documented accelerating issue alignment post-2004; our structural break analysis identifies a similar inflection point around 2008 for structural divergence. Baldassarri and Goldberg (2014) showed individual-level heterogeneity in belief organization; our Section 5 finding — that conservative modularity reflects individual independence, not factional coalition structure — is the population-level signature of this individual-level pattern.

### Mechanisms
Goldberg and Stein (2018) proposed associative diffusion as a mechanism for belief spread through association networks. If liberals and conservatives inhabit networks with different seed issues, associative diffusion would naturally produce diverging correlation structures — making their model a plausible mechanism for the structural divergence we document. Fishman and Davis (2022) demonstrated temporal belief network analysis in AJPS, providing a methodological precedent for our approach.

---

## Concept Definition: Structural Divergence

*Structural divergence* is the increasing dissimilarity between the conditional dependency structures of ideologically defined subpopulations. It is measured as the Euclidean distance between group-specific partial correlation matrices (primary metric) and as the declining Spearman rank correlation between group-specific centrality hierarchies (secondary metric). Structural divergence is distinct from attitudinal polarization (movement to opposite poles on shared issues), affective polarization (increasing dislike of outgroups), and sorting (alignment of social identities with partisan identity). Two groups can hold identical position distributions on every issue and still exhibit structural divergence if they organize the relationships among those positions differently. The concept captures how the population-level associational architecture — which issues co-vary with which other issues — differs between groups and changes over time.

---

## Methodological Argument: Why Conditional Dependency Structure (and When It Matters)

The paper uses two distinct levels of analysis. Being transparent about which findings depend on which level is essential for intellectual honesty.

**Level 1: Correlation matrices and belief vectors.** Several findings require only pairwise correlations or individual response profiles:
- Multi-dimensionality (PCA): PC1 = 10%, purely a vector-space finding
- Matrix distance between lib/con (Euclidean distance): matrix comparison, no graph needed
- Temporal divergence trend (slope = +0.0044/yr): comparing matrices over time
- Individual-level constraint and heterogeneity (sound_06): raw belief vectors
- Domain decomposition of divergence: matrix algebra on partitioned correlation matrices
- Raw weighted-degree centrality divergence (slope = -0.0037, r = -0.94): pairwise correlations suffice

These are valid and informative analyses in their own right.

**Level 2: Sparse conditional dependency graph (graphical LASSO → network).** The graphical LASSO transforms the dense ~1,891-entry correlation matrix into a sparse graph of ~200-400 edges by removing indirect associations. This sparsification enables:
- **Community detection** on a sparse topology (Louvain), recovering 10 interpretable belief domains
- **Bridge variables** (high betweenness, low PC1 loading) identifiable only where path structure exists
- **Structural balance** (99% balanced triads, p < 0.001 vs all three null models) — a signed-graph property requiring discrete edges
- **Centrality interpretation** — in the sparse graph, degree and betweenness distinguish specific hub variables from peripheral ones, enabling substantive interpretation of *which* variables are reorganizing

**Transparency note:** The headline divergence finding — increasing dissimilarity between liberal and conservative associational structures — does not depend on the graphical LASSO. Raw pairwise correlations produce the same trajectory (r=0.910 agreement across windows) and a stronger centrality divergence signal (weighted-degree rho slope=-0.0037 vs sparse degree slope=-0.0020). The graphical LASSO's contribution is not to *reveal* divergence but to *characterize* it: it produces a sparse graph where community structure, bridge variables, and structural balance are well-defined and interpretable. This parsimony-for-interpretability tradeoff is analogous to using LASSO regression rather than OLS — both can identify predictive variables, but LASSO selects a sparse set that is easier to interpret.

**Joint estimation:** We estimate liberal and conservative networks independently rather than using joint graphical LASSO methods (Danaher et al. 2014; Guo et al. 2011). Our design involves 22 time windows x 2 groups = 44 networks; joint estimation across this temporal dimension is computationally prohibitive and requires specifying a penalty structure across all 44 graphs. More importantly, our research question concerns the *independent evolution* of group-specific structures — joint estimation would shrink the two networks toward a common structure, potentially attenuating the very divergence we aim to detect.

**Ecological caveat:** The partial correlation networks we estimate are population-level statistical objects. They capture the associational structure among attitudes across respondents, not the cognitive architecture of any individual. A positive partial correlation between attitudes A and B means that, controlling for all other measured attitudes, respondents who score higher on A tend to score higher on B — but this could reflect individual-level cognitive consistency, social influence from shared environments, elite cue-giving that packages issues together, or compositional differences within ideological groups. Following Brandt and Sleegers (2021), we avoid inferring individual-level cognitive processes from population-level structure. We use "belief network" to refer to the statistical object and reserve "belief system" for theoretical discussions of individual cognition (cf. Converse 1964). The evolution of population-level associational structure is valuable to track regardless of its micro-level sources, because it reveals the macro-level architecture within which political communication, coalition formation, and issue packaging operate.

---

## Narrative Arc

### Section 1: Ideology is multi-dimensional, and its relational structure carries information beyond issue positions

**Claim:** The standard unidimensional model of ideology (liberal-conservative spectrum) fails to capture the structure of American attitudes at the population level. Beyond the high dimensionality of the belief space, the *relationships between attitudes* — which pairs co-vary with each other — carry information that individual-level positions do not.

**Evidence:**
- PCA on the 2000-2010 correlation matrix: PC1 explains only 10% of variance; 27 components needed for 50%. PC1 = moral/cultural axis; PC2 = partisan/economic axis — these are distinct dimensions (sound_01)
- Graphical LASSO on the same data yields a sparse graph (~370 of ~7,300 possible edges), revealing direct conditional dependencies distinct from marginal correlations
- Community detection on the sparse graph recovers 10 interpretable belief domains: political, civil liberties, morality/family, institutional confidence, spending, abortion, child-rearing, police, religion, social trust (sound_01)
- Bridge variables (NATFARE, AFFRMACT, CONLEGIS) have high betweenness but low PC1 loading — they are structurally important because they *connect* domains, not because they load on an ideological axis. These are invisible in PCA but clear in the graph (sound_01)
- Structural balance: 99% of triads are balanced, significantly exceeding all three null models — random sign assignment (null mean 51.1%), degree-preserving rewiring (51.1%), and the most stringent signed configuration model (50.5%) — all p < 0.001, 1000 permutations each. This indicates genuine constraint structure beyond degree-sequence artifacts (sound_01, sound_12)

**Implication:** The belief space is high-dimensional (PCA tells us this). But *which attitudes co-vary with which other attitudes at the population level* is a separate question that requires modeling conditional dependencies. The sparse graph that results has interpretable structure — communities, hubs, bridges — that the dense correlation matrix does not. The rest of the paper uses this framework where it adds value, and simpler methods where it doesn't.

**Figure role:** Establishes multi-dimensionality and the sparse dependency structure.

---

### Section 2: Liberals and conservatives have different dependency structures

**Claim:** When estimated separately, liberal and conservative conditional dependency structures are statistically distinguishable, with systematic differences in density, local clustering, and domain organization.

**Evidence:**
- Permutation test on correlation matrices: observed Euclidean distance = 0.890; null mean = 0.625, Z = 6.81, p < 0.001. Per-window permutation tests confirm significance at all 22 windows (22/22 at p<0.05, 16/22 at p<0.001). *[Matrix-level]* (sound_02, sound_09)
- 60 edges have bootstrap 95% CIs excluding zero. *[Matrix-level]* (sound_02)
- Liberal graph denser: 405 vs 356 edges, even after sample-size matching. *[Graph property]* (sound_02)
- Conservative graph more clustered: 0.421 vs 0.369. *[Graph property]* (sound_02)
- Community structure reorganizes: conservatives merge abortion with morality (19-member cluster); liberals keep them separate. Conservatives integrate partisan identity into the policy cluster; liberals separate them. *[Graph property]* (sound_02)
- 20 variables switch communities between the two graphs (sound_02)

**Implication:** The two groups don't just hold different positions — the population-level conditional dependencies among their attitudes differ. The matrix-level permutation test establishes that the difference is real; the graph-level analyses (density, clustering, communities) characterize *how* the structures differ.

**Figure role:** Shows the permutation null distribution and highlights the key structural contrasts.

---

### Section 3: This structural divergence has been growing for 48 years

**Claim:** The lib/con structural difference is not static — it has been increasing since at least 1974, with a possible acceleration around 2008.

**Evidence:**
- Euclidean distance between partial correlation matrices: +0.0044/yr (non-overlapping windows, p=0.021; HAC-corrected overlapping, p<0.001). *[Matrix-level]* (sound_07, sound_08)
- Structural break test detects acceleration at ~2008 (pre-break slope=0.0011, post-break slope=0.0184). Reported as exploratory. *[Matrix-level]* (sound_08)
- Raw Pearson correlation trajectory agrees with LASSO trajectory: r=0.910 across windows. The divergence finding is robust to method choice. *[Matrix-level]* (sound_10)
- Liberals have been denser (more non-zero partial correlations) in every single window across the full 48-year span. *[Graph property]* (sound_03)
- Conservative clustering shows a significant increasing trend (HAC p<0.001, non-overlapping p=0.004). *[Graph property]* (sound_08)

**Implication:** Structural divergence predates the internet era and the commonly cited acceleration points for attitudinal polarization. It is a long-run trend, not a recent rupture. The core divergence trend is a matrix comparison and does not depend on the network framing — it would survive even without sparsification.

**Figure role:** Time-series showing divergence trajectory with key historical markers.

---

### Section 4: The divergence is driven by reorganization of what's central, not reversal of what's connected

**Claim:** The mechanism of structural divergence is not that liberals and conservatives see opposite relationships between attitudes (they don't), but that the two groups increasingly differ in which attitudes are most central in the population-level dependency structure.

**Evidence:**
- Sign disagreements essentially zero across all 22 time windows (max = 1 per window). The divergence is quantitative, not qualitative. *[Matrix-level]* (sound_04)
- Dimensionality trends are parallel: both groups becoming more multi-dimensional at the same rate. The divergence is not about complexity. *[Matrix-level]* (sound_04)
- Within-domain divergence accounts for 73% of total; both within and between growing proportionally. *[Matrix-level]* (sound_04)
- Centrality rank correlation between groups shows a declining trend (HAC overlapping p<0.001; non-overlapping p=0.054 — directionally consistent but marginal under the strictest test). *[Graph-dependent]* (sound_08)
- The fastest-diverging hub variables (FDR q<0.01) are POLHITOK (police hitting citizens), POLMURDR (police striking murder suspects), SUICIDE1 (right to die), and POLABUSE (police abuse) — morality and authority variables, not partisan identity markers. *[Graph-dependent]* (sound_07, sound_08)
- Community structure is highly stable: NMI ~0.946 (total). The *containers* for belief stay constant; the *importance* of each container changes. *[Graph-dependent]* (sound_05)

**Implication:** Structural divergence is not about mirror-image opposition. Both groups agree on the *sign* of relationships (e.g., religiosity and social conservatism are positively correlated for everyone). What differs — and what is diverging — is which issues are most central in the population-level dependency structure. Morality and policing variables (attitudes toward pornography regulation, police use of force, the right to die) are the fastest-diverging organizing hubs, suggesting that the moral-authority dimension of ideology, rather than partisan identity itself, drives the structural divergence.

**Figure role:** Two panels — centrality rank correlation between groups over time, and hub-divergence plot showing which variables' centrality rank-differences are growing fastest (FDR q<0.05).

---

### Section 5: Conservative heterogeneity — individual independence, not factional coalition

**Claim:** Conservative belief networks are more modular not because the conservative coalition consists of distinct factions (religious right, libertarians, hawks), but because individual conservatives hold beliefs more independently across domains — a "big tent" of individually distinctive belief profiles.

**Note on method:** This section is primarily an individual-level analysis using raw belief vectors, not the network. The network observation (higher conservative modularity) motivates the question, but the evidence comes from comparing domain score distributions, constraint measures, and clustering of individual respondents. This is honest: the network flags the pattern, individual-level analysis explains it.

**Evidence:**
- Modularity gap is stable over time (~0.72 vs ~0.65), not itself diverging. *[Graph property — motivates the question]* (sound_04)
- Cross-domain correlations weaker for conservatives (mean |r| = 0.068 vs 0.075). *[Vector-level]* (sound_06)
- Belief constraint (PC1 R-squared) significantly lower for conservatives: 0.184 vs 0.216, p < 0.001. *[Vector-level]* (sound_06)
- GMM clustering: conservatives need *fewer* clusters (k=3 vs k=5 for liberals) — opposite of the coalition prediction. *[Vector-level]* (sound_06)
- Conservative distributions wider and more platykurtic in 5/9 domains, especially Spending (var ratio 1.37), Civil liberties (1.31), Morality (1.21). *[Vector-level]* (sound_06)
- Distributions are flatter, not bimodal — continuous heterogeneity, not discrete factions. *[Vector-level]* (sound_06)
- All four tests converge on the Independence hypothesis (sound_06)

**Implication:** The "big tent" metaphor is structurally accurate but for unexpected reasons. Conservative coalition coherence does not operate through bargains between internally homogeneous factions. Instead, individual conservatives hold more idiosyncratic combinations of positions across domains — each person's belief profile is more distinctive, producing lower cross-domain correlations at the population level. This has direct implications for theories of party management: the Republican coalition's heterogeneity is not a principal-agent problem (managing faction leaders) but a diversity-management problem (accommodating individually variable belief profiles). It also challenges standard typology approaches (Pew's political typologies, etc.) that assume discrete subgroups — the data support a continuous heterogeneity model.

**Figure role:** Belief constraint distributions and/or domain correlation matrices showing the independence pattern.

---

## Evidence Table

| # | Claim | Analysis | Key Numbers | Method Level | Figure |
|---|-------|----------|-------------|--------------|--------|
| 1a | Ideology is multi-dimensional | sound_01 | PC1=10%, 27 components for 50% | Matrix (PCA) | Fig 1 |
| 1b | Sparse dependency structure has interpretable topology | sound_01, sound_12 | 370 edges, 10 communities, 99% balance (all 3 nulls p<0.001, incl. signed config null=50.5%) | Graph | Fig 1 |
| 2 | Lib/con dependency structures are statistically distinguishable | sound_02, sound_09 | Z=6.81 (p<0.001), 22/22 windows sig., 405 vs 356 edges, clustering 0.369 vs 0.421 | Matrix + Graph | Fig 2 |
| 3 | Structural divergence has been growing for 48 years | sound_07, sound_08, sound_10 | slope=+0.0044/yr (non-overlap p=0.021, HAC p<0.001), raw Pearson r=0.910 agreement, break at ~2008 | Matrix (+ Graph for density) | Fig 3 |
| 4a | Divergence is magnitude, not direction | sound_04 | Sign disagreements ~0, dimensionality parallel, within-domain 73% | Matrix | Fig 4 |
| 4b | Hub divergence concentrated in morality/authority variables | sound_07, sound_08 | POLHITOK (q=0.005), POLMURDR (q=0.005), SUICIDE1 (q=0.005), POLABUSE (q=0.003); centrality rho: HAC p<0.001 / non-overlap p=0.054 | Graph (essential) | Fig 4 |
| 5 | Conservative modularity = individual heterogeneity, not factional coalition | sound_06 | Constraint: 0.184 vs 0.216 (p<0.001), GMM k: 3 vs 5, var ratio 1.10-1.37 | Graph (question) + Vector (evidence) | Fig 5 |

---

## Key Figures (5)

### Figure 1: From Correlation Matrix to Sparse Dependency Structure

**What it shows:** (A) Scree plot demonstrating PC1 explains only 10% of variance — ideology is not a single axis. (B) Sparse dependency graph (graphical LASSO) with nodes colored by community membership, showing 10 interpretable belief domains. (C, optional) Comparison showing what the dense correlation matrix looks like vs the sparse graph — motivating why sparsification matters.

**Reader takeaway:** The population-level associational structure among attitudes is high-dimensional and sparse. The graph reveals communities, hubs, and bridges that are invisible in PCA or the full correlation matrix.

**Source figures:** `sound_01_scree.png`, `sound_01_communities.png` (combine; panel C would be new)

---

### Figure 2: Liberal and Conservative Networks Differ Systematically

**What it shows:** (A) Permutation null distribution with observed distance marked (Z=6.81). (B) Side-by-side community structure comparison showing key domain reorganizations (e.g., abortion-morality merger in conservative network).

**Reader takeaway:** The structural difference is not noise — it is 6.8 standard deviations beyond chance. The two groups' population-level associational structures differ in specific, interpretable ways.

**Source figures:** `sound_02_permutation.png`, `sound_02_communities.png` (combine into panels)

---

### Figure 3: 48 Years of Structural Divergence

**What it shows:** (A) Euclidean distance between lib/con networks over time (1974-2022) with regression line and confidence band; HAC-corrected p-value annotated. (B) Raw Pearson distance trajectory alongside LASSO trajectory (r=0.910 agreement), demonstrating robustness. Optional: annotate with key political events (Reagan, Gingrich, Tea Party, Trump) and structural break at ~2008.

**Reader takeaway:** Liberal and conservative associational structures have been growing apart steadily for five decades. This is a long-run structural trend, not a recent phenomenon, and it is robust to analytical method choice.

**Source figures:** `sound_03_trajectories.png`, `sound_08_corrected_inference.png`, `sound_10_raw_pearson.png` (combine/redesign)

---

### Figure 4: Reorganization, Not Reversal

**What it shows:** (A) Centrality rank correlation (Spearman rho) between lib and con networks over time, with HAC-corrected trend line. (B) Hub divergence: variables whose centrality rank-difference between groups is growing fastest (FDR q<0.05): POLHITOK, POLMURDR, SUICIDE1 (becoming relatively more central for one group), POLABUSE (becoming relatively more central for the other). These morality and policing variables represent the substantive core of structural divergence. (C, optional) Sign disagreements near zero across all windows.

**Reader takeaway:** The two groups still agree on whether attitudes are positively or negatively related — they increasingly differ on *which attitudes are central* in the population-level dependency structure. Morality and authority variables, not partisan identity markers, are the fastest-diverging hubs.

**Source figures:** `sound_04_centrality_rank.png`, sound_07/sound_08 hub-divergence data (may need new figure from POLVIEWS-excluded results)

---

### Figure 5: The "Big Tent" — Individual Heterogeneity, Not Factional Coalitions

**What it shows:** (A) Distribution of individual-level belief constraint (PC1 R-squared) for liberals vs conservatives, showing conservatives are less constrained. (B) BIC curves for GMM clustering showing liberals need more clusters than conservatives (opposite of coalition prediction).

**Reader takeaway:** Conservative modularity reflects individual-level independence of belief domains, not factional subgroups. Each conservative holds a more idiosyncratic combination of positions.

**Source figures:** `sound_06_constraint.png`, `sound_06_clustering_bic.png` (combine into panels)

---

## Discussion Points

### For polarization theory
- Standard polarization measures (thermometer gaps, issue positions) capture *attitudinal* polarization. Structural divergence is a distinct phenomenon: the two groups could hold identical distributions on every issue and still have diverging associational architectures. We are measuring a different thing.
- The 48-year trend suggests structural divergence is a deeper, slower process than the attitudinal polarization spikes tied to specific political events.
- The centrality divergence, while statistically marginal under the strictest correction (non-overlapping p=0.054), identifies a specific structural mechanism: when the population-level dependency structures for two groups are organized around different issues (morality/policing for one group, different domains for the other), the groups inhabit different associational landscapes even when they agree on the sign of specific relationships.

### For political science
- The finding that morality and policing variables (PORNLAW, POLHITOK, POLMURDR) are the fastest-diverging hubs connects to research on moral foundations (Haidt 2012) and suggests that the structural divergence is rooted in attitudes toward authority and social regulation rather than partisan identity per se.
- The "big tent" finding (conservative heterogeneity = individual-level, not factional) challenges standard typology approaches (Pew's political typologies, etc.) that assume discrete subgroups. The data support a continuous heterogeneity model. This has implications for theories of party management: conservative coalition heterogeneity is a diversity-management problem, not a factional-bargaining problem.
- The denser liberal network may relate to asymmetric enforcement of ideological consistency within the two coalitions.

### Candidate mechanisms and temporal constraints
We document the pattern of structural divergence but do not identify its causal mechanism — a task that would require panel data or experimental designs beyond the scope of this study. However, the temporal trajectory constrains the space of plausible mechanisms. The divergence trend has been continuous for at least 48 years (1974-2022), with a possible acceleration around 2008. This timeline rules out a purely digital explanation: the trend predates social media, the internet, and even cable news. It is more consistent with slower processes — elite cue-giving that packages beliefs differently for the two groups (Zaller 1992; McCarty et al. 2006), associative diffusion through increasingly segregated social networks (Goldberg & Stein 2018), or generational replacement of cohorts with different formative political environments. The post-2008 acceleration, if confirmed, could reflect the compounding of these slower processes with the amplifying effects of partisan media ecosystems (Prior 2007). Disentangling these mechanisms is a priority for future work.

### For population-level associational structure
- Near-perfect structural balance (99% balanced triads) across the full network, significantly exceeding all three null models including the most stringent signed configuration model (null mean 50.5%, p<0.001), indicates genuine constraint structure in the population-level associational landscape — reflecting some combination of individual cognitive consistency, social influence, and elite cue-giving.
- The stability of community structure (NMI ~0.946) despite changing content suggests that these domain groupings are robust features of the population-level associational landscape, potentially reflecting shared cognitive categories, stable social cleavages, or enduring elite issue-framing — while the *importance* of each domain is responsive to political context.
- The independence finding (conservative lower constraint at the population level) raises questions about the sources of differential associational structure — whether driven by individual cognitive style, heterogeneous elite cues, or weaker social influence networks within the conservative population.

### For methodology
- Regularized partial correlations (graphical LASSO) serve a dual purpose: they handle multicollinearity across 120+ variables, and they produce sparse graphs where graph-theoretic metrics (centrality, clustering, balance) are well-defined. However, the headline divergence finding is robust to method choice — raw Pearson correlations produce the same trajectory (r=0.910 agreement). The LASSO's value is interpretability and parsimony, not revelation of hidden structure.
- The paper is explicit about which findings require the graph representation (community structure, bridge identification, structural balance) and which could be obtained from simpler matrix or vector comparisons (divergence trend, individual heterogeneity). This transparency strengthens rather than weakens the argument.
- All temporal trend p-values use Newey-West HAC standard errors to correct for serial autocorrelation from overlapping windows, with non-overlapping windows (step=4) as the primary specification.
- Per-variable centrality results use Benjamini-Hochberg FDR correction (62 tests).
- Sample-size matching at every comparison point prevents density artifacts from contaminating structural comparisons.
- The per-window permutation + bootstrap framework (1000 iterations each, convergence verified at 20,000) provides rigorous significance testing without parametric assumptions.

---

## Robustness Summary

| Check | Source | Key Result | Verdict |
|-------|--------|-----------|---------|
| Fixed variables (64, intersection) | sound_07 | slope=0.0044 (44% of original — variable-count inflation removed) | PASS |
| Alpha sensitivity (0.1-0.3) | sound_07 | 5/5 alphas significant | PASS |
| Raw Pearson agreement | sound_07, sound_10 | r=0.910 trajectory agreement; raw centrality divergence stronger | PASS |
| POLVIEWS/PARTYID excluded | sound_07 | Divergence survives (p=0.031 OLS; p=0.021 non-overlap HAC) | PASS |
| HAC-corrected p-values | sound_08 | Euclidean distance: p<0.001; centrality rho: p<0.001 (HAC) | PASS |
| Non-overlapping windows | sound_08 | Euclidean: p=0.021; centrality rho: p=0.054 (marginal) | PASS (Euc.) / MARGINAL (rho) |
| FDR-corrected hub variables | sound_08 | 4 variables survive FDR q<0.05 (POLHITOK, POLMURDR, SUICIDE1, POLABUSE) | PASS |
| Structural break | sound_08 | Acceleration at ~2008 for Euclidean distance; none for centrality rho | INFO |
| Per-window permutation | sound_09 | 22/22 windows significant at p<0.05 | PASS |
| Convergence test (20K iterations) | sound_09b | Stable from N=1000 | PASS |
| Exclude 2021-2022 (mode switch) | sound_11 | slope=0.0028, p=0.0025 | PASS |
| PARTYID alternative split | sound_11 | slope=0.0039, p=7.56e-04 | CONVERGENT |
| Survey weights (WTSSALL) | sound_11 | r(weighted,unweighted) > 0.99 | PASS |
| Eigenvalue audit | sound_11 | min_eig=0.0205, no negative eigenvalues | PASS |
| Balance null models (3 nulls) | sound_12 | 99% vs 50.5-51.1% nulls, all p<0.001 | PASS |

---

## Supplementary Material (brief notes for later)

- Full variable list with descriptions, coding, and community assignments
- Sensitivity analyses: alpha parameter for regularization, window size, matching method
- All 10 community profiles at the reference period
- Edge-level difference tables (60 significant edges)
- Domain decomposition details and temporal trajectories
- With-POLVIEWS/PARTYID centrality results (POLVIEWS centrality trajectory, GRASS trajectory)
- Robustness: alternative community detection algorithms, alternative centrality measures
- Additional centrality-disagreement tables for each time period
- Raw Pearson vs LASSO trajectory comparison (sound_10)
- POLVIEWS composition over time (sound_11)
- Eigenvalue audit details (sound_11)
