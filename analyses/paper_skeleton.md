# Paper Skeleton: Structural Divergence of Liberal and Conservative Belief Networks

## Title

**The Reorganization of American Ideology: How Liberals and Conservatives Increasingly Disagree on Which Beliefs Matter, 1974-2022**

## Thesis (one paragraph)

Using five decades of General Social Survey data (1974-2022, N=72,390), we document *structural divergence* — the increasing dissimilarity between the population-level associational structures of self-identified liberal and conservative Americans. We estimate the conditional dependency structure among ~120 attitude variables using regularized partial correlations (graphical LASSO), which separates direct conditional dependencies from indirect associations. While the divergence trend itself is robust to method choice (also appearing in raw correlations, with r=0.91 trajectory agreement), the sparse conditional dependency graph provides interpretive structure — communities, bridges, and structural balance — that characterizes the nature of the divergence. The Euclidean distance between group-specific partial correlation matrices increases at +0.0044 per year (non-overlapping windows, p=0.021; HAC-corrected overlapping, p<0.001), driven by reorganization rather than reversal (sign disagreements ~0) and primarily within-domain (73%). The fastest-rising hub variable is political self-identification (POLVIEWS, +0.004/yr), indicating that political identity is becoming increasingly central to the organization of the broader belief system — the very definition of ideological sorting at the structural level. When POLVIEWS and PARTYID are excluded as a robustness check, the divergence survives and the fastest-diverging hubs become morality and policing items — attitudes toward police use of force (POLHITOK, POLMURDR) and confidence in clergy (CONCLERG) — substantive attitude variables, not partisan markers. This paper makes two contributions. First, we show that structural divergence has been growing continuously for 48 years, with a possible acceleration around 2008, predating the internet era and consistent with slow-moving processes like elite polarization and associative diffusion. Second, we show that conservative belief networks are more modular than liberal ones — not because the conservative coalition consists of distinct factions, but because individual conservatives hold beliefs more independently across domains. These findings reframe polarization as a structural phenomenon: Americans are not just moving to opposite poles on shared issues — the population-level dependency structures for the two groups are increasingly organized around different hub variables.

---

## Related Work

### The constraint tradition
Converse (1964) introduced "constraint" as the degree to which positions on one issue predict positions on another, arguing that mass publics show far less constraint than elites. Subsequent work has debated whether constraint has increased. Baldassarri and Gelman (2008) showed that issue alignment increased among partisans but not the general public, using pairwise correlations between issue positions. Our work extends this tradition by modeling conditional dependencies rather than pairwise correlations, comparing the *structure* of these dependencies between groups rather than their average strength, and tracking structural evolution over 48 years.

### Belief network methods
Boutyline and Vaisey (2017) introduced belief network analysis to sociology, applying community detection to correlational structure among GSS attitudes. Their approach — treating the pattern of inter-attitude correlations as a network — is the methodological foundation for both DellaPosta (2020) and our work. We extend it by comparing group-specific networks over time. Brandt and Sleegers (2021) provided an important methodological critique, warning that population-level correlations should not be interpreted as individual cognitive architecture — a caveat we take seriously throughout (see Methods caveat).

### Pluralistic collapse
DellaPosta (2020) demonstrated that American beliefs are consolidating into fewer, more encompassing modules — a process he terms "pluralistic collapse." Using 219 GSS items and walktrap community detection, he showed that belief networks have fewer modules, higher modularity, and greater concentration over 1972-2016, even after removing or controlling for political ideology and party identification. His "oil spill" model describes polarization as the spread of alignment across previously unrelated issue domains.

Our contribution is complementary but methodologically distinct. DellaPosta analyzed one network (the whole population) and asked whether it is consolidating; we analyze two group-specific networks (liberal, conservative) and ask whether they are consolidating *differently*. His framework cannot detect centrality divergence — the finding that the two groups increasingly disagree on which beliefs are structurally central — because he does not compare group-specific structures. His metrics (module count, modularity, concentration) capture whole-network properties; ours (Euclidean distance between matrices, centrality rank correlation, hub migration) capture between-group structural dissimilarity. Where DellaPosta showed beliefs are becoming more packaged, we show they are becoming packaged around different hubs.

The methodological difference is also substantive. DellaPosta uses zero-order pairwise Pearson correlations (absolute value), producing a dense, fully connected weighted network. His "ideology-controlled" condition partials out only POLVIEWS and PARTYID — a two-variable partial correlation that avoids matrix inversion entirely. This approach can show which beliefs *cluster together* (total association, including direct, mediated, and confounded pathways), but cannot distinguish direct from indirect associations. Our use of regularized partial correlations (graphical LASSO) estimates the conditional dependency structure — each edge represents a direct association between two beliefs after controlling for all others. This supports a stronger claim: not just that beliefs cluster differently for the two groups, but that the *architecture of direct dependencies* differs. The tradeoff is that conditioning on all other variables simultaneously assumes confounding dominates mediation across the variable set and risks collider bias (see Methodological Argument below). We validate this choice by showing the divergence trajectory is robust to method — raw Pearson correlations produce the same trajectory (r=0.910 agreement) — confirming the findings are not artifacts of the partial correlation framework.

### Sorting, alignment, and polarization
Mason (2018) distinguished sorting (alignment of social identities with party) from polarization (movement to extreme positions). Our concept of "structural divergence" is distinct from both: two groups can hold identical position distributions on every issue and still exhibit structural divergence if the population-level dependency structures — which issues co-vary with which other issues — differ between groups. Kozlowski and Murphy (2021) documented accelerating issue alignment post-2004; our structural break analysis identifies a similar inflection point around 2008 for structural divergence. Baldassarri and Goldberg (2014) showed individual-level heterogeneity in belief organization; our Section 5 finding — that conservative modularity reflects individual independence, not factional coalition structure — is the population-level signature of this individual-level pattern.

### Mechanisms
Goldberg and Stein (2018) proposed associative diffusion as a mechanism for belief spread through association networks. If liberals and conservatives inhabit networks with different seed issues, associative diffusion would naturally produce diverging correlation structures — making their model a plausible mechanism for the structural divergence we document. Fishman and Davis (2022) demonstrated temporal belief network analysis in AJPS, providing a methodological precedent for our approach.

---

## Concept Definition: Structural Divergence

*Structural divergence* is the increasing dissimilarity between the conditional dependency structures of ideologically defined subpopulations. It is measured as the Euclidean distance between group-specific partial correlation matrices (primary metric) and as the declining Spearman rank correlation between group-specific centrality hierarchies (secondary metric). Structural divergence is distinct from attitudinal polarization (movement to opposite poles on shared issues), affective polarization (increasing dislike of outgroups), and sorting (alignment of social identities with partisan identity). Two groups can hold identical position distributions on every issue and still exhibit structural divergence if they organize the relationships among those positions differently. The concept captures how the population-level associational architecture — which issues co-vary with which other issues — differs between groups and changes over time.

---

## Methodological Argument: Why Conditional Dependency Structure (and When It Matters)

### Why partial correlations

We use regularized partial correlations rather than zero-order (bivariate) correlations because we are making a *structural* claim — that the architecture of direct belief-to-belief associations differs between liberals and conservatives, and that this architecture is diverging. Zero-order correlations cannot support this claim: every edge in a zero-order correlation network is an undecomposed mixture of direct effects, indirect/mediated effects, and confounded associations. In attitude data, where a few latent dimensions (moral traditionalism, partisan identity) drive correlations across dozens of variables, a zero-order network is near-complete — everything correlates with everything — yielding a dense weighted graph with no meaningful topology. This is effectively a visualization of factor loadings, not a network. Partial correlations strip away shared latent structure and ask whether two variables have a *specific* association beyond what is explained by all other variables. This produces a graph with meaningful edges, meaningful absences, and interpretable topology (communities, hubs, bridges, structural balance).

The tradeoff is real. Conditioning on all other variables simultaneously makes three assumptions we must acknowledge: (1) that confounding dominates mediation across the variable set — i.e., conditioning removes more spurious signal than real causal pathways; (2) that collider bias from blind conditioning on ~120 variables is not dominant — we have no causal DAG to identify which variables are colliders; and (3) that the conditional independence structure is meaningful for belief systems — i.e., beliefs have direct associations, not just shared latent factors. These assumptions are not testable within our framework. We address them by showing that the core findings are robust to method choice: raw Pearson correlations produce the same divergence trajectory (r=0.910 agreement), confirming that the findings do not depend on whether these assumptions hold.

### POLVIEWS and PARTYID as network nodes

We split respondents into liberal and conservative subgroups using POLVIEWS (self-reported political ideology) and include POLVIEWS and PARTYID as nodes in the resulting subgroup networks. A potential concern is circularity: splitting on ideology and then finding ideology is structurally central. This concern is misplaced. The split defines the population of interest; computing correlations within that population is a valid operation on a well-defined subgroup. Within-group variation in POLVIEWS — the difference between "slightly liberal" and "extremely liberal" — is the relevant variation, not a restriction artifact. The results are not predetermined: POLVIEWS could be central or peripheral within either group, and this is an empirical question. The finding that POLVIEWS centrality is *rising* over time actually works against any mechanical effect of range restriction, which if anything attenuates within-group correlations. As a robustness check, we repeat all analyses with POLVIEWS and PARTYID excluded; the core findings survive, with top centrality gainers shifting to CONCLERG, POLHITOK, and POLMURDR. The one legitimate concern is compositional: if the within-group distribution of POLVIEWS shifts over time (e.g., more respondents identifying as "very liberal"), within-group variance changes, which could affect centrality trends. This is a standard confound about changing group composition, not a circularity problem, and applies equally to every variable whose distribution shifts over time.

### Two levels of analysis

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
- POLVIEWS is the fastest-rising hub variable (+0.004/yr), indicating that political self-identification is becoming increasingly central to the organization of the broader belief system. GRASS (marijuana legalization) is the fastest-declining (-0.003/yr), consistent with consensus formation on resolved culture-war issues. *[Graph-dependent]* (sound_05)
- When POLVIEWS and PARTYID are excluded (robustness check), the fastest-diverging hub variables become CONCLERG (confidence in clergy, +0.0023/yr), POLHITOK (police hitting citizens), and POLMURDR (police striking murder suspects) — substantive morality and authority variables, not partisan identity markers. *[Graph-dependent]* (sound_05b, sound_07)
- Community structure is highly stable: NMI ~0.946 (total). The *containers* for belief stay constant; the *importance* of each container changes. *[Graph-dependent]* (sound_05)

**Implication:** Structural divergence is not about mirror-image opposition. Both groups agree on the *sign* of relationships (e.g., religiosity and social conservatism are positively correlated for everyone). What differs — and what is diverging — is which issues are most central in the population-level dependency structure. The rise of POLVIEWS as the dominant hub is the structural signature of ideological sorting: political identity is increasingly the organizing variable around which other beliefs are arranged. Simultaneously, resolved culture-war issues (marijuana, homosexuality tolerance) are losing centrality as public opinion converges. When POLVIEWS is excluded, the divergence is driven by morality and authority variables (police use of force, confidence in clergy), suggesting that the structural reorganization extends beyond partisan identity into substantive attitude domains.

**Figure role:** Two panels — centrality rank correlation between groups over time, and hub-divergence plot showing which variables' centrality is changing fastest (with POLVIEWS-excluded version as supplementary).

---

### Section 5: Two organizational logics — liberal integration vs conservative independence

**Claim:** Liberal and conservative belief systems use fundamentally different architectural principles. The liberal network is organized through *cross-domain integration* — dense bridges connecting otherwise separate attitude domains around a common framework. The conservative network is organized through *within-domain consolidation* — tighter internal structure within domains but fewer bridges between them. These differences are not just quantitative (more/fewer edges) but qualitative (different organizing principles), and they are explained by different individual-level patterns.

**Note on method:** This section combines network-level structural comparison (sound_13) with individual-level analysis (sound_06). The network comparison identifies *what* differs architecturally; the individual-level analysis explains *why*.

**Evidence — Network architecture (sound_13):**

*Liberal network = cross-domain integrator:*
- 81% of liberal-only edges are between-domain. Liberal density advantage comes almost entirely from cross-domain bridges, not tighter within-domain connections. *[Graph property]* (sound_13)
- Biggest liberal bridge concentrations: Civil liberties ↔ Morality/family (22 lib-only edges), Abortion ↔ Civil liberties (15), Abortion ↔ Morality/family (12), Morality ↔ Religion (8). These domains share an autonomy/rights thread. *[Graph property]* (sound_13)
- Liberal bridge variables: WRKWAYUP (meritocracy belief, betweenness rank 1), OBEY (child-rearing, rank 2), COLMSLM (Muslim civil liberties, rank 3), CONCLERG (confidence in clergy, rank 6). These connect social values to political attitudes. *[Graph property]* (sound_13)
- Liberals keep morality (15 members), abortion (10 members), and spending (6 members) as separate communities connected by bridges. *[Graph property]* (sound_13)

*Conservative network = within-domain consolidator:*
- Conservative-specific density is concentrated within domains: Spending (+0.117 denser), Abortion (+0.133), Institutions (+0.091), Political (+0.063) are all denser within-domain for conservatives. *[Graph property]* (sound_13)
- Conservative-specific between-domain connections are narrower: Political ↔ Spending (24 con-only edges) and Police ↔ Political (11 con-only) — fiscal views and law enforcement link to political identity. *[Graph property]* (sound_13)
- Conservatives merge morality items (HOMOSEX, GRASS, PORNLAW, PREMARSX, TEENSEX, XMARSEX) into the abortion cluster (17 members). Spending variables absorb into the political mega-cluster (31 members). Fewer, larger domains instead of many connected smaller ones. *[Graph property]* (sound_13)
- Conservative bridge variables: SPKCOM (free speech, betweenness rank 1), POLHITOK (police force, rank 2), SUICIDE1 (right to die, rank 3). These connect authority/enforcement domains. *[Graph property]* (sound_13)

**Evidence — Individual-level explanations (sound_06):**

*Why the conservative network is modular — independence, not factions:*
- Cross-domain correlations weaker for conservatives (mean |r| = 0.068 vs 0.075). *[Vector-level]* (sound_06)
- Belief constraint (PC1 R-squared) lower for conservatives: 0.184 vs 0.216, p < 0.001. *[Vector-level]* (sound_06)
- GMM clustering: conservatives need *fewer* clusters (k=3 vs k=5) — opposite of the coalition prediction. *[Vector-level]* (sound_06)
- Distributions wider and flatter (not bimodal) in 5/9 domains — continuous heterogeneity, not discrete factions. *[Vector-level]* (sound_06)
- Conclusion: conservative modularity reflects individual independence across domains, not factional coalition structure.

*Why the liberal network is dense — coherence, not conformity:*
- The analogous interpretive question: does liberal cross-domain density reflect genuine ideological coherence (a unifying framework that logically connects domains) or social enforcement of purity (pressure to hold "correct" positions across all domains)?
- Higher individual-level constraint (R²=0.216 vs 0.184) is consistent with both interpretations. *[Vector-level]* (sound_06)
- But liberals need *more* GMM subtypes (k=5 vs k=3). If conformity dominated, we would expect fewer subtypes (everyone converging on one package), not more. Multiple subtypes with high constraint suggests there are several valid ways to be a coherently integrated liberal. *[Vector-level]* (sound_06)
- The cross-domain bridges concentrate along a substantively interpretable axis: civil liberties ↔ morality ↔ abortion ↔ religion — domains connected by personal autonomy and bodily rights. This looks like ideological structure, not random social pressure. *[Graph property]* (sound_13)
- Conclusion: liberal density more consistent with genuine ideological coherence around a rights/autonomy framework than with uniform conformity pressure.

**Implication:** The two groups maintain viable but structurally different belief systems. Liberal *integration* means belief change can propagate across domains — shifting on one morality issue is structurally connected to civil liberties, abortion, and religion through dense bridges. Conservative *independence* means domain boundaries are stronger barriers — shifting on a spending issue does not propagate easily to morality views because fewer bridges connect them. This asymmetry has implications for political dynamics: the liberal architecture may produce faster ideological updating (cascade through bridges) but also more vulnerability to "purity tests" (everything is connected, so deviation on one domain is visible from others). The conservative architecture may produce slower updating but more tolerance for within-coalition disagreement (domains are insulated from each other). The "big tent" metaphor is structurally accurate: individual conservatives hold more idiosyncratic cross-domain belief profiles, producing a coalition that is held together not by ideological coherence but by shared domain-specific commitments (e.g., fiscal conservatism, moral traditionalism) that need not co-occur within any individual.

**Figure role:** (A) Domain-pair connectivity heatmaps showing liberal vs conservative density patterns and the difference matrix. (B) Belief constraint distributions (lib vs con). (C) Hub/bridge comparison showing different organizing variables for each group.

---

## Evidence Table

| # | Claim | Analysis | Key Numbers | Method Level | Figure |
|---|-------|----------|-------------|--------------|--------|
| 1a | Ideology is multi-dimensional | sound_01 | PC1=10%, 27 components for 50% | Matrix (PCA) | Fig 1 |
| 1b | Sparse dependency structure has interpretable topology | sound_01, sound_12 | 370 edges, 10 communities, 99% balance (all 3 nulls p<0.001, incl. signed config null=50.5%) | Graph | Fig 1 |
| 2 | Lib/con dependency structures are statistically distinguishable | sound_02, sound_09 | Z=6.81 (p<0.001), 22/22 windows sig., 405 vs 356 edges, clustering 0.369 vs 0.421 | Matrix + Graph | Fig 2 |
| 3 | Structural divergence has been growing for 48 years | sound_07, sound_08, sound_10 | slope=+0.0044/yr (non-overlap p=0.021, HAC p<0.001), raw Pearson r=0.910 agreement, break at ~2008 | Matrix (+ Graph for density) | Fig 3 |
| 4a | Divergence is magnitude, not direction | sound_04 | Sign disagreements ~0, dimensionality parallel, within-domain 73% | Matrix | Fig 4 |
| 4b | POLVIEWS rising as dominant hub; morality/authority variables rise when POLVIEWS excluded | sound_05, sound_05b, sound_08 | POLVIEWS +0.004/yr, GRASS -0.003/yr; excluded: CONCLERG +0.0023/yr, POLHITOK, POLMURDR; centrality rho: HAC p<0.001 / non-overlap p=0.054 | Graph (essential) | Fig 4 |
| 5a | Liberal integration: 81% of lib-only edges are between-domain; bridges concentrated in autonomy/rights domains | sound_13 | 186 lib-only edges (81% between-domain), 22 civil-lib↔morality, 15 abortion↔civil-lib | Graph | Fig 5 |
| 5b | Conservative consolidation: denser within-domain, mega-clusters absorb spending into politics & morality into abortion | sound_13 | 137 con-only edges (61% between-domain), con denser within Political/Spending/Abortion/Institutions | Graph | Fig 5 |
| 5c | Conservative modularity = individual independence, not factions; liberal density = coherence, not conformity | sound_06 | Constraint: 0.184 vs 0.216 (p<0.001), GMM k: 3 vs 5, var ratio 1.10-1.37 | Vector | Fig 5 |

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

**What it shows:** (A) Centrality rank correlation (Spearman rho) between lib and con networks over time, with HAC-corrected trend line. (B) Top centrality gainers and losers over time: POLVIEWS rising fastest (+0.004/yr), GRASS declining fastest (-0.003/yr). (C) POLVIEWS/PARTYID-excluded robustness: top gainers shift to CONCLERG, POLHITOK, POLMURDR — substantive attitude variables, not partisan markers. (D, optional) Sign disagreements near zero across all windows.

**Reader takeaway:** Political identity is becoming the dominant organizing variable in the belief system, while resolved culture-war issues lose centrality. Even without POLVIEWS, the reorganization is driven by morality and authority variables — the structural divergence extends beyond partisan identity.

**Source figures:** `sound_04_centrality_rank.png`, `sound_05_centrality_gainers_losers.png`, `sound_05b_centrality_gainers_losers.png`

---

### Figure 5: Two Organizational Logics — Integration vs Independence

**What it shows:** (A) Domain-pair connectivity heatmaps (liberal, conservative, difference) showing where each group's edges concentrate — liberals bridge domains, conservatives consolidate within them. (B) Distribution of individual-level belief constraint (PC1 R-squared) for liberals vs conservatives, showing conservatives are less constrained. (C) Hub/bridge comparison: top bridge variables differ completely between groups (liberal bridges connect social values to politics; conservative bridges connect authority/enforcement domains).

**Reader takeaway:** The two groups organize their belief systems around different architectural principles. Liberals integrate across domains through bridges anchored in autonomy/rights; conservatives consolidate within domains. At the individual level, conservative modularity reflects independence (not factions) while liberal density reflects coherence (not conformity).

**Source figures:** `sound_13_domain_connectivity.png`, `sound_06_constraint.png`, `sound_13_bridge_comparison.png` (combine into panels)

---

## Discussion Points

### For polarization theory
- Standard polarization measures (thermometer gaps, issue positions) capture *attitudinal* polarization. Structural divergence is a distinct phenomenon: the two groups could hold identical distributions on every issue and still have diverging associational architectures. We are measuring a different thing.
- The 48-year trend suggests structural divergence is a deeper, slower process than the attitudinal polarization spikes tied to specific political events.
- The centrality divergence, while statistically marginal under the strictest correction (non-overlapping p=0.054), identifies a specific structural mechanism: political identity (POLVIEWS) is becoming the dominant organizing hub in the belief system, structurally linking domains that were previously more independent. This is the structural signature of ideological sorting (Mason 2018) — not just that identities align with party, but that political identity increasingly *organizes* the conditional dependency structure among all other beliefs. When POLVIEWS is excluded, the divergence persists and is driven by morality and authority variables, indicating the reorganization extends beyond partisan identity.

### For political science
- The finding that POLVIEWS is the fastest-rising hub connects directly to the sorting literature (Mason 2018): political identity is becoming the structural backbone of the belief system. But the POLVIEWS-excluded robustness check reveals a deeper layer — morality and authority variables (POLHITOK, POLMURDR, CONCLERG) are independently rising in centrality, connecting to research on moral foundations (Haidt 2012) and suggesting the structural reorganization extends into substantive attitude domains beyond partisan identity.
- The two organizational logics have distinct political implications. Liberal *integration* (dense cross-domain bridges) means belief change can cascade across domains — shifting on one morality issue is structurally connected to civil liberties, abortion, and religion. This may produce faster ideological updating but also more vulnerability to "purity tests" where deviation on any domain is visible from all others. Conservative *independence* (strong domain boundaries, fewer bridges) means domains are insulated — shifting on spending does not propagate to morality views. This produces more tolerance for within-coalition disagreement but slower collective updating.
- The "big tent" finding (conservative heterogeneity = individual independence, not factions) challenges standard typology approaches (Pew's political typologies, etc.) that assume discrete subgroups. The data support a continuous heterogeneity model: conservative coalition coherence operates through shared domain-specific commitments (fiscal conservatism, moral traditionalism) that need not co-occur within any individual.
- Liberal cross-domain bridges concentrate along a substantively interpretable axis: civil liberties ↔ morality ↔ abortion ↔ religion, connected by personal autonomy and bodily rights. This suggests the liberal belief system is organized around a coherent ideological framework, not merely social conformity pressure — supported by the finding that liberals need *more* GMM subtypes (k=5 vs k=3), indicating multiple valid ways to be a coherently integrated liberal.

### Candidate mechanisms and temporal constraints
We document the pattern of structural divergence but do not identify its causal mechanism — a task that would require panel data or experimental designs beyond the scope of this study. However, the temporal trajectory constrains the space of plausible mechanisms. The divergence trend has been continuous for at least 48 years (1974-2022), with a possible acceleration around 2008. This timeline rules out a purely digital explanation: the trend predates social media, the internet, and even cable news. It is more consistent with slower processes — elite cue-giving that packages beliefs differently for the two groups (Zaller 1992; McCarty et al. 2006), associative diffusion through increasingly segregated social networks (Goldberg & Stein 2018), or generational replacement of cohorts with different formative political environments. The post-2008 acceleration, if confirmed, could reflect the compounding of these slower processes with the amplifying effects of partisan media ecosystems (Prior 2007). Disentangling these mechanisms is a priority for future work.

### For population-level associational structure
- Near-perfect structural balance (99% balanced triads) across the full network, significantly exceeding all three null models including the most stringent signed configuration model (null mean 50.5%, p<0.001), indicates genuine constraint structure in the population-level associational landscape — reflecting some combination of individual cognitive consistency, social influence, and elite cue-giving.
- The stability of community structure (NMI ~0.946) despite changing content suggests that these domain groupings are robust features of the population-level associational landscape, potentially reflecting shared cognitive categories, stable social cleavages, or enduring elite issue-framing — while the *importance* of each domain is responsive to political context.
- The independence finding (conservative lower constraint at the population level) raises questions about the sources of differential associational structure — whether driven by individual cognitive style, heterogeneous elite cues, or weaker social influence networks within the conservative population.

### For methodology
- We use regularized partial correlations because we are making a structural claim about direct belief-to-belief associations, not merely a clustering claim. Zero-order correlations (as in DellaPosta 2020) can show which beliefs move together but cannot distinguish direct from indirect associations; the resulting near-complete graph has no meaningful topology. Partial correlations produce a sparse graph with interpretable edges, absences, communities, and structural balance — supporting the kind of architectural analysis this paper contributes. The graphical LASSO is the regularization method that makes partial correlations computable given our data (the unregularized precision matrix is singular); the sparsity it produces is a byproduct of regularization, not the analytical goal.
- This choice requires acknowledging three assumptions: that confounding dominates mediation across the variable set, that collider bias from conditioning on ~120 variables is not dominant, and that conditional independence structure is meaningful for belief systems. These are not testable within our framework, but we address them through robustness: raw Pearson correlations produce the same divergence trajectory (r=0.910 agreement), confirming the findings do not depend on whether these assumptions hold.
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
- POLVIEWS/PARTYID-excluded centrality results (sound_05b: CONCLERG, POLHITOK, POLMURDR as top gainers)
- Robustness: alternative community detection algorithms, alternative centrality measures
- Additional centrality-disagreement tables for each time period
- Raw Pearson vs LASSO trajectory comparison (sound_10)
- POLVIEWS composition over time (sound_11)
- Eigenvalue audit details (sound_11)
