# Paper 1 — skeleton detail (parked)

> **Status: parked, not agreed.** This was written before the section structure was
> reviewed, and it treats as settled a set of decisions that are still open — see the
> "Open calls" list in `paper1-skeleton.md`, which is the live document. Kept because the
> content below (citation slots, the landing sites for P1.1–P1.3, the rival explanations,
> the worked caveats) is the next resolution down and will be wanted once the structure is
> agreed. Anything here that contradicts `paper1-skeleton.md` loses.

Built 2026-08-13 by merging `merged_outline.md` (Victor) with `paper1-narrative-draft.md`
(Tim + Claude). Comparison and reasoning: `notes/2026-08-12-victor-outline-comparison.md`.

Audience: practitioners, sociologists, theory-minded computational social scientists.
Not philosophers of mind. The piece should partly create its reader.

> **Citations below are slots, not references.** Details are from memory and none has been
> checked. `literature/` currently holds DellaPosta only. Verify before any of this reaches
> a bibliography.

---

## Settled decisions

Carried in so they are not relitigated.

1. **Ontology goes last, not third.** Victor's ordering. The consequence the paper needs
   from the ontology — that no network is platonic — is a *concession*, and a concession
   lands better after the reader has been paid than before.
2. **"What we did" is not a section.** Cut. It is method, not narrative.
3. **Polarization potential is filed under U2 (upstream), not U3.** It requires the
   association to be causal, directed and intervention-stable, which is an upstream claim.
   This explains argument 8's known weakness rather than adding to it.
4. **The shock test lives in Paper 2.** §3 points at it as the thing that would discharge
   the U2 debt; it does not run here.
5. **U3 is the engine, U1 and U2 are supporting** — and this is said out loud, because
   Victor read the draft as accidentally U3-only and he will not be the last.
6. **§1 is not a concession.** We have never advocated individual-level belief networks
   and we endorse the criticism of them. The charge is ours to press.
7. **Argument 5 moves out of §5 and into §3 as the worked U3 example.** It is already one
   of Victor's U3 bullets ("shaping responses to discourse"), it gives U3 a concrete
   demonstration, and it leaves §5 as a genuine coda.
8. **The separability criterion gets one paragraph in §2, not a section and not the spine.**
   A network claim is about a *system* only if the quantity it rests on fails to decompose
   into a sum over pairs. Worth stating, because a large share of the field's output is
   separable; worth applying to ourselves, because our cross-sectional lib/con comparison
   is largely structural (clustering, triangles, communities, betweenness, domain
   connectivity) while our temporal divergence, measured as Frobenius distance, is not.

---

## The spine, in one paragraph

Correlational belief networks cannot tell us how individuals think. They lack causal
structure and they lack individual resolution, and these are two separate failures, not
one. This criticism is ours — we are not defending belief networks against it, we are
pressing it — because the usual response, which is to note the limitation and then carry on
in individual-level language anyway, mistakes a category error for a measurement problem.
Population co-occurrence is not a contaminated proxy for individual structure. It is an
accurate measurement of a different thing: the pattern of which beliefs occur together
across a population. That pattern is worth three distinct things. It predicts. It
constrains what can be inferred about the causes upstream of it. And it exerts causal force
of its own — which is this paper's engine. Studying it costs remarkably little: beliefs must
persist and be objective, but they need not be discrete entities. That cheapness has a
price, and we pay it in full. Without discreteness the network exists only relative to the
questions asked, so no absolute figure means anything and every claim must be comparative.
But comparison does two different jobs, and only one of them is repaired by going
comparative. What survives is a claim with real stakes: two populations can hold nearly
identical beliefs and still be arranged to move in opposite directions. Agreement is not
stability.

---

## Manuscript layout

§1–§5 below are the paper's **argument moves**, not its physical sections. This is the map
between them.

| Manuscript element | Content | Venue-dependent? |
|---|---|---|
| **Title** | Candidate: *Agreement Is Not Stability*. Needs a subtitle naming the object — belief networks and what they measure. | style only |
| **Abstract** | Written last, from the spine paragraph. | length, structured or not |
| **Significance statement** | PNAS carries one; the others may not. **Verify.** | yes — may not exist |
| **1. Introduction** | **= §1, in full.** In a piece this length the problem *is* the introduction; there is no separate scene-setting section before it. Needs one move §1 currently lacks — see 1.7. | length |
| **2. The object** | = §2 | — |
| **3. What it is good for** | = §3. Longest section by design. | length |
| **4. What it assumes, and what that costs** | = §4 | — |
| **5. What follows** | = §5, functioning as the conclusion. A generative piece ends by opening questions, so 5.3 *is* the conclusion rather than preceding one. | — |
| **References** | Four must-engage papers still missing from `literature/`. | style, cap |
| **Figures and boxes** | See below. | allowance |

**Figures and boxes.**

- **Figure 1 — the carving demo** (§3.1). Centrality and modularity rankings swinging across
  resampled variable subsets. The paper's only piece of new analysis, and it does double duty
  as the evidence for §4.4.
- **Box 1 — the shuffle test** (§2.4). Is your claim about the system or about a pile of
  correlations?
- **Box 2 — the deletion test** (§4.6). Is your comparison calibration or object?
- The two boxes are the paper's takeaway tools and are what a reader will photograph. Worth
  designing them to be liftable out of context.
- Possible **Figure 2** — the liberal and conservative networks side by side — but only if
  it carries a structural claim. A pretty pair of graph layouts is exactly the decoration
  §2.4 warns about, and including one would undercut the argument.

**What the venue decision actually changes.** PNAS and the Royal Society journals are the
same shape at different sizes, so the layout above holds and only the budget moves. **BBS is
a different architecture** — a target article is far longer and built to expose surface for
open commentary, which would mean more numbered subsections, each claim stated separately
enough to be attacked on its own, and the open questions promoted rather than saved for the
end. Deciding BBS is deciding to rebuild, not to resize. (OQ 006.)

---

# §1 — The problem

**Job.** State the criticism in our own voice. We have never advocated individual-level
belief networks, so nothing here is a concession — the charge is ours to press, and we
press it harder than the people it is aimed at usually do.
**Claim.** Correlational networks fail as descriptions of individual cognition, for two
independent reasons, and the failure is fatal rather than awkward.

### 1.1 — Open inside the practitioner's situation

- Open on a single number, not a literature review. Two GSS items correlate at some value
  in the pooled sample. **TODO: pull a real pair and figure from `sound_01`** — it should
  be a pair whose association feels obvious, so the reader supplies the causal story
  unprompted and then watches it dissolve.
- Give the list of things that could have produced that number, deliberately unranked: one
  person reasoning from a shared premise; both beliefs taught by the same church; both
  signalled by the same party; both common in the same region or birth cohort; a survey
  artifact from shared wording or response format.
- The point of the list is that nothing in the correlation distinguishes them, and no
  amount of care in estimating the correlation will.
- One paragraph. The reader recognises the situation immediately and the paper buys speed
  by not belabouring it.

### 1.2 — Deficit one: no causal structure

- A cross-section identifies association, not direction and not mechanism.
- The standard fixes do not fix it. Larger *N* sharpens the estimate without changing what
  is being estimated. Controls remove confounds only if you already know what they are.
  Partial correlation conditions on the measured set and inherits whatever that set omits.
- State the honest limit: with no causal DAG we cannot separate confounding from mediation,
  and conditioning blindly on ~120 variables can open collider paths as readily as it
  closes confounding ones.
- **flag:** this is the assumption already recorded in `MEMORY.md` — that confounding
  dominates mediation across the variable set. It should appear in the paper as a stated
  assumption rather than an implicit one.

### 1.3 — Deficit two: no individual resolution

- The measurement is a property of a population. Nothing in it licenses a statement about
  any particular person.
- Sharpen with a concrete impossibility rather than a slogan: one correlation value is
  compatible with a minority holding the pair tightly and everyone else not at all, and
  equally compatible with everyone holding it weakly. Those are different worlds and the
  correlation does not distinguish them.
- → **cite:** Robinson (1950), ecological correlation. *Candidate, unverified.*
- Note this is not repaired by better estimation either. It is a property of the level at
  which the measurement is taken.

### 1.4 — The two deficits are independent — the hinge of the section

- The load-bearing move, and Victor's correction to our earlier draft, which collapsed them
  into one implication.
- Hand someone the complete population-level causal structure — every edge oriented, every
  mechanism named. They still cannot say whether a given cause runs through a head or
  through an environment, nor that it operates in any particular person.
- Run it the other way: perfect individual resolution without causal structure still leaves
  direction and mechanism wide open.
- Therefore any repair aimed at one leaves the other untouched.
- The payoff, and the reason this move earns its place: the field's caveat paragraphs
  almost always address 1.2, and then the results prose commits 1.3. Naming the
  independence is what exposes that.

### 1.5 — Name the standard response and refuse it

- The habit, described precisely: a limitations paragraph acknowledging correlational data,
  followed by results prose in individual-level language — respondents who "link" X to Y,
  beliefs "organised around" Z in people's minds.
- The diagnosis: this treats a category error as a measurement problem. Measurement
  problems are fixed by better measurement. Category errors are not fixed at all, because
  the sentence is about the wrong thing.
- State the position without hedging: for individual-level inference the failure is fatal,
  not awkward.
- **flag:** tone stays diagnostic, not prosecutorial. The audience includes people who do
  this, and the piece is meant to recruit them, not to win against them.

### 1.6 — Position against the literature

- **Dalege et al., Causal Attitude Network** — the most explicit individual-level
  commitment. Engage as careful work on a mis-specified object. → *cite, unverified, not in
  `literature/`.*
- **Boutyline & Vaisey (2017)**, belief network analysis with centrality — the closest
  methodological neighbour. → *cite, unverified, not in `literature/`.*
- **DellaPosta (2020)**, *Pluralistic Collapse* — the closest substantive prior work.
  Zero-order pairwise, dense, unsigned; shows clustering rises. Our object is the
  architecture of direct associations, not the clustering. → *in `literature/`.*
- **Converse (1964)** — the origin of "constraint" and the reason the field measures
  correlations at all. → *cite, unverified.*
- One paragraph total. We are relocating an object, not scoring points.

### 1.7 — The contribution claim and the roadmap

The move §1 was missing. Everything above diagnoses; nothing yet says what the paper does
about it, and an introduction that only diagnoses reads as a complaint.

- **State the contribution in three sentences, plainly.** We relocate the object: population
  co-occurrence is not a degraded measurement of individual belief structure but an accurate
  measurement of a collective one. We set out what that object supports, what it costs to
  study, and the form its claims must take. And we supply two tests a reader can apply to
  their own work — one for whether a claim is about a system at all, one for what a
  comparison is doing.
- **Say what the paper is not.** Not a new estimator, not a new dataset, not a
  meta-analysis. A perspective piece that changes what existing measurements are
  measurements *of*.
- **Be honest about the empirical status in the introduction, not only at the end.** The
  downstream-causal-power claim (§3.3) is argued, not demonstrated. Saying so early costs
  little and buys the reader's trust for §3 and §4.
- **Roadmap in four sentences**, one per remaining section. Perspective readers skim; the
  roadmap is what they navigate by.
- **flag:** this is also where the reader learns whether they are the audience. Written well
  it recruits practitioners; written defensively it filters them out.

---

# §2 — The reframe

**Job.** Perform the inversion and define the object precisely enough to study. The field
treats population co-occurrence as a degraded measurement of individual structure; the move
is to deny that it is a measurement of individual structure at all.
**Claim.** What the field reads as noise about cognition is signal about a collective, and
it is measured accurately.

### 2.1 — The inversion

- The sentence the section turns on: what ruins the individual-level reading is exactly what
  makes the collective-level reading work.
- Every source in 1.1's list — church, party, region, cohort, media — is noise if you want
  cognition and is *constitutive* if you want to know how beliefs are distributed in a
  population.
- So this is not a consolation prize and not a weaker claim. It is a different object,
  measured well, rather than the intended object measured badly.
- **flag:** this sentence has to land or the paper does not work. Draft it several times.

### 2.2 — Define the object

- Nodes: belief categories as operationalised by survey items. Not beliefs in heads.
- Edges: population co-occurrence, signed, estimated on a shared instrument.
- Unit: the collective — the population the survey samples.
- Say plainly what the network is *of*: a population-and-instrument pair, never a population
  alone. §4.4 collects on this, so it must be stated here without apology.

### 2.3 — What an edge aggregates, without embarrassment

- In-brain association, social transmission, shared environment, demography, media diet.
- The move is to present this as the edge's *content* rather than its contamination. At this
  level of description the mixture is what we mean by the edge.
- Make the contrast with §1.1 explicit: the same list appeared there as the reason
  individual-level inference fails. Same list, opposite valence. That contrast is the paper
  in miniature and it is cheap to make visible.

### 2.4 — What makes this an object rather than a container

- Voice the objection directly: if every interesting claim is about individual correlations,
  the network is a filing cabinet and we are describing its contents.
- The criterion: a claim is about the *system* only if the quantity it rests on fails to
  decompose into a sum over pairs.
- The test, stated concretely: reassign the correlation values to random pairs, keeping the
  multiset identical. Density, mean edge weight and summed pairwise divergence come out
  unchanged. Clustering, community structure, betweenness and structural balance break.
- Apply it to ourselves in print. Our cross-sectional lib/con comparison is structural —
  clustering, triangles, Louvain communities with composition matching, betweenness,
  degree-rank hub hierarchy, domain-to-domain connectivity. Our divergence trend, being
  Frobenius distance, is not.
- The service: this sorts a literature that has no such test, and it explains why a lot of
  belief-network output reads as a rendered matrix.
- **flag:** a paragraph plus the test, no more. It is a tool, not the thesis. (Decision 8.)

### 2.5 — Dispatch the three confusions

- **Not individual belief systems.** The ecological fallacy, named and refused. One
  paragraph — §1 already did the work.
- **Not an objective structure possessed by a collective.** Reification. Note that §4.4 will
  show the structure is instrument-relative, so this is a consequence rather than a caution.
- **Not a societal brain.** Out of scope. One dismissive paragraph, no engagement with
  distributed cognition.
- **flag:** pace matters here. Three short paragraphs, then move.

### 2.6 — Name the three utilities and declare the weighting

- U1 prediction and passive inference; U2 upstream causal inference; U3 downstream causal
  force.
- State the weighting outright: **U3 is the engine, U1 and U2 support it.**
- Say why we are stating it: a reader who has to infer the weighting files it as an
  oversight. Victor did, on the draft.
- Forward-reference: §3 takes them in order, and §3.5 explains why the order is not
  arbitrary.

### 2.7 — The measurement precondition

- Every comparison in this paper assumes items mean the same thing to the populations being
  compared.
- If they do not, a measured network difference is a measurement artifact, and every
  argument in §3 and §5 is contaminated — not weakened, contaminated.
- Differential item functioning is how it would be tested. → **cite:** measurement
  invariance literature, candidate Meredith (1993). *Unverified.*
- Placed at the front as a precondition rather than at the back as a limitation. The
  placement is itself a claim about how seriously it should be taken.
- **flag:** OQ 004, untested in our data. Say so. Do not let the placement imply it is
  discharged.

---

# §3 — Why correlations matter

**Job.** Pay the reader. The generative core, and the longest section.
**Claim.** The population pattern supports three distinct kinds of knowledge — prediction,
constrained upstream inference, and downstream causal force — and the third means the
pattern acts in the world rather than merely describing it.

### 3.1 — U1, passive inference

- The claim: correlations are the *right* level of description here, not a proxy. If the
  question is what knowing one thing about a person tells you about the rest, the population
  pattern is the answer, not an approximation to it.
- Substantive readings: hubs as signalling nodes (high degree means learning this narrows
  the most); modules as informational neighbourhoods; density as constraint on how many
  independent dimensions of belief are actually available.
- **The dimensionality result belongs here and we have under-used it.** PC1 ≈ 10%, roughly
  27 components for half the variance. Non-separable, spectral, and it contradicts the
  one-dimensional ideology story directly.
- → **cite:** the standard low-dimensional claim, as contrast. Candidate: Poole & Rosenthal
  on elite voting dimensionality. *Unverified* — and note the elite/mass distinction so the
  contrast is fair rather than a straw man.
- **Worked example: the carving demo.** Resample subsets of the ~120 GSS variables and show
  centrality and modularity rankings swing. Runs on existing code and data; no loader work.
- Why it goes first: it doubles as the empirical proof of §4.4's concession, so the reader
  meets the evidence two sections before the concession it forces.
- **flag:** say plainly that the demo undercuts absolute readings of our own U1 numbers.
  That is the point, not a cost.

### 3.2 — U2, upstream causation

- What the pattern licenses, as two rules.
- **Rule 1.** No correlation implies no *homogeneous population-wide* relationship — an
  argument for subgroup analysis, not a dead end. A relationship present with opposite signs
  in two subgroups vanishes in the pooled sample.
- **Rule 2 — P1.1 lands here.** Different networks imply different causal structures **or**
  different distributions of exogenous inputs **or** item non-equivalence. Three branches,
  and for liberals and conservatives the second is the *expected* case: the groups differ in
  age, education, region, religiosity and media diet. So the claim reduces to "something
  differs," which the networks differing already told us.
- Say what would rescue it: the compositional decomposition (P2.5), or conditioning on the
  inputs. Both blocked on infra gap 1 — the loader retains no demographics or weights. State
  the blocker rather than hiding it.
- Additional hazard worth naming: splitting on POLVIEWS is conditioning on POLVIEWS, which
  can induce association by different amounts in each stratum and manufacture a difference
  out of one common structure.
- **Worked example: polarization potential.** Edges whose sign flips between groups,
  weighted by how similar the groups' marginals are — presented as the upstream claim it is
  (decision 3).
- State all five rivals in the text: genuinely opposite wiring (the interesting case),
  different confounding, collider artifacts from conditioning, estimation noise worst at the
  sign boundary, and curvilinear relations across the ideological range giving each half of
  a split sample a different limb of one curve.
- Name the shock test as what would settle it, and say it runs in Paper 2 (decision 4).
- **flag:** "sign-flipped edges are a screening device, not a measurement" goes in the body
  text, not a footnote.

### 3.3 — U3, downstream causation

- The thesis, stated as an inversion of the field's explanatory direction: most work treats
  population structure as an *outcome* of individual processes. We claim it is also a
  *cause*.
- The prediction that makes it contentful rather than definitional: hold the marginals
  fixed, change the correlation structure, and outcomes change. Nobody's belief has to move
  for the politics to move.
- Four mechanisms. **Search** — which solutions get considered when a topic comes up, a
  locality claim about what sits adjacent. **Coalition** — which alliances are assemblable
  and which wedges can be driven, a claim about cuts and sign structure. **Reception** — how
  a public message lands, via the composition of who is listening. **In-brainification** —
  population regularities becoming felt normative links, feeding U3 back into U2 over time.
- *Association* is retained here as the mechanism name; the rest of the trichotomy stays cut.
- **Worked example: the same message means different things to each group.** Restated as a
  composition effect — a message reaches the people who care about the topic, and those
  people carry different other views in each population. No individual-level "calls up"
  step. (Argument 5, moved here per decision 7.)
- **flag:** drop the dog-whistle framing. Dog-whistling is about decoded meaning, which is
  an individual-level story and pulls against the composition reading.
- **flag:** this is the most exciting leg and the least evidenced. No intervention, no
  experiment. State it as a research programme — the honesty here is what makes §3.1 and
  §4 credible.
- → **cite:** Zaller (1992) on which considerations come to mind, for the search mechanism.
  *Candidate, unverified.*

### 3.4 — The payoff

- Network statistics stop being decorative and acquire substantive readings.
- Modularity → cleavability. Hub degree → signalling efficiency. Density → constraint on
  livable belief-sets. Sign structure → which coalitions are buildable at all.
- **flag:** argument 4's caveat belongs here. Cleavability quietly imports a model of
  political behaviour — that support tracks how many of your positions a platform matches.
  Plausible and standard, but a model. State it and cite it rather than letting it read as
  following from the network alone.
- All four readings are non-separable quantities, which closes the loop with §2.4 without
  restating it.

### 3.5 — The meta-point

- One sentence: for U1 and U3 correlation is the *right* level of description; for U2 it is
  merely the *available* one.
- Why it matters: it explains why U2 carries all the caveats and U1/U3 do not. The asymmetry
  is principled, not rhetorical convenience.
- It is the premise §4.7's estimator corollary discharges, so state it crisply enough to be
  picked up two sections later.

### 3.6 — Close on Victor's synthesis

- Different networks mean subpopulations react differently to the same stimulus.
- Two routes, kept apart: causal connections between beliefs (U2), or correlations with
  causal impact in the world (U3).
- One observable, two mechanisms — and part of the contribution is that these get conflated.
- **flag:** the formulation is Victor's and it is better than what we had. Keep his wording.

---

# §4 — Why the nature of beliefs doesn't matter (much)

**Job.** Charge the bill, having paid the reader first.
**Claim.** The framework assumes only persistence and objectivity, not discreteness — and
therefore no belief network is platonic, so claims must be comparative; but comparison plays
two roles and only the calibration role is repaired by comparativism.

### 4.1 — Voice the objection

- Belief networks look like they presume beliefs are discrete objects stored in the mind,
  mutually referencing, like a set of files.
- Voice it in its strongest form. A reader who holds this worry should feel understood
  before being answered.

### 4.2 — Deny it, generatively

- A belief could be a projection of a brain-state vector in some direction, or a region of a
  high-dimensional embedding space. Nothing above changes. (Victor: "belief box? High
  dimensional embedding space? No problem.")
- → **cite:** Kozlowski, Taddy & Evans (2019) on the geometry of culture, for the embedding
  case. *Candidate, unverified.*
- The generative argument for minimality: every additional commitment forecloses questions,
  so minimality is a research strategy rather than modesty. This is the piece's stance in
  miniature.
- The representationalism / dispositionalism / eliminativism survey compresses to a sentence
  here (deliberately cut).

### 4.3 — State what *is* required: persistence and objectivity

- **Persistence** — beliefs hold still long enough to be measured and correlated.
- **Objectivity** — the propositional content is the same across respondents (A2), and the
  belief-attitude is a real property of a person (A1).
- Persistence is Victor's addition and it is load-bearing: 4.7's trajectory corollary has no
  object to track without it. Credit it.
- **flag:** our A0/A1/A2 covered objectivity twice and persistence not at all. Worth one
  clause on why — the assumptions were written when the framework had no temporal claim.

### 4.4 — Pay the bill

- Without discreteness the network is not an objective structure. It exists relative to the
  questions asked, because a researcher chose how to discretize the space.
- No belief network is platonic. "The modularity of American belief" is not a quantity.
- §3.1 already showed this happening rather than asserting it — call back explicitly.
- Concede the full cost: this invalidates absolute readings of our own published numbers
  too. Saying so is what makes the concession credible rather than rhetorical.

### 4.5 — Therefore comparative by construction

- State it as a design principle, not a caveat.
- What it forbids: absolute figures, cross-instrument comparisons, claims about "the" belief
  network of a population.
- What it licenses: comparisons on a shared instrument, where instrument-dependence cancels.
- **flag:** "cancels" is doing real work and must be stated precisely. It cancels for
  quantities monotone in the shared instrument — an assumption, not a theorem. Do not
  overclaim here; the whole section's credibility rests on this sentence being careful.

### 4.6 — But comparison does two jobs

The hardest and most valuable paragraph in the paper.

- **Calibration.** The property belongs to one network; the second is a ruler. Relativity
  cancels and comparativism repairs the claim outright. Example: liberals divide into camps
  more easily than conservatives.
- **Object.** The difference itself is the claim. Not repaired by comparison — *made* of it,
  so each such claim needs its own defence. Example: the same message means different things
  to each group.
- **The deletion test.** Remove the comparison group. Unreadable → calibration. Nonexistent
  → object.
- The consequence that matters: §5's arguments are object-type, so §4.5 does not discharge
  them and §5 pays separately.
- Where our eight arguments fall: 1–4 calibration, 5–8 object — and 7–8, the two we
  originally led with, are the weakest of the set.
- (OQ 007.)

### 4.7 — Two corollaries

- **P1.2 lands here. The estimator follows the utility.** Partial correlations for U2, where
  the claim is about direct structure; raw correlations for U3, where the claim is about what
  co-occurs in the world regardless of why. Follows directly from §3.5.
- Supporting evidence worth citing to ourselves: raw Pearson tracks the LASSO trajectory at
  r ≈ 0.91 (`sound_10`), so this is a principled split rather than a hedge.
- **P1.3 lands here. Trajectories.** Repeated cross-sections give three sources of change:
  causal structure, input distribution, and cohort replacement. All three produce the same
  observable.
- Cohort replacement is the one routinely ignored and it is not a nuisance term: a population
  can change its network with no individual changing any belief.
- **flag:** this corollary needs the persistence assumption from 4.3, and is blocked
  empirically on infra gap 1.

---

# §5 — What follows

**Job.** Stakes, then open the question space rather than closing it. Short — a coda, now
that argument 5 has moved to §3.3.
**Claim.** Two populations can agree almost entirely and still be arranged to respond to the
same event in opposite directions.

### 5.1 — Fragmenting as collectives, not merely disagreeing

- If a collective is constituted by a shared set of associations, two subpopulations with
  different networks are becoming two collectives rather than one.
- Network distance measures fragmentation of the shared background against which
  disagreement is possible — not disagreement itself. That distinction is the whole content
  of the argument.
- Label it a **proposal**, not a finding. It follows from a definition, so it cannot be a
  discovery about liberals and conservatives.
- It needs a threshold: any two subgroups differ somewhat, and as stated the argument
  triggers on any difference at all.
- **flag:** our operationalisation is Frobenius distance, which §2.4 classifies as
  separable. Either adopt a structural distance or say plainly that the measure does not
  match the claim. This is the one place the separability criterion bites our own text, and
  the paper is more credible for catching it than for hiding it.
- The normative material belongs here and nowhere else.
- → **cite:** Mason (2018), *Uncivil Agreement*, as the nearest empirical neighbour.
  *Unverified, not in `literature/`.*
- → **cite:** Baldassarri & Gelman (2008) on partisan sorting without constraint — the
  closest thing to a rival account. *Unverified, not in `literature/`.*

### 5.2 — Agreement is not stability

- The closing claim: two populations can hold nearly identical belief distributions and
  still be arranged to respond to the same event in opposite directions.
- Title candidate.
- One paragraph. It is a claim, not a section.
- Restate honestly that it is currently a programme — polarization potential is a screening
  device and the shock test runs in Paper 2.

### 5.3 — Close on the question space

- Not a summary. The last page opens questions rather than closing them, which is what
  "generative" means operationally.
- Cleavability over time. Signalling-node turnover. Structural versus compositional
  divergence (OQ 003). DIF as a precondition for any comparison at all (OQ 004). Structure
  at *t* predicting outcomes at *t+1*.
- One more that this week produced: **which published belief-network claims survive the
  separability test.** That is an audit somebody could actually run, and it hands the reader
  a job.
- **flag:** end on the audit or on the *t+1* prediction. Not on a recapitulation.

---

## Deliberately cut

- Evolution / group selection over belief configurations — uncited, walks into the
  cultural-group-selection minefield. Spin out or gesture with citations.
- The representationalism / dispositionalism / eliminativism survey — serves a reader we
  are not writing for. Compresses to a sentence in §4.2.
- The correlation / association / regulation trichotomy as its own section. *Association*
  survives inside §3.3 as the mechanism name.
- "Societal brains" beyond one dismissive paragraph in §2.5.
- "What we did" as a standalone section (decision 2).

---

## What pass 4 needs

- **A real opening number** for §1.1 — a variable pair and correlation pulled from
  `sound_01`.
- **Verified citations.** Every slot above is from memory. Four of the must-engage papers
  are still missing from `literature/`.
- **A word budget.** §3 is the longest by design; §5 is a coda; §1 and §2 should be fast.
  Splitting the target length across five sections is a pass-4 decision and depends on
  venue, which is still open (OQ 006).
- **Two unresolved empirical dependencies to state honestly in the text:** infra gap 1
  blocks the U2 rescue in §3.2 and the trajectory corollary in §4.7.
- **A decision on §5.1's distance measure** — structural replacement, or an admission in
  place.
