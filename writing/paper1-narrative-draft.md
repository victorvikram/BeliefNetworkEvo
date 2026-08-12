# Paper 1 — narrative structure draft (Tim + Claude)

Written 2026-08-06. To be compared against Victor's independent draft.

Audience: practitioners, sociologists, and theory-minded computational social
scientists. Not philosophers of mind. The piece should partly create its reader.

Sources: `notes/filed/conceptual-development.md`, `notes/conceptual-piece-review.md`,
`notes/research-plan.md`.

---

## The story in one paragraph

Surveys can show which beliefs tend to occur together in a population, but not why.
When two beliefs are correlated, the data cannot distinguish people who connect the
two ideas themselves from people who simply grew up in the same place, attend the same
church, or read the same news. This is usually treated as a weakness, and it is one if
the goal is to describe how individuals think. But the same data accurately describes
something else: the pattern of which beliefs occur together across a population. That
pattern matters on its own, because it shapes what happens in the population. Where
people who worry about falling birth rates also tend to hold traditional views about
family, discussions of birth rates will tend to raise traditional solutions and pass
over others — not because anyone has connected the two ideas, but because of who takes
part in the conversation. Understood this way, standard network measures describe real
features of a population: how easily it divides into opposed camps, which beliefs
reveal the most about a person's other views, and how restricted the available
combinations of belief have become. These measures depend on which questions the survey
asked, so they say little on their own, and a great deal when two populations are
compared using the same questions. That comparison leads to the paper's main point: two
populations can hold almost the same beliefs and still respond to the same event in
opposite directions. Agreement is not stability.

---

## The narrative, in four beats

1. **The problem.** Correlations between survey items cannot tell us why two beliefs
   occur together, so they cannot describe how individuals think. // i'd decompose the problem as lacking causal structure and lacking individual resolution

2. **The solution.** They do describe something else accurately — the pattern of which
   beliefs occur together across a whole population — and that pattern has real effects. // sounds like we are focusing on downstream causation. is this what we want?

3. **What we did.** We set out what you have to assume to study that pattern, what it
   can and cannot tell you, and why claims about it have to be comparative. // 

4. **Why it matters.** Two populations sharing a country can drift into hearing
   different things in the same words, and into responding to the same event in
   opposite directions. // this is just one example from downstream causation and upstream causation

**This is the working structure.** The six sections below and the appendix beats are
detail to draw on, not a competing outline.

---

## How the four beats break into sections

Beats 1 and 2 are one section each; beat 3 is three; beat 4 is one.

1. **The problem.** Correlations between survey items cannot tell us why two beliefs
   occur together, so they cannot describe how individuals think.

2. **The reframe.** They do describe something else accurately: the pattern of which
   beliefs occur together across a whole population.

3. **What you have to assume.** Studying that pattern requires very little in the way of
   assumptions about beliefs, but it does mean the network depends on which questions
   were asked.

4. **What it is good for.** The pattern predicts a person's other views, says something
   partial about underlying causes, and — most importantly — produces effects of its
   own.

5. **How claims must be made.** Because the network depends on the survey, absolute
   figures say little and comparisons using the same questions say a lot.

6. **Polarization potential.** Two populations can hold nearly the same beliefs and
   still be set to move in opposite directions, and we show how to measure that.  // I feel like we might want to give one example for each of the "what is it good for"

---

# How two populations with different networks actually differ

Worked through 2026-08-06. This is the substance of beat 4 and the part that needs the
most care, because the arguments are not equally good and the most exciting one is the
weakest.

## Comparison does two different jobs

We had been running two different things together, and they need different defences.

Sometimes what we want to say is about one population, and the second one is there only
to make the number mean something. Every network statistic depends on which questions
the survey asked. A liberal modularity of 0.42 would have come out differently if the
GSS had included fifteen more questions about religion, and there is no scale to read
the figure against. Measuring a second population with the same questions supplies one.
"Liberals divide into camps more easily than conservatives" holds up even though neither
figure means anything on its own. Call this **calibration**: the second network is a
ruler, and the claim is still about the first population.

Other times the difference itself is what we want to talk about. "The same speech is
heard differently by the two groups" cannot be said about liberals alone. Remove
conservatives and the claim does not get harder to read, it stops existing. Call this
**the difference as the object**: the second network is half of what is being described.

A quick way to tell them apart is to imagine deleting the comparison group. If the claim
becomes unreadable, it was calibration. If it becomes nonexistent, the difference was the
object.

This matters because the paper's central move only pays for the first kind. Section 5
argues that no network has an absolute interpretation, so claims have to be comparative,
and that repairs calibration claims completely: the dependence on the survey cancels out
and nothing further is owed. Claims about a difference get no such help. They are not
repaired by comparison, they are made of it, so each one has to be defended on its own.
Argument 8 needs a sign flip to mean opposite wiring rather than noise or confounding.
Argument 7 needs the two groups to differ in their causes rather than merely in who
belongs to them.

That is the trouble with the two arguments we first chose to lead with. Arguments 7 and 8
are both claims about a difference, and both are the weakest of the eight. The four that
need the least defending, 1 to 4, are all calibration. Two others, 5 and 6, are claims
about a difference and still hold up, which is why beat 4 is built on those.

## Calibration-type arguments (properties of one network)

1. **You can infer different things about a person.** Airtight, because it restates what
   a correlation is — and nearly empty for the same reason. It describes the
   measurement, not a finding.
2. **Different combinations of belief actually exist.** Holds up well, and more useful
   than it looks because it is a claim about which people exist. Caveat to state:
   correlation is a lossy summary, so different correlations imply different joint
   distributions, but equal correlations do not imply equal ones.
3. **Different things get discussed and different solutions get considered.** The
   strongest of the interesting arguments and under-sold in the draft. Needs no
   assumption about anyone's reasoning and no causal claim — it is a composition
   effect. Assumes only that venues assemble by topic and people bring their other
   views with them.
4. **The population divides into camps more or less easily.** Good, but it quietly
   imports a model of political behaviour — that support tracks how many of your
   positions a platform matches. Plausible and standard, but it is a model, and the
   draft presents it as following from the network alone. State it and cite it.

## Object-type arguments (genuinely about the difference)

5. **The same message means different things to each group.** *Lead with this.* If a
   word or event sits next to different beliefs in each network, the same public
   statement calls up different ideas in each population. Holds up better than 7
   because it needs only the associative reading, never a causal one — nothing has to
   be wired anywhere. Requires only that the stimulus reaches both groups, which for
   national politics is true by construction. Has a recognisable phenomenon behind it
   (dog-whistling), and it demonstrates the paper's own thesis, since it works entirely
   at the level of correlation with nothing underneath. Present in the draft as the
   "political discourse" illustration; promote it.
   **To consider:** "calls up" is an individual-level claim and population co-occurrence
   does not license it, so it needs restating as a composition effect like argument 3 —
   a message reaches the people who care about the topic, and those people carry
   different other views in each population. The dog-whistle example pulls the wrong way,
   since dog-whistling is about decoded meaning rather than co-occurrence, and "true by
   construction" overstates shared exposure given how differently the two groups are
   reached.
6. **How far apart the two are as collectives.** If a collective is constituted by a
   shared set of associations (whiteboard 2), then two subpopulations with different
   networks are becoming two collectives rather than one. Network distance measures
   fragmentation of the shared background against which disagreement is possible — not
   disagreement itself. Conceptual rather than empirical, so it needs no causal
   support. This is where the normative material belongs.
   **To consider:** because it follows from the definition of a collective it makes no
   empirical claim, so it cannot be a finding about liberals and conservatives and should
   be labelled a proposal. It also needs a threshold, since any two subgroups have
   somewhat different networks, and network distance may not measure shared background at
   all — two networks can be far apart while agreeing on all their strong edges.
7. **Different underlying causes.** Corrected: different causal structures **or**
   different distributions of the inputs feeding them. Weak, and the reason should be
   stated plainly: for liberals and conservatives the second branch is the *expected*
   case, since the groups differ in age, education, region, religiosity and media diet.
   The claim reduces to "something differs," which we knew from the networks differing.
   Rescued only by doing the decomposition (P2.5) or conditioning on the inputs. A
   scaffold for a question, not a result.
   **To consider:** the disjunction has a third branch, since items that function
   differently across the two groups produce different networks from identical structure
   and identical inputs — so measurement belongs here and not only in the precondition
   below. Separately, splitting on POLVIEWS is conditioning on POLVIEWS, which can induce
   association between belief pairs by different amounts in each stratum and so
   manufacture a difference out of one common structure.
8. **Opposite responses to the same event (polarization potential).** Most striking,
   currently an assertion. A sign-flipped edge is consistent with four things: genuinely
   opposite causal wiring (the interesting case), different confounding structure,
   collider artifacts from conditioning in the partial version, and estimation noise,
   which is worst exactly at the sign boundary. **Sign-flipped edges are a screening
   device, not a measurement.** That is a gap, not a footnote.
   **To consider:** a fifth rival is missing and may be the leading one — if two beliefs
   relate curvilinearly across the ideological range, each half of a split sample
   recovers a different limb of that curve, so the signs can differ with one common
   structure and no polarization potential at all. The step from a cross-sectional
   partial correlation to a prediction about movement also needs the association to be
   causal, directed, and stable under intervention, which is more than the corrected
   claim 3 supplies on its own.

## The precondition

All eight assume the survey items mean the same thing to both groups. If they do not,
the networks differ for measurement reasons and every argument above is contaminated.
This is OQ 004, it is untested, and it belongs near the front as a stated precondition
rather than at the back as a limitation.

## The test that would fix argument 8

Identify sign-flipped edges in a window *before* a shock. Then check whether the two
groups actually moved in opposite directions on those items *after* it, against
non-flipped control edges. This converts polarization potential from a description into
a testable forecast, and 2008 is sitting right there — the unexplained 16× acceleration
would stop being unexplained if the pre-2008 flipped edges are the ones that moved.

**To consider:** as specified the test measures whether the groups diverged on the
*items*, which is marginal movement rather than edge change, so it does not test the
mechanism being claimed. And flipped edges are selected for sitting near the sign
boundary, which is where sampling error is largest, so they regress toward the mean by
construction — the controls have to be matched on edge magnitude, not merely on being
unflipped.

Runs on the existing rolling-window machinery; no loader work. It is also more than a
mini-demo, so it may belong in Paper 2 or Paper 3 rather than Paper 1. Decide rather
than drift.

## How beat 4 should be built

Three things, in order of how well they stand:

1. The same message means different things to each group.
2. The groups are drifting apart as collectives, not merely disagreeing.
3. They may respond to the same event in opposite directions — stated as the open
   question it is, with the shock test as the way to settle it.

---

# Appendix — expanded beats

Kept from the first pass. Not the working outline; detail to draw on once the six
sections above are agreed.

### 1. The residual

- Open inside the practitioner's situation, not a literature review.
- Name the identification problem concretely — co-occurrence has many possible sources
  and cross-sections cannot separate them.
- Concede it fully. For individual-level inference this is fatal, not awkward. Engage
  Dalege, Brandt & Sleegers, Boutyline & Vaisey respectfully — careful work, but a
  mis-specified object.
- Refuse the standard response, which is to caveat and proceed anyway.

### 2. The residual is the object

- The inversion: what is noise for inferring cognition is signal about a collective.
- Define the object. Nodes are belief categories; edges are population co-occurrence;
  the collective is the unit.
- State what it aggregates, deliberately: in-brain association, social transmission,
  shared environment, demography, media diet.
- Dispatch the three confusions fast — not individual belief systems (ecological
  fallacy), not objective structures of a collective (reification), not societal brains
  (out of scope). One short paragraph each.

### 3. What you must assume, and what you needn't

- Argued generatively: every additional commitment forecloses questions, so minimality
  is a research strategy, not modesty.
- A0 (multiple simultaneous beliefs), A1 (belief-attitude objectivity), A2
  (propositional-content objectivity). Explicitly **no discreteness assumption**.
- Pay the bill: A2 is empirical, not conceptual, and differential item functioning is
  how you would test it. A pointer to survey methodology, not a battle.
- Land the consequence section 5 needs: no belief network is platonic.

### 4. What it is good for

- **U1 — passive inference.** Correlations *are* the right level of description. Hubs
  as signalling nodes, modules as informational neighbourhoods, density as dimensional
  constraint.
- **U2 — upstream causation.** A compressed, many-to-one image of the causal structure
  that produced it. Corrected claim: different networks imply different causal
  structures **or** different distributions of exogenous inputs — framed as a research
  question, not a retreat. Same for claim 2: absence of correlation implies absence of a
  *homogeneous* population-wide relationship, which argues for subgroup analysis.
- **U3 — downstream causation.** The paper's engine. Social co-occurrence steers which
  solutions get searched; coalition structure determines cleavability; in-brainification
  turns population regularities into felt normative links, feeding U3 back into U2 over
  time. *Association* is retained here as the mechanism name.
- **The payoff.** Network statistics acquire substantive interpretations — modularity as
  cleavability, hub degree as signalling efficiency, density as constraint on livable
  belief-sets, sign structure as which coalitions are buildable at all.
- **The meta-point.** For U1 and U3, correlation is the *right* level of description.
  For U2 it is merely the *available* one.

### 5. The form claims must take

- Relativity is fatal for absolute claims. "The modularity of American belief" is not a
  quantity.
- But it **cancels under comparison on a shared instrument**.
- State it as a design principle: **comparative by construction**. Say what it forbids,
  not only what it licenses.
- Corollary 1 — **the estimator follows the utility.** Partial correlations for U2, raw
  for U3.
- Corollary 2 — **trajectories.** Repeated cross-sections give three sources of change:
  causal structure, input distribution, and cohort replacement.

### 6. Polarization potential

- The concept: near-identical belief distributions, opposite propagation of the same
  shock.
- Operationalise: edges whose sign flips between groups, weighted by how similar the
  groups' current marginals are. Separates **observed** from **latent** polarization.
- The demonstration (OQ 005 candidate). Caveats stated in the text: the step from
  sign-flipped partial correlation to opposite shock response runs through the
  corrected claim 3; and sign flips in sparse graphs are where estimation noise lives,
  so bootstrap first.
- **Agreement is not stability.**
- Close on the question space — cleavability over time, signalling-node turnover,
  structural vs compositional divergence, DIF as a precondition for comparison,
  structure at *t* predicting outcomes at *t+1*.

---

## Deliberately cut

- **Evolution / group selection over belief configurations.** Uncited, walks into the
  cultural-group-selection minefield, not needed. Spin out or gesture with citations.
- **The representationalism / dispositionalism / eliminativism survey.** Serves a
  philosophy-of-mind reader we are not writing for. Compresses to a sentence.
- **The correlation / association / regulation trichotomy as its own section.**
  *Association* survives inside U3 as the mechanism name; the full taxonomy does not
  earn a section at this audience.
- **The "societal brains" discussion** beyond a single dismissive paragraph.
