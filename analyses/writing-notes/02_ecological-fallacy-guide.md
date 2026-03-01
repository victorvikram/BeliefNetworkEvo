# Ecological Fallacy Language Guide

## The Problem

The paper analyzes population-level partial correlations between survey responses.
These correlations reflect the joint product of individual cognition, social
influence, elite cue-giving, media environment, and compositional change. The
skeleton repeatedly implies individual-level cognition from aggregate data —
a textbook ecological fallacy (Robinson 1950).

---

## Terminology Rules

### "Belief network"
**Use for:** The statistical object — the population-level partial correlation graph.
- YES: "the liberal belief network shows higher density"
- YES: "the belief network for conservatives has 356 edges"

### "Belief system"
**Use only when:** Discussing theoretical constructs about individual-level cognition
(e.g., referencing Converse 1964's theory of constraint).
- YES: "Converse's theory of belief systems posits..."
- NO: "the conservative belief system is more modular" (should be "belief network")

### Replace: "organize beliefs"
- BEFORE: "liberals organize their beliefs around urban policy"
- AFTER: "in the liberal network, urban policy variables co-vary more strongly with other beliefs" or "urban policy variables are more central in the liberal associational structure"

### Replace: "reasoning from different structural premises"
- BEFORE: "they are reasoning from different structural premises"
- AFTER: "they are embedded in different associational structures" or "the population-level dependency structures differ"

### Replace: "cognitive consistency pressures"
- BEFORE: "cognitive consistency pressures operate at the population level"
- AFTER: "consistency at the population level — reflecting some combination of individual cognition, social influence, and elite cue-giving — produces near-perfect structural balance"

### Replace: "which issues serve as organizing hubs for the rest of the belief system"
- BEFORE: "which issues serve as organizing hubs for the rest of the belief system"
- AFTER: "which issues are most central in the population-level dependency structure"

### Replace: "constructing different axes"
- BEFORE: "constructing different axes altogether"
- AFTER: "the population-level associational structures are organized around different hub variables"

---

## Required Caveat Paragraph

**Location:** Methods section, after describing the network estimation procedure.

> **Draft text:** "An important caveat governs interpretation throughout this paper.
> The partial correlation networks we estimate are population-level statistical
> objects. They capture the associational structure among attitudes across
> respondents, not the cognitive architecture of any individual. A positive
> partial correlation between attitudes A and B means that, controlling for
> all other measured attitudes, respondents who score higher on A tend to score
> higher on B — but this could reflect individual-level cognitive consistency,
> social influence from shared environments, elite cue-giving that packages
> issues together, or compositional differences within ideological groups.
> Following Brandt and Sleegers (2021), we avoid inferring individual-level
> cognitive processes from population-level structure. We use 'belief network'
> to refer to the statistical object and reserve 'belief system' for theoretical
> discussions of individual cognition (cf. Converse 1964). The evolution of
> population-level associational structure is valuable to track regardless of
> its micro-level sources, because it reveals the macro-level architecture
> within which political communication, coalition formation, and issue
> packaging operate."

---

## Find-and-Replace Checklist for Skeleton

### Thesis paragraph (line 9)

1. "they organize their beliefs according to fundamentally different structural logics"
   → "the population-level associational structures among their beliefs follow different organizational patterns"

2. "which issues serve as organizing hubs for the rest of the belief system"
   → "which issues are most central in the population-level dependency structure"

3. "Conservative belief systems are more modular"
   → "Conservative belief networks are more modular"

4. "Americans are not just moving to opposite poles on a shared axis, but constructing different axes altogether"
   → "Americans are not just moving to opposite poles on shared issues — the population-level dependency structures for the two groups are increasingly organized around different hub variables"

### Section 4, Implication paragraph (line 104)

5. "they are not just disagreeing — they are reasoning from different structural premises"
   → "they are not just disagreeing on positions — the population-level associational structures for the two groups are organized around different issues"

6. "This finding is invisible without the network representation"
   → "This finding is invisible without comparing group-specific dependency structures"

### Section 5, heading (line 110)

7. "Conservative belief systems are more modular" (if used)
   → "Conservative belief networks are more modular"

### Discussion — Cognitive science section (lines 211-213)

8. "cognitive consistency pressures operate at the population level, not just within individuals"
   → "population-level consistency — reflecting some combination of individual cognition, social influence, and elite cue-giving — produces near-perfect structural balance"

9. "the *domains* of belief are cognitive primitives — they reflect how people naturally organize attitudes"
   → "the stability of community structure suggests these domain groupings are robust features of the population-level associational landscape, potentially reflecting shared cognitive categories, stable social cleavages, or enduring elite issue-framing"

10. "The independence finding (conservative lower constraint) raises questions about the relationship between ideological constraint and cognitive style"
    → "The independence finding (conservative lower constraint at the population level) raises questions about the sources of differential associational structure — whether driven by individual cognitive style, heterogeneous elite cues, or weaker social influence networks"

### Discussion — Polarization theory (lines 199-203)

11. "they are reasoning from different structural premises"
    → (same fix as #5 above)

### Discussion — Political science (line 205)

12. "political identity is increasingly the organizing principle of the full belief system"
    → "political identity variables are increasingly central in the population-level dependency structure" (NOTE: this moves to supplementary under POLVIEWS exclusion)

### Methodological Argument (line 32)

13. "not dress up matrix comparisons as network findings"
    → This phrasing is fine; it's a methodological note, not an ecological fallacy.

---

## General Principles

- **When in doubt, add "population-level" or "associational."** These qualifiers
  make claims empirically precise without weakening them.
- **Never use "think," "reason," "cognize," or "process" for what the network shows.**
  The network shows co-variation patterns, not thought processes.
- **"Structural" is OK.** It describes the statistical object, not cognition.
- **"Hub" and "central" are OK** as graph-theoretic terms — just make clear they
  refer to the network, not to cognitive importance.
- **"Different dependency structures" is the safest general framing.**
  It is empirically precise: the conditional dependencies differ.
