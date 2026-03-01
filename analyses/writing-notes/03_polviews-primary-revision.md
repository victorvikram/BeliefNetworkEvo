# POLVIEWS Primary Revision

## The Problem

POLVIEWS is used to *define* the liberal/conservative groups AND appears as a
node in the network. Within the liberal group, POLVIEWS has restricted range
(only values < 0), mechanically affecting its correlations with all other
variables. Four of six reviewers flagged this circularity. The with-POLVIEWS
analysis must become supplementary; the POLVIEWS-excluded version becomes primary.

---

## Corrected Headline Numbers (POLVIEWS/PARTYID excluded, N=62 vars)

### Euclidean Distance Trend

| Specification | Slope (/yr) | p |
|---------------|-------------|-------|
| HAC overlapping | 0.00417 | 8.08e-04 |
| **Non-overlapping (primary)** | **0.00440** | **0.0207** |

### Centrality Rho Trend

| Specification | Slope (/yr) | p |
|---------------|-------------|-------|
| HAC overlapping | -0.00145 | 5.40e-04 |
| **Non-overlapping (primary)** | **-0.00102** | **0.054** |

**NOTE:** The centrality rho trend is marginal (p=0.054) under the cleanest
specification. This must be reported honestly. Frame as "directionally
consistent" or "suggestive," not as a strong standalone finding.

---

## New Hub Variables

From sound_07 Check 4 (excluding POLVIEWS/PARTYID), top centrality movers:

| Variable | Slope | r | Substantive Meaning |
|----------|-------|-------|---------------------|
| PORNLAW | 0.5644 | 0.649 | Attitudes toward pornography laws |
| POLHITOK | 0.4983 | 0.703 | Whether police hitting citizens is OK |
| POLMURDR | 0.3171 | 0.577 | Whether police striking murder suspects is OK |
| CONPRESS | 0.2998 | 0.536 | Confidence in the press |
| SPKHOMO | 0.2364 | 0.418 | Whether homosexuals should be allowed to speak |

FDR-surviving (q<0.05) from sound_08:
- POLHITOK (q=0.0051)
- POLMURDR (q=0.0051)
- SUICIDE1 (q=0.0051)
- POLABUSE (q=0.0030)

---

## Reinterpretation

**Old framing (circular):** "POLVIEWS is the fastest-rising hub, meaning
political identity is increasingly the organizing principle of belief systems."

**New framing (clean):** "Morality and policing variables — attitudes toward
pornography regulation, police use of force, and euthanasia — are the
fastest-diverging organizing hubs. This pattern suggests that the moral
and authority dimensions of ideology, rather than partisan identity itself,
drive the structural divergence between liberal and conservative associational
structures."

**Why this is MORE interesting:**
1. Not circular — the splitting variable is excluded
2. Substantively richer — points to specific issue domains (moral authority,
   policing) as the locus of structural divergence
3. Connects to broader literature on moral foundations (Haidt 2012) and the
   "culture wars" (Hunter 1991)
4. The policing variables (POLHITOK, POLMURDR, POLABUSE) form a coherent
   cluster around attitudes toward state authority and use of force

---

## What Moves to Supplementary

1. **With-POLVIEWS centrality results** — the original "rho 0.79→0.58" trajectory
2. **POLVIEWS centrality trajectory** — the "+0.004/yr" finding
3. **GRASS centrality trajectory** — the "-0.003/yr" finding (dependent on
   POLVIEWS being in the variable set)
4. Any interpretation framed around "politicization of the belief system"

**Supplementary framing:** "When POLVIEWS and PARTYID are included as network
nodes (supplementary analysis), POLVIEWS shows the fastest centrality increase.
However, because POLVIEWS is also used to define the ideological groups, these
results are potentially contaminated by mechanical restriction-of-range effects.
All primary analyses therefore exclude POLVIEWS and PARTYID from the variable set."

---

## Skeleton Changes Required

### Section 4, Evidence bullets (lines 96-102)

Replace the centrality evidence block with:

> - **Key finding:** Euclidean distance between group-specific partial correlation
>   matrices increases at +0.0044 per year (non-overlapping windows, p=0.021;
>   HAC-corrected overlapping, p<0.001). Centrality rank correlation declines
>   (HAC p<0.001; non-overlapping p=0.054 — directionally consistent).
> - The fastest-diverging hub variables (FDR q<0.01) are POLHITOK (police
>   hitting citizens), POLMURDR (police striking suspects), SUICIDE1
>   (right to die), and POLABUSE (police abuse) — morality and authority
>   variables, not partisan identity markers.
> - Sign disagreements essentially zero across all windows: the divergence is
>   quantitative reorganization, not qualitative reversal.
> - Community structure is highly stable (NMI ~0.946): the containers for
>   belief stay constant; the importance of each container changes.

### Evidence Table row 4b (line 140)

Replace:

> | 4b | **Centrality divergence** | sound_04, sound_05, sound_07, sound_08 |
> **Centrality rho: HAC p<0.001 / non-overlap p=0.054; Hub movers: POLHITOK
> (q=0.005), POLMURDR (q=0.005), SUICIDE1 (q=0.005)** | **Graph (essential)** | Fig 4 |

### Figure 4 description (lines 177-183)

Replace hub-migration description:

> (B) Hub divergence: variables whose centrality rank-difference between groups
> is growing fastest (FDR q<0.05): POLHITOK, POLMURDR, SUICIDE1 (becoming
> relatively more central for conservatives), POLABUSE (becoming relatively
> more central for liberals). These morality and policing variables represent
> the substantive core of structural divergence.

### Discussion — Political science (lines 204-206)

Remove or move to supplementary:
> "The finding that POLVIEWS centrality is rising fastest connects to the
> 'sorting' literature (Mason 2018)"

Replace with:
> "The finding that policing and moral-authority variables are the fastest-
> diverging hubs connects to research on moral foundations (Haidt 2012) and
> suggests that the structural divergence is rooted in attitudes toward
> authority and social regulation rather than in partisan identity per se."

---

## Note on sound_05 Rerun

Ideally, sound_05 (structural evolution — hub migration trajectories, community
stability) should be rerun with `vars_no_pol` (62 vars excluding POLVIEWS/PARTYID)
to produce clean hub-migration figures without the contaminated POLVIEWS
trajectory. This would give us:
- Updated NMI stability trajectory
- Clean centrality gainer/loser plots
- Hub migration visualization without POLVIEWS

**Status:** Optional code task. The current sound_07 Check 4 provides the key
numbers (top movers), but the figures from sound_05 still show POLVIEWS as the
top gainer. For submission, either rerun sound_05 or create new figures from
sound_07 Check 4 output.
