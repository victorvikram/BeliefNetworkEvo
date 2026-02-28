# Aug 2024
## Hypothesis/Question motivated analysis
1. What are the persistence structure (triangles) in the network and how do they move around over the course of history (e.g. polviews, partyid, preslast_demrep)?
2. To what extent is there framework fracturing or perspectival polarization in different subpopulations? How much does this explain the polarization that we see (even the basic types like single-issue polarization)
3. What are some general principles by which the network evolves?
  - how does centrality tend to change?
  - can frustration predict network change?
  - does the network tend toward greater connectedness?
4. Can we group belief vectors into clusters? Could the clusters correspond to different energy valleys in the space? What is the shape of these clusters and what are ways that beliefs could conceivably move through the space?

# Jun 2024
1. To what extent do beliefs constrain each other. [updated 16 May]
    *description* 
    - Constraint == one thing encouraging/discouraging (tax/subsidise) another from being adopted

    *so far*
    - Degree analysis gives local picture
    - Modularity gives some intuition about how many sets of beliefs vary independently from one another 
    - If we can remove the spurious v-correlations (common effect links a la smoking, asbestos, and lung cancer) then the remaining partial correlation schemas are evidence of underlying interactions
    - In future, energy landscapes give global picture.
    - preliminary result is that constraints seem to be increasing over time and that the liberal network is more constrained that the conservative one

    *next steps*
    - [Victor, 16 May] look into causality book to figure out how to remove the common-effect associations
    - [Victor, 16 May] ask mirta/rudi/?eddie about how to calculate an energy landscape on things — does it make sense to just use our partial correlations as couplings for the energy landscape?
    - comment and clean code so far

    *long term*
    create a belief network that removes sprurious "common effect " correlations and that can be used to calculate an energy landcape. then, do analysis of the energy landscape to see how many valleys there are, how steep they are, and so on. also would be interesting if there was a measure of modularity where, if there are *n* modules, that would imply that there are 2^n energy valleys (each module would have two valleys, and they can be set independently of one another). and it would be cool to interpret degree and stuff in light of the energy valley interpretation — that is, I am hoping to link the energy valley interpretation with the network interpretation. what does high degree mean? what does modularity mean?

2. are subgroups perspectivally polarized? How much? And how much does this drive changes in opinion?
    *so far*
    - This basically means: if we condition on some belief (like being conservative or liberal), in what way to the contraints between beliefs change? 
    - This would give some insight into "intepretation" of beliefs. I.e., how if conservatives and liberals have different patterns of contraints, this could imply either a different intepretation certain beliefs, or a different underlying method of taxation and subsidisation.  
    - One interesting initial observation was the liberal and conservative-identifying beliefs networks had clearly different connectivity pattterns. Liberal networks had higher degree, seemingly across the board. This could could be consistent with the idea that liberal beliefs are more normative at the level of societies, whereas conservative beliefs are more individualistic. 

    *Next steps*
    - Playing around with other conditionings, or algorithmically searching for interesting conditionings is something we should do.

    *long term*
    would be interesting to look at more creatively defined subgroups. for example, if we looked at groups of people defined by clusters in the belief space, or as inhabiting a particular energy valley, etc, what groups do we get, and how do their networks differ? also, if certain clusters move more than others, is there a quality that characterizes belief systems that change more over time?

3. What are the dynamics by which belief networks evolve. 
    *next steps*
    - write code to make rate of change networks where the nodes are colored according to how much the believers of the belief changes, and the edges are colored according to how much the edges change. hopefully this will give intuition for what nodes and edges are changing the most
    - write a historical account of how the belief networks of the US changed from the 70s to the present. are there a
    
    *long term*
    Would be interesting to look at whether there are some general dynamics by which the network changes. one hypothesis is to look at belief networks and see for example if there is directionality in the formation or dissolution of modules (can we make a module definition that works with the energy landscape definition)

4. permissible and impermissible regions of the belief space.
    *description*
    maybe certain combinations of beliefs are hard to hold simultaneously, while others are easier. if there are clusters in high dimensional space, that would be evidence of this phenomenon. 

    *next steps*
    - could find the clusters in the high dimensional space without any dimensionality reduction, by using a simple algorithm like k-means or the like. 
    - could then collapse the dimensionality using the diffusion map, which should preserve point proximity (diff map collapses dimensionality while hopefully preserving point proximity)
    - [16 May, Tim] principal cunt analysis -- easy first step. main challenge is to vectorise all the data

    *long term*


# May 2024
## Hypothesis/Question-motivated Analysis
1. **Stability and instability of local structures in the belief network.**
  a. Does an analog of social balance hold on the network? That is, are "stable" triads more likely to persist than unstable ones?
  b. What about cycles? Cycles with an odd number of negative edges might be unstable.
2. **Module structure and dynamics**
  a. Is it easier to enter a module than to leave it?
  b. Do modules aggregate and split in different ways (e.g. does merging happen gradually while splitting is sudden)?
  c. Is there a defined and persistent module structure to the beliefs (beyond just different question types or something)? Do different modules correspond to salient value dimensions?
  d. This may relate to 9, since modules 
  d. How flexible is the correlation structure? That is 
4. **Ideology clusters**
  a. Are there unallowable regions of the ideology space?
  b. Do moving clusters follow ideology-respecting paths?
  c. Do the people in the clusters that move have a particular structure to their belief networks?
5. **Centrality questions**
  a. Do more central beliefs get trapped or do they evolve more?
  b. Are more central beliefs associated with voting behavior?
6. **Subpopulations**
  a. would be interesting to see if there is a difference in beliefs structures when conditioning on the strength of party affiliation. Do independents have a different structure of collective beliefs than the others?
  b. what about people who changed party votes? in any sample we often have to voting records for a person so we can see if they voted for different parties
   c. can define subpopulations by regions in the space, and then look at their belief network and compare, see 4c.
7. **Voting**
  a. are beliefs correlated with voting more central?
  b. are beliefs associated with voting slower-changing?
8. **Party and ideology** how do the variables that most strongly influence party identification change over time? is there some pattern where, for instance, certain variables are particularly well-suited to becoming associated with a particular party?
9. **Structure flexibility** ising models have energy valleys of configurations that work together. I wonder if there is an analog of that for this system. Can we look at how many "allowable configurations" there are, and how distributed they are in the space? This might be tied up with modules because modules might be sets of nodes that can vary relatively independently of one another.

## Exploratory Analysis
1. **Rate of change**
  a. Does link creation and deletion happen at a constant rate? If not, when are the spikes?
  b. Can make a rate of change network to see what its structure is
2. **Typical measures**
  a. degree distribution, ...
