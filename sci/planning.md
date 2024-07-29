# Planning 
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

2. do subgroups agree on network structure.
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

# Meeting notes

## 23 Jul 2024 - 
1. how to calculate entropy -- binning? huffman coding? 
2. 
3. 


## 16 May 2024 - post workshop meeting
Goals for this meeting:
1. Summarise our findings wrt our hypotheses in the poster -- what is missing?
2. Write down future hypotheses.
3. Make goals for next week. 

