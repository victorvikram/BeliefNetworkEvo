# Source of data
Here is the GSS data: https://gssdataexplorer.norc.org/gss_data

# Variables that might be of interest (as of 29/04 these have been replaced by the transformed variable names -- this impacts the naming of variables that are split into discrete categories.)
## Note that the election-related variables are put in their own list since for visual clarity.

```
variables = ["PARTYID","POLVIEWS","NATSPAC","NATENVIR","NATHEAL","NATCITY","NATCRIME","NATDRUG","NATEDUC","NATRACE","NATARMS",
"NATAID","NATFARE","NATROAD","NATSOC","NATMASS","NATPARK","NATCHLD","NATSCI","NATENRGY","NATSPACY","NATENVIY","NATHEALY","NATCITYY","NATCRIMY","NATDRUGY","NATEDUCY",
"NATRACEY","NATARMSY","NATAIDY","NATFAREY","EQWLTH","SPKATH","COLATH","LIBATH","SPKRAC","COLRAC","LIBRAC","SPKCOM","COLCOM","LIBCOM","SPKMIL","COLMIL","LIBMIL","SPKHOMO",
"COLHOMO","LIBHOMO","SPKMSLM","COLMSLM","LIBMSLM","CAPPUN","GUNLAW","COURTS","GRASS","ATTEND","RELITEN","POSTLIFE","PRAYER","AFFRMACT","WRKWAYUP","HELPFUL",
"FAIR","TRUST","CONFINAN","CONBUS","CONCLERG","CONEDUC","CONFED","CONLABOR","CONPRESS","CONMEDIC","CONTV","CONJUDGE","CONSCI","CONLEGIS","CONARMY","OBEY","POPULAR","THNKSELF","WORKHARD","HELPOTH","GETAHEAD","FEPOL","ABDEFECT","ABNOMORE","ABHLTH","ABPOOR","ABRAPE","ABSINGLE","ABANY","SEXEDUC","DIVLAW","PREMARSX","TEENSEX","XMARSEX","HOMOSEX","PORNLAW","SPANKING","LETDIE1","SUICIDE1","SUICIDE2","POLHITOK","POLABUSE","POLMURDR","POLESCAP","POLATTAK","NEWS","TVHOURS","FECHLD","FEPRESCH","FEFAM","RACDIF1","RACDIF2","RACDIF3","RACDIF4","HELPPOOR","MARHOMO"]

child_variables = ["OBEY","POPULAR","THNKSELF", "WORKHARD","HELPOTH","GETAHEAD"]

PRES_variables = [
    'VOTE68', 'VOTE72', 'VOTE76', 'VOTE80', 'VOTE84', 'VOTE88', 'VOTE92', 'VOTE96', 'VOTE00', 'VOTE04', 'VOTE08', 'VOTE12', 'VOTE16', 'VOTE20',
    'VOTE68_ELIGIBLE', 'VOTE72_ELIGIBLE', 'VOTE76_ELIGIBLE', 'VOTE80_ELIGIBLE', 'VOTE84_ELIGIBLE', 'VOTE88_ELIGIBLE', 'VOTE92_ELIGIBLE', 'VOTE96_ELIGIBLE', 'VOTE00_ELIGIBLE', 'VOTE04_ELIGIBLE', 'VOTE08_ELIGIBLE', 'VOTE12_ELIGIBLE', 'VOTE16_ELIGIBLE', 'VOTE20_ELIGIBLE',
    'PRES68_HUMPHREY', 'PRES68_NIXON', 'PRES68_WALLACE', 'PRES68_OTHER', 'PRES68_REFUSED',
    'PRES72_MCGOVERN', 'PRES72_NIXON', 'PRES72_OTHER', 'PRES72_REFUSED', 'PRES72_WOULDNT_VOTE', 'PRES72_DONT_KNOW',
    'PRES76_CARTER', 'PRES76_FORD', 'PRES76_OTHER', 'PRES76_REFUSED', 'PRES76_NO_PRES_VOTE', 'PRES76_DONT_KNOW',
    'PRES80_CARTER', 'PRES80_REAGAN', 'PRES80_ANDERSON', 'PRES80_OTHER', 'PRES80_REFUSED', 'PRES80_DIDNT_VOTE', 'PRES80_DONT_KNOW',
    'PRES84_MONDALE', 'PRES84_REAGAN', 'PRES84_OTHER', 'PRES84_REFUSED', 'PRES84_NO_PRES_VOTE', 'PRES84_DONT_KNOW',
    'PRES88_BUSH', 'PRES88_DUKAKIS', 'PRES88_OTHER', 'PRES88_REFUSED', 'PRES88_NO_PRES_VOTE', 'PRES88_DONT_KNOW',
    'PRES92_CLINTON', 'PRES92_BUSH', 'PRES92_PEROT', 'PRES92_OTHER', 'PRES92_NO_PRES_VOTE', 'PRES92_DONT_KNOW',
    'PRES96_CLINTON', 'PRES96_DOLE', 'PRES96_PEROT', 'PRES96_OTHER', 'PRES96_DIDNT_VOTE', 'PRES96_DONT_KNOW',
    'PRES00_GORE', 'PRES00_BUSH', 'PRES00_NADER', 'PRES00_OTHER', 'PRES00_DIDNT_VOTE', 'PRES00_DONT_KNOW',
    'PRES04_KERRY', 'PRES04_BUSH', 'PRES04_NADER', 'PRES04_NO_PRES_VOTE', 'PRES04_DONT_KNOW',
    'PRES08_OBAMA', 'PRES08_MCCAIN', 'PRES08_OTHER', 'PRES08_DIDNT_VOTE', 'PRES08_DONT_KNOW',
    'PRES12_OBAMA', 'PRES12_ROMNEY', 'PRES12_OTHER', 'PRES12_DIDNT_VOTE', 'PRES12_DONT_KNOW',
    'PRES16_CLINTON', 'PRES16_TRUMP', 'PRES16_OTHER', 'PRES16_DIDNT_VOTE', 'PRES16_DONT_KNOW',
    'PRES20_BIDEN', 'PRES20_TRUMP', 'PRES20_OTHER', 'PRES20_DIDNT_VOTE', 'PRES20_DONT_KNOW',
    'IF68WHO_HUMPHREY', 'IF68WHO_NIXON', 'IF68WHO_WALLACE', 'IF68WHO_OTHER', 'IF68WHO_WLDNT_VT_RELIG', 'IF68WHO_DONT_KNOW',
    'IF72WHO_MCGOVERN', 'IF72WHO_NIXON', 'IF72WHO_OTHER', 'IF72WHO_REFUSED', 'IF72WHO_WOULDNT_VOTE', 'IF72WHO_WLDNT_VT_RELIG', 'IF72WHO_DONT_KNOW',
    'IF76WHO_CARTER', 'IF76WHO_FORD', 'IF76WHO_OTHER', 'IF76WHO_REFUSED', 'IF76WHO_WOULDNT_VOTE', 'IF76WHO_DONT_KNOW',
    'IF80WHO_CARTER', 'IF80WHO_REAGAN', 'IF80WHO_ANDERSON', 'IF80WHO_OTHER', 'IF80WHO_WOULDNT_VOTE', 'IF80WHO_REFUSED', 'IF80WHO_DONT_KNOW',
    'IF84WHO_MONDALE', 'IF84WHO_REAGAN', 'IF84WHO_OTHER', 'IF84WHO_WOULDNT_VOTE', 'IF84WHO_DONT_KNOW',
    'IF88WHO_DUKAKIS', 'IF88WHO_BUSH', 'IF88WHO_OTHER', 'IF88WHO_DONT_KNOW',
    'IF92WHO_CLINTON', 'IF92WHO_BUSH', 'IF92WHO_PEROT', 'IF92WHO_OTHER', 'IF92WHO_DONT_KNOW',
    'IF96WHO_CLINTON', 'IF96WHO_DOLE', 'IF96WHO_PEROT', 'IF96WHO_OTHER', 'IF96WHO_DONT_KNOW',
    'IF00WHO_GORE', 'IF00WHO_BUSH', 'IF00WHO_NADER', 'IF00WHO_OTHER', 'IF00WHO_DONT_KNOW',
    'IF04WHO_KERRY', 'IF04WHO_BUSH', 'IF04WHO_NADER', 'IF04WHO_DONT_KNOW',
    'IF08WHO_OBAMA', 'IF08WHO_MCCAIN', 'IF08WHO_OTHER', 'IF08WHO_DONT_KNOW',
    'IF12WHO_OBAMA', 'IF12WHO_ROMNEY', 'IF12WHO_OTHER', 'IF12WHO_DONT_KNOW',
    'IF16WHO_CLINTON', 'IF16WHO_TRUMP', 'IF16WHO_OTHER', 'IF16WHO_CANT_REMEMBER', 'IF16WHO_DONT_KNOW',
    'IF20WHO_BIDEN', 'IF20WHO_TRUMP', 'IF20WHO_OTHER', 'IF20WHO_CANT_REMEMBER', 'IF20WHO_DONT_KNOW'
]

# Generating corr network
What do hope from the corr networks? That
1. they are not too sensitive to minor fluctuations in data
2. edges denote genuine interactions between variables
3. they are not too dense
4. that there is persistent structure across the windows but some level of change in each

questions:
1. should we include political party and liberal/conservative?
2. how do we deal with cases where variables are perfectly correlated (like the variables about raising children) so they don't have any partial correlations with the other ones. somehow for each variable I need to be able to say these are the ones I want to control for in the partial correlation, but of course then I cannot use the matrix inversion method which doesn't account for that stuff

# Issues that have been important over the years
from here: https://www.nytimes.com/interactive/2017/02/27/us/politics/most-important-problem-gallup-polling-question.html

Gallup poll asking "What do you think is the most important problem facing this country today?” since 1935


```
[
  'budget',
   'civil rights and race',
   'civil rights and race relations',
   'corruption',
   'crime',
   'dissatisfaction with government',
   'drugs',
   'economy in general',
   'education',
   'energy',
   'healthcare',
   'immigration',
   'inflation',
   'lack of money',
   'moral decline',
   'poverty',
   'terrorism',
   'unemployment',
   'unification of country',
   'war'
 ]
```

