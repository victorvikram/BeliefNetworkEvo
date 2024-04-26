import pandas as pd
import pyreadstat as prs
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# The purpose of this script is to read in the GSS dataset and perform some basic data cleaning and transformation.
# The dataset is a SAS7BDAT file, so we will use the pandas library to read in the data.
# The basic idea here is to manually write the mappings for the variables in the dataset and then apply them to the dataset.

def transform_dataframe(df):
    # Read the data
    # The variables of interest are:
    column_codes = ["YEAR","PARTYID","VOTE68","PRES68","IF68WHO","VOTE72","PRES72","IF72WHO","VOTE76","PRES76","IF76WHO","VOTE80","PRES80","IF80WHO","VOTE84","PRES84",
    "IF84WHO","VOTE88","PRES88","IF88WHO","VOTE92","PRES92","IF92WHO","VOTE96","PRES96","IF96WHO","VOTE00","PRES00","IF00WHO","VOTE04","PRES04","IF04WHO","VOTE08","PRES08",
    "IF08WHO","VOTE12","PRES12","IF12WHO","VOTE16","PRES16","IF16WHO","VOTE20", "PRES20", "IF20WHO","POLVIEWS","NATSPAC","NATENVIR","NATHEAL","NATCITY","NATCRIME","NATDRUG","NATEDUC","NATRACE","NATARMS",
    "NATAID","NATFARE","NATROAD","NATSOC","NATMASS","NATPARK","NATCHLD","NATSCI","NATENRGY","NATSPACY","NATENVIY","NATHEALY","NATCITYY","NATCRIMY","NATDRUGY","NATEDUCY",
    "NATRACEY","NATARMSY","NATAIDY","NATFAREY","EQWLTH","SPKATH","COLATH","LIBATH","SPKRAC","COLRAC","LIBRAC","SPKCOM","COLCOM","LIBCOM","SPKMIL","COLMIL","LIBMIL","SPKHOMO",
    "COLHOMO","LIBHOMO","SPKMSLM","COLMSLM","LIBMSLM","CAPPUN","GUNLAW","COURTS","GRASS","RELIG","ATTEND","RELITEN","POSTLIFE","PRAYER","RACOPEN","AFFRMACT","WRKWAYUP","HELPFUL",
    "FAIR","TRUST","CONFINAN","CONBUS","CONCLERG","CONEDUC","CONFED","CONLABOR","CONPRESS","CONMEDIC","CONTV","CONJUDGE","CONSCI","CONLEGIS","CONARMY","OBEY","POPULAR","THNKSELF",
    "WORKHARD","HELPOTH","GETAHEAD","FEPOL","ABDEFECT","ABNOMORE","ABHLTH","ABPOOR","ABRAPE","ABSINGLE","ABANY","SEXEDUC","DIVLAW","PREMARSX","TEENSEX","XMARSEX","HOMOSEX","PORNLAW",
    "SPANKING","LETDIE1","SUICIDE1","SUICIDE2","POLHITOK","POLABUSE","POLMURDR","POLESCAP","POLATTAK","NEWS","TVHOURS","FECHLD","FEPRESCH","FEFAM","RACDIF1","RACDIF2","RACDIF3",
    "RACDIF4","HELPPOOR","HELPNOT","HELPBLK","MARHOMO","BALLOT"]

    # Setup the mappings
    # PARTYID: Generally speaking, do you usually think of yourself as a Republican, Democrat, Independent, or what?
    # SIGNED 
    # negative DEMOCRAT --- positive REPUBLICAN
    # also add category for OTHER
    PARTYID_map = {0: -3, 1: -2, 2: -1, 3: 0, 4: 1, 5: 2, 6: 3}
    other_map = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 1}



    # VOTE68, VOTE72, VOTE76, VOTE80, VOTE84, VOTE88, VOTE92, VOTE96, VOTE00, VOTE04, VOTE08, VOTE12, VOTE16, VOTE20
    # Did r vote in <YEAR> election?
    # Three BINARY categories: "Did they vote?", "Were they eligible?", and "Don't know/remember".
    # 0 is NO --- 1 is YES
    VOTE_map = {1: 1, 2: 0}
    ELIGIBLE_map = {1: 1, 2: 1, 3: 0}
    DONT_KNOW_map = {-98: 1, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0}


    # PRES68, PRES72, PRES76, PRES80, PRES84, PRES88, PRES92, PRES96, PRES00, PRES04, PRES08, PRES12, PRES16...
    # For whom did r vote for?
    # Binary category maps for all the listed candidates of the year.
    # 1 is for the candidate, 0 is not for the candidate.
    category_map_A = {1: 1, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0} # For mapping to the first option
    category_map_B = {1: 0, 2: 1, 3: 0, 4: 0, 5: 0, 6: 0} # For mapping to the second option...
    category_map_C = {1: 0, 2: 0, 3: 1, 4: 0, 5: 0, 6: 0}
    category_map_D = {1: 0, 2: 0, 3: 0, 4: 1, 5: 0, 6: 0}
    category_map_E = {1: 0, 2: 0, 3: 0, 4: 0, 5: 1, 6: 0}
    category_map_F = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 1}

    # POLVIEWS: Does r think of self as liberal or conservative?
    # SIGNED
    # negative LIBERAL --- positive CONSERVATIVE
    POLVIEWS_map = {1: -3, 2: -2, 3: -1, 4: 0, 5: 1, 6: 2, 7: 3} 

    # NATSPAC, NATENVIR, NATHEAL, NATCITY, NATCRIME, NATDRUG, NATEDUC, NATRACE, NATARMS, NATAID, NATFARE, NATROAD, NATSOC, NATMASS, NATPARK, NATCHLD, NATSCI, NATENRGY, NATSPACY, NATENVIY, NATHEALY, NATCITYY, NATCRIMY, NATDRUGY, NATEDUCY, NATRACEY, NATARMSY, NATAIDY, NATFAREY
    # If Government spending too little, the right amount, or too much on <issue>?
    # negative TOO LITTLE --- zero JUST RIGHT --- positive TOO MUCH
    NAT_map = {1: -1, 2: 0, 3: 1}

    # EQWLTH: Should govt reduce income differences?
    # UNSIGNED
    # zero NO --- positive YES
    EQWLTH_map = {1: 6, 2: 5, 3: 4, 4: 3, 5: 2, 6: 1, 7: 0}
    #DONT_KNOW_map works here

    # Speaking variables: SPKATH, SPKRAC, SPKCOM, SPKMIL, SPKHOMO, SPKMSLM
    # Can <person> speak in community?
    # zero is NO --- one is YES
    SPK_map = {1: 1, 2: 0}

    # Teaching variables: COLATH, COLRAC, COLCOM, COLMIL, COLHOMO, COLMSLM
    # Can <person> teach in college?
    # zero is NO --- one is YES (EXCEPT FOR COLCOM (communists), for this, zero is YES and one is NO!!)
    COLATH_map = {4: 1, 5: 0}

    # Library book variables: LIBATH, LIBRAC, LIBCOM, LIBMIL, LIBHOMO, LIBMSLM
    # Would remove book on <topic> from library?
    # zero WOULD NOT REMOVE --- positive WOULD REMOVE
    LIB_map = {1: 1, 2: 0}

    # CAPPUN and GUNLAW
    # Do you favor or oppose the death penalty for persons convicted of murder?
    # Would you favor or oppose a law which would require a person to obtain a police permit before he or she could buy a gun?
    # BINARY
    # zero OPPOSE --- positive FAVOUR
    POL1_map = {1: 1, 2: 0}

    # COURTS: Courts deal too harshly or not harshly enough with criminals?
    # SIGNED
    # negative TOO HARSH --- zero JUST RIGHT --- positive NEEDS TO BE HARSHER
    COURTS_map = {1: -1, 2: 0, 3: 1}

    # GRASS: Should the use of marijuana should be made legal or not?
    # BINARY
    # zero ILLEGAL --- positive LEGAL
    GRASS_map = {1: 1, 2: 0}

    # RELIG: What is your religious preference?
    # BINARY categories for various religions.
    # zero NOT THIS RELIGION --- positive THIS RELIGION
    category_map_A = {1: 1, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0}
    category_map_B = {1: 0, 2: 1, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0}
    category_map_C = {1: 0, 2: 0, 3: 1, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0}
    category_map_D = {1: 0, 2: 0, 3: 0, 4: 1, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0}
    category_map_E = {1: 0, 2: 0, 3: 0, 4: 0, 5: 1, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0}
    category_map_F = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 1, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0}
    category_map_G = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 1, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0}
    category_map_H = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 1, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0}
    category_map_I = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 1, 10: 0, 11: 0, 12: 0, 13: 0}
    category_map_J = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 1, 11: 0, 12: 0, 13: 0}
    category_map_K = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 1, 12: 0, 13: 0}
    category_map_L = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 1, 13: 0}
    category_map_M = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 1}

    # ATTEND: How often do you attend religious services? 
    # UNSIGNED
    # zero NEVER --- positive ALL THE TIME
    ATTEND_map = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8}

    # RELITEN: Would you call yourself a strongly religious or a not very strong?
    # UNSIGNED
    # zero NOT RELIGIOUS --- positive VERY RELIGIOUS
    RELITEN_map = {1: 3, 2: 2, 3: 1, 4: 0}
    NO_ANSWER_map = {-99: 1, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0}

    # POSTLIFE: Do you believe in life after death?
    # BINARY
    # zero DO NOT BELIEVE --- positive BELIEVE
    POSTLIFE_map = {1: 1, 2: 0}

    # PRAYER: Do you approve or disapprove that mandatory prayer is banned in public schools?
    # BINARY
    # zero DISAPPROVE BAN (dislikes prayer) --- positive APPROVE BAN (likes prayer)
    PRAYER_map = {1: 1, 2: 0}


    # RACOPEN
    ### IDK!!

    # AFFRMACT: Are you for or against preferential hiring and promotion of blacks? 
    # SIGNED
    # negative OPPOSED --- positive SUPPORT
    AFFRMACT_map = {1: 2, 2: 1, 3: -1, 4: -2}

    # WRKWAYUP: Irish, Italians, Jewish and many other minorities overcame prejudice and worked their way up. 
    # Blacks should do the same without special favors.
    # SIGNED
    # negative DISAGREE --- positive AGREE
    WRKWAYUP_map = {1: 2, 2: 1, 3: 0, 4: -1, 5: -2}


    # HELPFUL: Do you think people try to be helpful, or that they are mostly just looking out for themselves?
    # SIGNED
    # negative SELFISH --- zero DEPENDS --- positive HELPFUL
    HELPFUL_map = {1: 1, 2: -1, 3: 0}


    # FAIR: Do you think people would try to take advantage of you if they got a chance, or would they try to be fair?
    # SIGNED
    # negative WOULD EXPLOIT YOU --- zero DEPENDS --- positive WOULD BE FAIR
    FAIR_map = {2: 1, 1: -1, 3: 0}


    # TRUST: Can or can't most people be trusted?
    # SIGNED
    # negative CAN'T BE TRUSTED --- zero DEPENDS --- CAN BE TRUSTED
    TRUST_map = {1: 1, 2: -1, 3: 0}


    # Confidence variables: CONFINAN, CONBUS, CONCLERG, CONEDUC, CONFED, CONLABOR, CONPRESS, CONMEDIC, CONTV, CONJUDGE, CONSCI, CONLEGIS, CONARMY
    # How condident are you in <institution>?
    # UNSIGNED
    # zero HARDLY ANY CONFIDENCE --- positive VERY CONFIDENT
    CON_map = {1: 2, 2: 1, 3: 0}


    # Kid learning ranking: OBEY, POPULAR, THNKSELF, WORKHARD, HELPOTH
    # Rank the importance of <option> for kids to learn.
    # UNSIGNED
    # zero LEAST IMPORTANT --- positive MOST IMPORTANT
    KID_map = {1: 4, 2: 3, 3: 2, 4: 1, 5: 0}


    # GETAHEAD: Does hardwork or luck get you ahead in life?
    # SIGNED
    # negative HARD WORK --- zero BOTH EQUAL --- positive LUCK
    GETAHEAD_map = {1: -1, 2: 0, 3: 1}
    other_GETAHEAD_map = {1: 0, 2: 0, 3: 0, 4: 1}


    # FEPOL: Most men are better suited emotionally for politics than are most women.
    # BINARY
    # zero DISAGREE --- positive AGREE
    FEPOL_map = {1: 1, 2: 0}


    # Abortion questions: ABDEFECT, ABNOMORE, ABHLTH, ABPOOR, ABRAPE, ABSINGLE, ABANY
    # Is abortion okay if <case>?
    # BINARY
    # zero NO --- positive YES
    AB_map = {1: 1, 2: 0}


    # SEXEDUC: Sex education in the public schools?
    # zero AGAINST --- positive FOR
    SEXEDUC_map = {1: 1, 2: 0}


    # DIVLAW: Should divorce in this country be easier or more difficult to obtain than it is now?
    # SIGNED
    # negative HARDER --- zero SAME --- positive EASIER
    DIVLAW_map = {1: 1, 2: 0, 3: -1}


    # SEX: PREMARSX, TEENSEX, XMARSEX, HOMOSEX
    # Is <option> wrong?  
    # UNSIGNED
    # zero NEVER WRONG --- positive ALWAYS WRONG
    SEX_map = {1: 3, 2: 2, 3: 1, 4: 0}


    # PORNLAW: Should laws forbid porn?
    # UNSIGNED
    # zero NO LAWS --- positive STRONG LAWS
    PORNLAW_map = {1: 2, 2: 1, 3: 0}


    # SPANKING: It is sometimes necessary to discipline a child with a good, hard spanking.
    # SIGNED
    # negative DISAGREE --- positive AGREE
    SPANKING_map = {1: 2, 2: 1, 3: -1, 4: -2}


    # Death: LETDIE1, SUICIDE1, SUICIDE2
    # Is it ok to allow death in <situation>?
    # BINARY
    # zero NO --- positive YES
    DEATH_map = {1: 1, 2: 0}


    # Police: POLHITOK, POLABUSE, POLMURDR, POLESCAP, POLATTAK
    # Are there any situations where it's okay for police to <action>?
    # BINARY
    # zero NO --- positive YES
    POLICE_map = {1: 1, 2: 0}


    # NEWS: How often do you read the newspaper?
    # UNSIGNED
    # zero NEVER --- positive EVERYDAY
    NEWS_map = {1: 4, 2: 3, 3: 2, 4: 1, 5: 0}


    # TVHOURS: On average, how many hours per day on TV?
    # UNSIGNED
    # zero NONE --- positive 24
    TVHOURS_map = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10, 11:11, 12:12, 13:13, 14:14, 15:15, 16:16, 17:17, 18:18, 19:19, 20:20, 21:21, 22:22, 23:23, 24:24}


    # female: FECHLD, FEPRESCH, and FEFAM
    # Do you agree with <statement> (concerning women's rights)?
    # SIGNED
    # negative DISAGREE --- positive AGREE
    FE_map = {1: 2, 2: 1, 3: -1, 4: -2}


    # Racial differences: RACDIF1, RACDIF2, RACDIF3, RACDIF4
    # On the average (Negroes/Blacks/African-Americans) have worse jobs, income, and housing than white people. 
    # Do you think these differences are due to <option>? 
    # BINARY
    # zero NO DIFF --- positive YES DIFF
    RACDIF_map = {1: 1, 2: 0}


    # Help people: HELPPOOR, HELPNOT, HELPBLK
    # Should the government help <people>?
    # SIGNED
    # negative LET THEM HELP THEMSELF --- zero AGREE WITH BOTH --- positive YES GOVT ACTION
    HELP_map = {1: 2, 2: 1, 3: 0, 4: -1, 5: -2}


    # MARHOMO: Homosexual couples should have the right to marry one another.
    # SIGNED
    # negative DISAGREE --- positive AGREE
    MARHOMO_map = {1: 2, 2: 1, 3: 0, 4: -1, 5: -2}

    # region: APPLYING ALL THE MAPS
    df['PARTY_ID (mapped)'] = df['PARTYID'].map(PARTYID_map)
    df['OTHER_PARTY (mapped)'] = df['PARTYID'].map(other_map)

    df['VOTE68 (mapped)'] = df['VOTE68'].map(VOTE_map)
    df['VOTE72 (mapped)'] = df['VOTE72'].map(VOTE_map)
    df['VOTE76 (mapped)'] = df['VOTE76'].map(VOTE_map)
    df['VOTE80 (mapped)'] = df['VOTE80'].map(VOTE_map)
    df['VOTE84 (mapped)'] = df['VOTE84'].map(VOTE_map)
    df['VOTE88 (mapped)'] = df['VOTE88'].map(VOTE_map)
    df['VOTE92 (mapped)'] = df['VOTE92'].map(VOTE_map)
    df['VOTE96 (mapped)'] = df['VOTE96'].map(VOTE_map)
    df['VOTE00 (mapped)'] = df['VOTE00'].map(VOTE_map)
    df['VOTE04 (mapped)'] = df['VOTE04'].map(VOTE_map)
    df['VOTE08 (mapped)'] = df['VOTE08'].map(VOTE_map)
    df['VOTE12 (mapped)'] = df['VOTE12'].map(VOTE_map)
    df['VOTE16 (mapped)'] = df['VOTE16'].map(VOTE_map)
    df['VOTE20 (mapped)'] = df['VOTE20'].map(VOTE_map)

    df['VOTE68_ELIGIBLE (mapped)'] = df['VOTE68'].map(ELIGIBLE_map)
    df['VOTE72_ELIGIBLE (mapped)'] = df['VOTE72'].map(ELIGIBLE_map)
    df['VOTE76_ELIGIBLE (mapped)'] = df['VOTE76'].map(ELIGIBLE_map)
    df['VOTE80_ELIGIBLE (mapped)'] = df['VOTE80'].map(ELIGIBLE_map)
    df['VOTE84_ELIGIBLE (mapped)'] = df['VOTE84'].map(ELIGIBLE_map)
    df['VOTE88_ELIGIBLE (mapped)'] = df['VOTE88'].map(ELIGIBLE_map)
    df['VOTE92_ELIGIBLE (mapped)'] = df['VOTE92'].map(ELIGIBLE_map)
    df['VOTE96_ELIGIBLE (mapped)'] = df['VOTE96'].map(ELIGIBLE_map)
    df['VOTE00_ELIGIBLE (mapped)'] = df['VOTE00'].map(ELIGIBLE_map)
    df['VOTE04_ELIGIBLE (mapped)'] = df['VOTE04'].map(ELIGIBLE_map)
    df['VOTE08_ELIGIBLE (mapped)'] = df['VOTE08'].map(ELIGIBLE_map)
    df['VOTE12_ELIGIBLE (mapped)'] = df['VOTE12'].map(ELIGIBLE_map)
    df['VOTE16_ELIGIBLE (mapped)'] = df['VOTE16'].map(ELIGIBLE_map)
    df['VOTE20_ELIGIBLE (mapped)'] = df['VOTE20'].map(ELIGIBLE_map)

    df['VOTE68 (dont know)'] = df['VOTE68'].map(DONT_KNOW_map)
    df['VOTE72 (dont know)'] = df['VOTE72'].map(DONT_KNOW_map)
    df['VOTE76 (dont know)'] = df['VOTE76'].map(DONT_KNOW_map)
    df['VOTE80 (dont know)'] = df['VOTE80'].map(DONT_KNOW_map)
    df['VOTE84 (dont know)'] = df['VOTE84'].map(DONT_KNOW_map)
    df['VOTE88 (dont know)'] = df['VOTE88'].map(DONT_KNOW_map)
    df['VOTE92 (dont know)'] = df['VOTE92'].map(DONT_KNOW_map)
    df['VOTE96 (dont know)'] = df['VOTE96'].map(DONT_KNOW_map)
    df['VOTE00 (dont know)'] = df['VOTE00'].map(DONT_KNOW_map)
    df['VOTE04 (dont know)'] = df['VOTE04'].map(DONT_KNOW_map)
    df['VOTE08 (dont know)'] = df['VOTE08'].map(DONT_KNOW_map)
    df['VOTE12 (dont know)'] = df['VOTE12'].map(DONT_KNOW_map)
    df['VOTE16 (dont know)'] = df['VOTE16'].map(DONT_KNOW_map)
    df['VOTE20 (dont know)'] = df['VOTE20'].map(DONT_KNOW_map)

    df['PRES68 (HUMPHREY)'] = df['PRES68'].map(category_map_A)
    df['PRES68 (NIXON)'] = df['PRES68'].map(category_map_B)
    df['PRES68 (WALLACE)'] = df['PRES68'].map(category_map_C)
    df['PRES68 (OTHER)'] = df['PRES68'].map(category_map_D)
    df['PRES68 (REFUSED)'] = df['PRES68'].map(category_map_E)
    df['PRES68 (dont know)'] = df['PRES68'].map(DONT_KNOW_map)


    df['PRES72 (MCGOVERN)'] = df['PRES72'].map(category_map_A)
    df['PRES72 (NIXON)'] = df['PRES72'].map(category_map_B)
    df['PRES72 (OTHER)'] = df['PRES72'].map(category_map_C)
    df['PRES72 (REFUSED)'] = df['PRES72'].map(category_map_D)
    df['PRES72 (WOULDNT VOTE)'] = df['PRES72'].map(category_map_E)
    df['PRES72 (dont know)'] = df['PRES72'].map(DONT_KNOW_map)

    df['PRES76 (CARTER)'] = df['PRES76'].map(category_map_A)
    df['PRES76 (FORD)'] = df['PRES76'].map(category_map_B)
    df['PRES76 (OTHER)'] = df['PRES76'].map(category_map_C)
    df['PRES76 (REFUSED)'] = df['PRES76'].map(category_map_D)
    df['PRES76 (NO. PRES VOTE)'] = df['PRES76'].map(category_map_E)
    df['PRES76 (dont know)'] = df['PRES76'].map(DONT_KNOW_map)

    df['PRES80 (CARTER)'] = df['PRES80'].map(category_map_A)
    df['PRES80 (REAGAN)'] = df['PRES80'].map(category_map_B)
    df['PRES80 (ANDERSON)'] = df['PRES80'].map(category_map_C)
    df['PRES80 (OTHER)'] = df['PRES80'].map(category_map_D)
    df['PRES80 (REFUSED)'] = df['PRES80'].map(category_map_E)
    df['PRES80 (DIDNT VOTE)'] = df['PRES80'].map(category_map_F)
    df['PRES80 (dont know)'] = df['PRES80'].map(DONT_KNOW_map)

    df['PRES84 (MONDALE)'] = df['PRES84'].map(category_map_A)
    df['PRES84 (REAGAN)'] = df['PRES84'].map(category_map_B)
    df['PRES84 (OTHER)'] = df['PRES84'].map(category_map_C)
    df['PRES84 (REFUSED)'] = df['PRES84'].map(category_map_D)
    df['PRES84 (NO. PRES VOTE)'] = df['PRES84'].map(category_map_E)
    df['PRES84 (dont know)'] = df['PRES84'].map(DONT_KNOW_map)

    df['PRES88 (BUSH)'] = df['PRES88'].map(category_map_A)
    df['PRES88 (DUKAKIS)'] = df['PRES88'].map(category_map_B)
    df['PRES88 (OTHER)'] = df['PRES88'].map(category_map_C)
    df['PRES88 (REFUSED)'] = df['PRES88'].map(category_map_D)
    df['PRES88 (NO. PRES VOTE)'] = df['PRES88'].map(category_map_E)
    df['PRES88 (dont know)'] = df['PRES88'].map(DONT_KNOW_map)

    df['PRES92 (CLINTON)'] = df['PRES92'].map(category_map_A)
    df['PRES92 (BUSH)'] = df['PRES92'].map(category_map_B)
    df['PRES92 (PEROT)'] = df['PRES92'].map(category_map_C)
    df['PRES92 (OTHER)'] = df['PRES92'].map(category_map_D)
    df['PRES92 (NO. PRES VOTE)'] = df['PRES92'].map(category_map_E)
    df['PRES92 (dont know)'] = df['PRES92'].map(DONT_KNOW_map)

    df['PRES96 (CLINTON)'] = df['PRES96'].map(category_map_A)
    df['PRES96 (DOLE)'] = df['PRES96'].map(category_map_B)
    df['PRES96 (PEROT)'] = df['PRES96'].map(category_map_C)
    df['PRES96 (OTHER)'] = df['PRES96'].map(category_map_D)
    df['PRES96 (DIDNT VOTE VOTE)'] = df['PRES96'].map(category_map_E)
    df['PRES96 (dont know)'] = df['PRES96'].map(DONT_KNOW_map)

    df['PRES00 (GORE)'] = df['PRES00'].map(category_map_A)
    df['PRES00 (BUSH)'] = df['PRES00'].map(category_map_B)
    df['PRES00 (NADER)'] = df['PRES00'].map(category_map_C)
    df['PRES00 (OTHER)'] = df['PRES00'].map(category_map_D)
    df['PRES00 (DIDNT VOTE)'] = df['PRES00'].map(category_map_E)
    df['PRES00 (dont know)'] = df['PRES00'].map(DONT_KNOW_map)

    df['PRES04 (KERRY)'] = df['PRES04'].map(category_map_A)
    df['PRES04 (BUSH)'] = df['PRES04'].map(category_map_B)
    df['PRES04 (NADER)'] = df['PRES04'].map(category_map_C)
    df['PRES04 (NO. PRES VOTE)'] = df['PRES04'].map(category_map_D)
    df['PRES04 (dont know)'] = df['PRES04'].map(DONT_KNOW_map)

    df['PRES08 (OBAMA)'] = df['PRES08'].map(category_map_A)
    df['PRES08 (MCCAIN)'] = df['PRES08'].map(category_map_B)
    df['PRES08 (OTHER)'] = df['PRES08'].map(category_map_C)
    df['PRES08 (DIDNT VOTE)'] = df['PRES08'].map(category_map_D)
    df['PRES08 (dont know)'] = df['PRES08'].map(DONT_KNOW_map)

    df['PRES12 (OBAMA)'] = df['PRES12'].map(category_map_A)
    df['PRES12 (ROMNEY)'] = df['PRES12'].map(category_map_B)
    df['PRES12 (OTHER)'] = df['PRES12'].map(category_map_C)
    df['PRES12 (DIDNT VOTE)'] = df['PRES12'].map(category_map_D)
    df['PRES12 (dont know)'] = df['PRES12'].map(DONT_KNOW_map)

    df['PRES16 (CLINTON)'] = df['PRES16'].map(category_map_A)
    df['PRES16 (TRUMP)'] = df['PRES16'].map(category_map_B)
    df['PRES16 (OTHER)'] = df['PRES16'].map(category_map_C)
    df['PRES16 (DIDNT VOTE)'] = df['PRES16'].map(category_map_D)
    df['PRES16 (dont know)'] = df['PRES16'].map(DONT_KNOW_map)

    df['PRES20 (BIDEN)'] = df['PRES20'].map(category_map_A)
    df['PRES20 (TRUMP)'] = df['PRES20'].map(category_map_B)
    df['PRES20 (OTHER)'] = df['PRES20'].map(category_map_C)
    df['PRES20 (DIDNT VOTE)'] = df['PRES20'].map(category_map_D)
    df['PRES20 (dont know)'] = df['PRES20'].map(DONT_KNOW_map)

    df['IF68WHO (HUMPHREY)'] = df['IF68WHO'].map(category_map_A)
    df['IF68WHO (NIXON)'] = df['IF68WHO'].map(category_map_B)
    df['IF68WHO (WALLACE)'] = df['IF68WHO'].map(category_map_C)
    df['IF68WHO (OTHER)'] = df['IF68WHO'].map(category_map_D)
    df['IF68WHO (WLDNT VT-RELIG)'] = df['IF68WHO'].map(category_map_E)
    df['IF68WHO (dont know)'] = df['IF68WHO'].map(DONT_KNOW_map)

    df['IF72WHO (MCGOVERN)'] = df['IF72WHO'].map(category_map_A)
    df['IF72WHO (NIXON)'] = df['IF72WHO'].map(category_map_B)
    df['IF72WHO (OTHER)'] = df['IF72WHO'].map(category_map_C)
    df['IF72WHO (REFUSED)'] = df['IF72WHO'].map(category_map_D)
    df['IF72WHO (WOULDNT VOTE)'] = df['IF72WHO'].map(category_map_E)
    df['IF72WHO (WLDNT VT-RELIG)'] = df['IF72WHO'].map(category_map_F)
    df['IF72WHO (dont know)'] = df['IF72WHO'].map(DONT_KNOW_map)

    df['IF76WHO (CARTER)'] = df['IF76WHO'].map(category_map_A)
    df['IF76WHO (FORD)'] = df['IF76WHO'].map(category_map_B)
    df['IF76WHO (OTHER)'] = df['IF76WHO'].map(category_map_C)
    df['IF76WHO (REFUSED)'] = df['IF76WHO'].map(category_map_D)
    df['IF76WHO (WOULDNT VOTE)'] = df['IF76WHO'].map(category_map_E)
    df['IF76WHO (dont know)'] = df['IF76WHO'].map(DONT_KNOW_map)

    df['IF80WHO (CARTER)'] = df['IF80WHO'].map(category_map_A)
    df['IF80WHO (REAGAN)'] = df['IF80WHO'].map(category_map_B)
    df['IF80WHO (ANDERSON)'] = df['IF80WHO'].map(category_map_C)
    df['IF80WHO (OTHER)'] = df['IF80WHO'].map(category_map_D)
    df['IF80WHO (WOULDNT VOTE)'] = df['IF80WHO'].map(category_map_E)
    df['IF80WHO (REFUSED)'] = df['IF80WHO'].map(category_map_F)
    df['IF80WHO (dont know)'] = df['IF80WHO'].map(DONT_KNOW_map)

    df['IF84WHO (MONDALE)'] = df['IF84WHO'].map(category_map_A)
    df['IF84WHO (REAGAN)'] = df['IF84WHO'].map(category_map_B)
    df['IF84WHO (OTHER)'] = df['IF84WHO'].map(category_map_C)
    df['IF84WHO (WOULDNT VOTE)'] = df['IF84WHO'].map(category_map_D)
    df['IF84WHO (dont know)'] = df['IF84WHO'].map(DONT_KNOW_map)

    df['IF88WHO (DUKAKIS)'] = df['IF88WHO'].map(category_map_A)
    df['IF88WHO (BUSH)'] = df['IF88WHO'].map(category_map_B)
    df['IF88WHO (OTHER)'] = df['IF88WHO'].map(category_map_C)
    df['IF88WHO (dont know)'] = df['IF88WHO'].map(DONT_KNOW_map)

    df['IF92WHO (CLINTON)'] = df['IF92WHO'].map(category_map_A)
    df['IF92WHO (BUSH)'] = df['IF92WHO'].map(category_map_B)
    df['IF92WHO (PEROT)'] = df['IF92WHO'].map(category_map_C)
    df['IF92WHO (OTHER)'] = df['IF92WHO'].map(category_map_D)
    df['IF92WHO (dont know)'] = df['IF92WHO'].map(DONT_KNOW_map)

    df['IF96WHO (CLINTON)'] = df['IF96WHO'].map(category_map_A)
    df['IF96WHO (DOLE)'] = df['IF96WHO'].map(category_map_B)
    df['IF96WHO (PEROT)'] = df['IF96WHO'].map(category_map_C)
    df['IF96WHO (OTHER)'] = df['IF96WHO'].map(category_map_D)
    df['IF96WHO (dont know)'] = df['IF96WHO'].map(DONT_KNOW_map)

    df['IF00WHO (GORE)'] = df['IF00WHO'].map(category_map_A)
    df['IF00WHO (BUSH)'] = df['IF00WHO'].map(category_map_B)
    df['IF00WHO (NADER)'] = df['IF00WHO'].map(category_map_C)
    df['IF00WHO (OTHER)'] = df['IF00WHO'].map(category_map_D)
    df['IF00WHO (dont know)'] = df['IF00WHO'].map(DONT_KNOW_map)

    df['IF04WHO (KERRY)'] = df['IF04WHO'].map(category_map_A)
    df['IF04WHO (BUSH)'] = df['IF04WHO'].map(category_map_B)
    df['IF04WHO (NADER)'] = df['IF04WHO'].map(category_map_C)
    df['IF04WHO (dont know)'] = df['IF04WHO'].map(DONT_KNOW_map)

    df['IF08WHO (OBAMA)'] = df['IF08WHO'].map(category_map_A)
    df['IF08WHO (MCCAIN)'] = df['IF08WHO'].map(category_map_B)
    df['IF08WHO (OTHER)'] = df['IF08WHO'].map(category_map_C)
    df['IF08WHO (dont know)'] = df['IF08WHO'].map(DONT_KNOW_map)

    df['IF12WHO (OBAMA)'] = df['IF12WHO'].map(category_map_A)
    df['IF12WHO (ROMNEY)'] = df['IF12WHO'].map(category_map_B)
    df['IF12WHO (OTHER)'] = df['IF12WHO'].map(category_map_C)
    df['IF12WHO (dont know)'] = df['IF12WHO'].map(DONT_KNOW_map)

    df['IF16WHO (CLINTON)'] = df['IF16WHO'].map(category_map_A)
    df['IF16WHO (TRUMP)'] = df['IF16WHO'].map(category_map_B)
    df['IF16WHO (OTHER)'] = df['IF16WHO'].map(category_map_C)
    df['IF16WHO (CANT REMEMBER)'] = df['IF16WHO'].map(category_map_D)
    df['IF16WHO (dont know)'] = df['IF16WHO'].map(DONT_KNOW_map)

    df['POLVIEWS (mapped)'] = df['POLVIEWS'].map(POLVIEWS_map)

    df['NATSPAC (mapped)'] = df['NATSPAC'].map(NAT_map)
    df['NATENVIR (mapped)'] = df['NATENVIR'].map(NAT_map)
    df['NATHEAL (mapped)'] = df['NATHEAL'].map(NAT_map)
    df['NATCITY (mapped)'] = df['NATCITY'].map(NAT_map)
    df['NATCRIME (mapped)'] = df['NATCRIME'].map(NAT_map)
    df['NATDRUG (mapped)'] = df['NATDRUG'].map(NAT_map)
    df['NATEDUC (mapped)'] = df['NATEDUC'].map(NAT_map)
    df['NATRACE (mapped)'] = df['NATRACE'].map(NAT_map)
    df['NATARMS (mapped)'] = df['NATARMS'].map(NAT_map)
    df['NATAID (mapped)'] = df['NATAID'].map(NAT_map)
    df['NATFARE (mapped)'] = df['NATFARE'].map(NAT_map)
    df['NATROAD (mapped)'] = df['NATROAD'].map(NAT_map)
    df['NATSOC (mapped)'] = df['NATSOC'].map(NAT_map)
    df['NATMASS (mapped)'] = df['NATMASS'].map(NAT_map)
    df['NATPARK (mapped)'] = df['NATPARK'].map(NAT_map)
    df['NATCHLD (mapped)'] = df['NATCHLD'].map(NAT_map)
    df['NATSCI (mapped)'] = df['NATSCI'].map(NAT_map)
    df['NATENRGY (mapped)'] = df['NATENRGY'].map(NAT_map)
    df['NATSPACY (mapped)'] = df['NATSPACY'].map(NAT_map)
    df['NATENVIY (mapped)'] = df['NATENVIY'].map(NAT_map)
    df['NATHEALY (mapped)'] = df['NATHEALY'].map(NAT_map)
    df['NATCITYY (mapped)'] = df['NATCITYY'].map(NAT_map)
    df['NATCRIMY (mapped)'] = df['NATCRIMY'].map(NAT_map)
    df['NATDRUGY (mapped)'] = df['NATDRUGY'].map(NAT_map)
    df['NATEDUCY (mapped)'] = df['NATEDUCY'].map(NAT_map)
    df['NATRACEY (mapped)'] = df['NATRACEY'].map(NAT_map)
    df['NATARMSY (mapped)'] = df['NATARMSY'].map(NAT_map)
    df['NATAIDY (mapped)'] = df['NATAIDY'].map(NAT_map)
    df['NATFAREY (mapped)'] = df['NATFAREY'].map(NAT_map)

    df['EQWLTH (mapped)'] = df['EQWLTH'].map(EQWLTH_map)

    df['SPKATH (mapped)'] = df['SPKATH'].map(SPK_map)
    df['SPKRAC (mapped)'] = df['SPKRAC'].map(SPK_map)
    df['SPKCOM (mapped)'] = df['SPKCOM'].map(SPK_map)
    df['SPKMIL (mapped)'] = df['SPKMIL'].map(SPK_map)
    df['SPKHOMO (mapped)'] = df['SPKHOMO'].map(SPK_map)
    df['SPKMSLM (mapped)'] = df['SPKMSLM'].map(SPK_map)

    df['COLATH (mapped)'] = df['COLATH'].map(COLATH_map)
    df['COLRAC (mapped)'] = df['COLRAC'].map(COLATH_map)
    df['COLCOM (mapped)'] = df['COLCOM'].map(COLATH_map)
    df['COLMIL (mapped)'] = df['COLMIL'].map(COLATH_map)
    df['COLHOMO (mapped)'] = df['COLHOMO'].map(COLATH_map)
    df['COLMSLM (mapped)'] = df['COLMSLM'].map(COLATH_map)

    df['LIBATH (mapped)'] = df['LIBATH'].map(LIB_map)
    df['LIBRAC (mapped)'] = df['LIBRAC'].map(LIB_map)
    df['LIBCOM (mapped)'] = df['LIBCOM'].map(LIB_map)
    df['LIBMIL (mapped)'] = df['LIBMIL'].map(LIB_map)
    df['LIBHOMO (mapped)'] = df['LIBHOMO'].map(LIB_map)
    df['LIBMSLM (mapped)'] = df['LIBMSLM'].map(LIB_map)

    df['CAPPUN (mapped)'] = df['CAPPUN'].map(POL1_map)
    df['GUNLAW (mapped)'] = df['GUNLAW'].map(POL1_map)

    df['COURTS (mapped)'] = df['COURTS'].map(COURTS_map)

    df['GRASS (mapped)'] = df['GRASS'].map(GRASS_map)

    df['RELIG (PROT)'] = df['RELIG'].map(category_map_A)
    df['RELIG (CATHOLIC)'] = df['RELIG'].map(category_map_B)
    df['RELIG (JEWISH)'] = df['RELIG'].map(category_map_C)
    df['RELIG (NONE)'] = df['RELIG'].map(category_map_D)
    df['RELIG (OTHER)'] = df['RELIG'].map(category_map_E)
    df['RELIG (BUDDHISM)'] = df['RELIG'].map(category_map_F)
    df['RELIG (HINDUISM)'] = df['RELIG'].map(category_map_G)
    df['RELIG (Other eastern religions	)'] = df['RELIG'].map(category_map_H)
    df['RELIG (Muslim/islam)'] = df['RELIG'].map(category_map_I)
    df['RELIG (Orthodox-christian)'] = df['RELIG'].map(category_map_J)
    df['RELIG (Christian)'] = df['RELIG'].map(category_map_K)
    df['RELIG (Native american)'] = df['RELIG'].map(category_map_L)
    df['RELIG (Inter-nondenominational)'] = df['RELIG'].map(category_map_M)

    df['ATTEND (mapped)'] = df['ATTEND'].map(ATTEND_map)

    df['RELITEN (mapped)'] = df['RELITEN'].map(RELITEN_map)

    df['POSTLIFE (mapped)'] = df['POSTLIFE'].map(POSTLIFE_map)

    df['PRAYER (mapped)'] = df['PRAYER'].map(PRAYER_map)

    df['AFFRMACT (mapped)'] = df['AFFRMACT'].map(AFFRMACT_map)

    df['WRKWAYUP (mapped)'] = df['WRKWAYUP'].map(WRKWAYUP_map)

    df['HELPFUL (mapped)'] = df['HELPFUL'].map(HELPFUL_map)

    df['FAIR (mapped)'] = df['FAIR'].map(FAIR_map)

    df['TRUST (mapped)'] = df['TRUST'].map(TRUST_map)

    df['CONFINAN (mapped)'] = df['CONFINAN'].map(CON_map)
    df['CONBUS (mapped)'] = df['CONBUS'].map(CON_map)
    df['CONCLERG (mapped)'] = df['CONCLERG'].map(CON_map)
    df['CONEDUC (mapped)'] = df['CONEDUC'].map(CON_map)
    df['CONFED (mapped)'] = df['CONFED'].map(CON_map)
    df['CONLABOR (mapped)'] = df['CONLABOR'].map(CON_map)
    df['CONPRESS (mapped)'] = df['CONPRESS'].map(CON_map)
    df['CONMEDIC (mapped)'] = df['CONMEDIC'].map(CON_map)
    df['CONTV (mapped)'] = df['CONTV'].map(CON_map)
    df['CONJUDGE (mapped)'] = df['CONJUDGE'].map(CON_map)
    df['CONSCI (mapped)'] = df['CONSCI'].map(CON_map)
    df['CONLEGIS (mapped)'] = df['CONLEGIS'].map(CON_map)
    df['CONARMY (mapped)'] = df['CONARMY'].map(CON_map)

    df['OBEY (mapped)'] = df['OBEY'].map(KID_map)
    df['POPULAR (mapped)'] = df['POPULAR'].map(KID_map)
    df['THNKSELF (mapped)'] = df['THNKSELF'].map(KID_map)
    df['WORKHARD (mapped)'] = df['WORKHARD'].map(KID_map)
    df['HELPOTH (mapped)'] = df['HELPOTH'].map(KID_map)

    df['GETAHEAD (mapped)'] = df['GETAHEAD'].map(GETAHEAD_map)

    df['FEPOL (mapped)'] = df['FEPOL'].map(FEPOL_map)

    df['ABDEFECT (mapped)'] = df['ABDEFECT'].map(AB_map)
    df['ABNOMORE (mapped)'] = df['ABNOMORE'].map(AB_map)
    df['ABHLTH (mapped)'] = df['ABHLTH'].map(AB_map)
    df['ABPOOR (mapped)'] = df['ABPOOR'].map(AB_map)
    df['ABRAPE (mapped)'] = df['ABRAPE'].map(AB_map)
    df['ABSINGLE (mapped)'] = df['ABSINGLE'].map(AB_map)
    df['ABANY (mapped)'] = df['ABANY'].map(AB_map)


    df['SEXEDUC (mapped)'] = df['SEXEDUC'].map(SEXEDUC_map)

    df['DIVLAW (mapped)'] = df['DIVLAW'].map(DIVLAW_map)

    df['PREMARSX (mapped)'] = df['PREMARSX'].map(SEX_map)
    df['TEENSEX (mapped)'] = df['TEENSEX'].map(SEX_map)
    df['XMARSEX (mapped)'] = df['XMARSEX'].map(SEX_map)
    df['HOMOSEX (mapped)'] = df['HOMOSEX'].map(SEX_map)

    df['PORNLAW (mapped)'] = df['PORNLAW'].map(PORNLAW_map)

    df['SPANKING (mapped)'] = df['SPANKING'].map(SPANKING_map)

    df['LETDIE1 (mapped)'] = df['LETDIE1'].map(DEATH_map)
    df['SUICIDE1 (mapped)'] = df['SUICIDE1'].map(DEATH_map)
    df['SUICIDE2 (mapped)'] = df['SUICIDE2'].map(DEATH_map)

    df['POLHITOK (mapped)'] = df['POLHITOK'].map(POLICE_map)
    df['POLABUSE (mapped)'] = df['POLABUSE'].map(POLICE_map)
    df['POLMURDR (mapped)'] = df['POLMURDR'].map(POLICE_map)
    df['POLESCAP (mapped)'] = df['POLESCAP'].map(POLICE_map)
    df['POLATTAK (mapped)'] = df['POLATTAK'].map(POLICE_map)

    df['NEWS (mapped)'] = df['NEWS'].map(NEWS_map)

    df['TVHOURS (mapped)'] = df['TVHOURS'].map(TVHOURS_map)

    df['FECHLD (mapped)'] = df['FECHLD'].map(FE_map)
    df['FEPRESCH (mapped)'] = df['FEPRESCH'].map(FE_map)
    df['FEFAM (mapped)'] = df['FEFAM'].map(FE_map)

    df['RACDIF1 (mapped)'] = df['RACDIF1'].map(RACDIF_map)
    df['RACDIF2 (mapped)'] = df['RACDIF2'].map(RACDIF_map)
    df['RACDIF3 (mapped)'] = df['RACDIF3'].map(RACDIF_map)
    df['RACDIF4 (mapped)'] = df['RACDIF4'].map(RACDIF_map)

    df['HELPPOOR (mapped)'] = df['HELPPOOR'].map(HELP_map)

    df['MARHOMO (mapped)'] = df['MARHOMO'].map(MARHOMO_map)
    # endregion


    ### Uncomment this to print select columns to a text file.
    # with open('partyid.txt', 'w') as f:
    #     for index, row in df.iterrows():
    #         f.write(str(row['MARHOMO']) + ' ' + str(row['MARHOMO (mapped)']) + ' ' + str(row['PRES68 (dont know)']) + '\n')

    return df