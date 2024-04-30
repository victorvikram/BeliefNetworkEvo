import pandas as pd
import pyreadstat as prs
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



def make_variable_summary(df):
    """
    this function goes year by year and for each ballot puts in the percentage of non nan answers to the question
    """
    counts = df.groupby(['YEAR', 'BALLOT'], dropna=False).count()
    pcts = counts / counts["ID"]

    partially_complete_ballot_mask = (0 < pcts) & (pcts < 0.8)
    pc_row, pc_col = np.where(partially_complete_ballot_mask)
    partially_complete_ballot = [(partially_complete_ballot_mask.index[row_ind], partially_complete_ballot_mask.column[col_ind]) for (row_ind, col_ind) in zip(pc_row, pc_col)]

    return counts
    
# The purpose of this script is to read in the GSS dataset and perform some basic data cleaning and transformation.
# The dataset is a SAS7BDAT file, so we will use the pandas library to read in the data.
# The basic idea here is to manually write the mappings for the variables in the dataset and then apply them to the dataset.

def transform_dataframe(df, combine_variants=True):
    # Read the data
    # The variables of interest are:
    column_codes = ["YEAR", "ID", "PARTYID","VOTE68","PRES68","IF68WHO","VOTE72","PRES72","IF72WHO","VOTE76","PRES76","IF76WHO","VOTE80","PRES80","IF80WHO","VOTE84","PRES84",
    "IF84WHO","VOTE88","PRES88","IF88WHO","VOTE92","PRES92","IF92WHO","VOTE96","PRES96","IF96WHO","VOTE00","PRES00","IF00WHO","VOTE04","PRES04","IF04WHO","VOTE08","PRES08",
    "IF08WHO","VOTE12","PRES12","IF12WHO","VOTE16","PRES16","IF16WHO","VOTE20", "PRES20", "IF20WHO","POLVIEWS","NATSPAC","NATENVIR","NATHEAL","NATCITY","NATCRIME","NATDRUG","NATEDUC","NATRACE","NATARMS",
    "NATAID","NATFARE","NATROAD","NATSOC","NATMASS","NATPARK","NATCHLD","NATSCI","NATENRGY","NATSPACY","NATENVIY","NATHEALY","NATCITYY","NATCRIMY","NATDRUGY","NATEDUCY",
    "NATRACEY","NATARMSY","NATAIDY","NATFAREY","EQWLTH","SPKATH","COLATH","LIBATH","SPKRAC","COLRAC","LIBRAC","SPKCOM","COLCOM","LIBCOM","SPKMIL","COLMIL","LIBMIL","SPKHOMO",
    "COLHOMO","LIBHOMO","SPKMSLM","COLMSLM","LIBMSLM","CAPPUN","GUNLAW","COURTS","GRASS","RELIG","ATTEND","RELITEN","POSTLIFE","PRAYER","RACOPEN","AFFRMACT","WRKWAYUP","HELPFUL",
    "FAIR","TRUST","CONFINAN","CONBUS","CONCLERG","CONEDUC","CONFED","CONLABOR","CONPRESS","CONMEDIC","CONTV","CONJUDGE","CONSCI","CONLEGIS","CONARMY","OBEY","POPULAR","THNKSELF",
    "WORKHARD","HELPOTH","GETAHEAD","FEPOL","ABDEFECT","ABNOMORE","ABHLTH","ABPOOR","ABRAPE","ABSINGLE","ABANY","SEXEDUC","DIVLAW","PREMARSX","TEENSEX","XMARSEX","HOMOSEX","PORNLAW",
    "SPANKING","LETDIE1","SUICIDE1","SUICIDE2","POLHITOK","POLABUSE","POLMURDR","POLESCAP","POLATTAK","NEWS","TVHOURS","FECHLD","FEPRESCH","FEFAM","RACDIF1","RACDIF2","RACDIF3",
    "RACDIF4","HELPPOOR","HELPNOT","HELPBLK","MARHOMO","BALLOT"]

    variants = {
        "NATSPACY": "NATSPAC",
        "NATENVIY": "NATENVIY",
        "NATHEALY": "NATHEAL",
        "NATCITYY": "NATCITY",
        "NATCRIMY": "NATCRIME",
        "NATDRUGY": "NATDRUG",
        "NATEDUCY": "NATEDUC",
        "NATRACEY": "NATRACE",
        "NATARMSY": "NATARMS",
        "NATAIDY": "NATAID",
        "NATFAREY": "NATFARE"
    }

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
    # negative ALLOWED to discriminate --- 0 neither law or can't choose --- positive NOT ALLOWED 
    RACOPEN_map = {1: -1, 2: 1, 3: 0, -98: 0}

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


    # Make a new dataframe and add df['PARTYID'].map(PARTYID_map) to it
    transformed_df = pd.DataFrame()
    transformed_df['YEAR'] = df['YEAR']
    transformed_df['ID'] = df['ID']
    transformed_df['BALLOT'] = df['BALLOT']


    # region: APPLYING ALL THE MAPS
    transformed_df['PARTYID'] = df['PARTYID'].map(PARTYID_map)
    transformed_df['OTHER_PARTY'] = df['PARTYID'].map(other_map)

    transformed_df['VOTE68'] = df['VOTE68'].map(VOTE_map)
    transformed_df['VOTE72'] = df['VOTE72'].map(VOTE_map)
    transformed_df['VOTE76'] = df['VOTE76'].map(VOTE_map)
    transformed_df['VOTE80'] = df['VOTE80'].map(VOTE_map)
    transformed_df['VOTE84'] = df['VOTE84'].map(VOTE_map)
    transformed_df['VOTE88'] = df['VOTE88'].map(VOTE_map)
    transformed_df['VOTE92'] = df['VOTE92'].map(VOTE_map)
    transformed_df['VOTE96'] = df['VOTE96'].map(VOTE_map)
    transformed_df['VOTE00'] = df['VOTE00'].map(VOTE_map)
    transformed_df['VOTE04'] = df['VOTE04'].map(VOTE_map)
    transformed_df['VOTE08'] = df['VOTE08'].map(VOTE_map)
    transformed_df['VOTE12'] = df['VOTE12'].map(VOTE_map)
    transformed_df['VOTE16'] = df['VOTE16'].map(VOTE_map)
    transformed_df['VOTE20'] = df['VOTE20'].map(VOTE_map)

    transformed_df['VOTE68_ELIGIBLE'] = df['VOTE68'].map(ELIGIBLE_map)
    transformed_df['VOTE72_ELIGIBLE'] = df['VOTE72'].map(ELIGIBLE_map)
    transformed_df['VOTE76_ELIGIBLE'] = df['VOTE76'].map(ELIGIBLE_map)
    transformed_df['VOTE80_ELIGIBLE'] = df['VOTE80'].map(ELIGIBLE_map)
    transformed_df['VOTE84_ELIGIBLE'] = df['VOTE84'].map(ELIGIBLE_map)
    transformed_df['VOTE88_ELIGIBLE'] = df['VOTE88'].map(ELIGIBLE_map)
    transformed_df['VOTE92_ELIGIBLE'] = df['VOTE92'].map(ELIGIBLE_map)
    transformed_df['VOTE96_ELIGIBLE'] = df['VOTE96'].map(ELIGIBLE_map)
    transformed_df['VOTE00_ELIGIBLE'] = df['VOTE00'].map(ELIGIBLE_map)
    transformed_df['VOTE04_ELIGIBLE'] = df['VOTE04'].map(ELIGIBLE_map)
    transformed_df['VOTE08_ELIGIBLE'] = df['VOTE08'].map(ELIGIBLE_map)
    transformed_df['VOTE12_ELIGIBLE'] = df['VOTE12'].map(ELIGIBLE_map)
    transformed_df['VOTE16_ELIGIBLE'] = df['VOTE16'].map(ELIGIBLE_map)
    transformed_df['VOTE20_ELIGIBLE'] = df['VOTE20'].map(ELIGIBLE_map)

    transformed_df['VOTE68'] = df['VOTE68'].map(DONT_KNOW_map)
    transformed_df['VOTE72'] = df['VOTE72'].map(DONT_KNOW_map)
    transformed_df['VOTE76'] = df['VOTE76'].map(DONT_KNOW_map)
    transformed_df['VOTE80'] = df['VOTE80'].map(DONT_KNOW_map)
    transformed_df['VOTE84'] = df['VOTE84'].map(DONT_KNOW_map)
    transformed_df['VOTE88'] = df['VOTE88'].map(DONT_KNOW_map)
    transformed_df['VOTE92'] = df['VOTE92'].map(DONT_KNOW_map)
    transformed_df['VOTE96'] = df['VOTE96'].map(DONT_KNOW_map)
    transformed_df['VOTE00'] = df['VOTE00'].map(DONT_KNOW_map)
    transformed_df['VOTE04'] = df['VOTE04'].map(DONT_KNOW_map)
    transformed_df['VOTE08'] = df['VOTE08'].map(DONT_KNOW_map)
    transformed_df['VOTE12'] = df['VOTE12'].map(DONT_KNOW_map)
    transformed_df['VOTE16'] = df['VOTE16'].map(DONT_KNOW_map)
    transformed_df['VOTE20'] = df['VOTE20'].map(DONT_KNOW_map)

    transformed_df['PRES68_HUMPHREY'] = df['PRES68'].map(category_map_A)
    transformed_df['PRES68_NIXON'] = df['PRES68'].map(category_map_B)
    transformed_df['PRES68_WALLACE'] = df['PRES68'].map(category_map_C)
    transformed_df['PRES68_OTHER'] = df['PRES68'].map(category_map_D)
    transformed_df['PRES68_REFUSED'] = df['PRES68'].map(category_map_E)
    # transformed_df['PRES68_DONT_KNOW)'] = df['PRES68'].map(DONT_KNOW_map)

    transformed_df['PRES72_MCGOVERN'] = df['PRES72'].map(category_map_A)
    transformed_df['PRES72_NIXON'] = df['PRES72'].map(category_map_B)
    transformed_df['PRES72_OTHER'] = df['PRES72'].map(category_map_C)
    transformed_df['PRES72_REFUSED'] = df['PRES72'].map(category_map_D)
    transformed_df['PRES72_WOULDNT_VOTE'] = df['PRES72'].map(category_map_E)
    transformed_df['PRES72_DONT_KNOW'] = df['PRES72'].map(DONT_KNOW_map)

    transformed_df['PRES76_CARTER'] = df['PRES76'].map(category_map_A)
    transformed_df['PRES76_FORD'] = df['PRES76'].map(category_map_B)
    transformed_df['PRES76_OTHER'] = df['PRES76'].map(category_map_C)
    transformed_df['PRES76_REFUSED'] = df['PRES76'].map(category_map_D)
    transformed_df['PRES76_NO_PRES_VOTE'] = df['PRES76'].map(category_map_E)
    transformed_df['PRES76_DONT_KNOW'] = df['PRES76'].map(DONT_KNOW_map)

    transformed_df['PRES80_CARTER'] = df['PRES80'].map(category_map_A)
    transformed_df['PRES80_REAGAN'] = df['PRES80'].map(category_map_B)
    transformed_df['PRES80_ANDERSON'] = df['PRES80'].map(category_map_C)
    transformed_df['PRES80_OTHER'] = df['PRES80'].map(category_map_D)
    transformed_df['PRES80_REFUSED'] = df['PRES80'].map(category_map_E)
    transformed_df['PRES80_DIDNT_VOTE'] = df['PRES80'].map(category_map_F)
    transformed_df['PRES80_DONT_KNOW'] = df['PRES80'].map(DONT_KNOW_map)

    transformed_df['PRES84_MONDALE'] = df['PRES84'].map(category_map_A)
    transformed_df['PRES84_REAGAN'] = df['PRES84'].map(category_map_B)
    transformed_df['PRES84_OTHER'] = df['PRES84'].map(category_map_C)
    transformed_df['PRES84_REFUSED'] = df['PRES84'].map(category_map_D)
    transformed_df['PRES84_NO_PRES_VOTE'] = df['PRES84'].map(category_map_E)
    transformed_df['PRES84_DONT_KNOW'] = df['PRES84'].map(DONT_KNOW_map)

    transformed_df['PRES88_BUSH'] = df['PRES88'].map(category_map_A)
    transformed_df['PRES88_DUKAKIS'] = df['PRES88'].map(category_map_B)
    transformed_df['PRES88_OTHER'] = df['PRES88'].map(category_map_C)
    transformed_df['PRES88_REFUSED'] = df['PRES88'].map(category_map_D)
    transformed_df['PRES88_NO_PRES_VOTE'] = df['PRES88'].map(category_map_E)
    transformed_df['PRES88_DONT_KNOW'] = df['PRES88'].map(DONT_KNOW_map)

    transformed_df['PRES92_CLINTON'] = df['PRES92'].map(category_map_A)
    transformed_df['PRES92_BUSH'] = df['PRES92'].map(category_map_B)
    transformed_df['PRES92_PEROT'] = df['PRES92'].map(category_map_C)
    transformed_df['PRES92_OTHER'] = df['PRES92'].map(category_map_D)
    transformed_df['PRES92_NO_PRES_VOTE'] = df['PRES92'].map(category_map_E)
    transformed_df['PRES92_DONT_KNOW'] = df['PRES92'].map(DONT_KNOW_map)

    transformed_df['PRES96_CLINTON'] = df['PRES96'].map(category_map_A)
    transformed_df['PRES96_DOLE'] = df['PRES96'].map(category_map_B)
    transformed_df['PRES96_PEROT'] = df['PRES96'].map(category_map_C)
    transformed_df['PRES96_OTHER'] = df['PRES96'].map(category_map_D)
    transformed_df['PRES96_DIDNT_VOTE'] = df['PRES96'].map(category_map_E)
    transformed_df['PRES96_DONT_KNOW'] = df['PRES96'].map(DONT_KNOW_map)

    transformed_df['PRES00_GORE'] = df['PRES00'].map(category_map_A)
    transformed_df['PRES00_BUSH'] = df['PRES00'].map(category_map_B)
    transformed_df['PRES00_NADER'] = df['PRES00'].map(category_map_C)
    transformed_df['PRES00_OTHER'] = df['PRES00'].map(category_map_D)
    transformed_df['PRES00_DIDNT_VOTE'] = df['PRES00'].map(category_map_E)
    transformed_df['PRES00_DONT_KNOW'] = df['PRES00'].map(DONT_KNOW_map)

    transformed_df['PRES04_KERRY'] = df['PRES04'].map(category_map_A)
    transformed_df['PRES04_BUSH'] = df['PRES04'].map(category_map_B)
    transformed_df['PRES04_NADER'] = df['PRES04'].map(category_map_C)
    transformed_df['PRES04_NO_PRES_VOTE'] = df['PRES04'].map(category_map_D)
    transformed_df['PRES04_DONT_KNOW'] = df['PRES04'].map(DONT_KNOW_map)

    transformed_df['PRES08_OBAMA'] = df['PRES08'].map(category_map_A)
    transformed_df['PRES08_MCCAIN'] = df['PRES08'].map(category_map_B)
    transformed_df['PRES08_OTHER'] = df['PRES08'].map(category_map_C)
    transformed_df['PRES08_DIDNT_VOTE'] = df['PRES08'].map(category_map_D)
    transformed_df['PRES08_DONT_KNOW'] = df['PRES08'].map(DONT_KNOW_map)

    transformed_df['PRES12_OBAMA'] = df['PRES12'].map(category_map_A)
    transformed_df['PRES12_ROMNEY'] = df['PRES12'].map(category_map_B)
    transformed_df['PRES12_OTHER'] = df['PRES12'].map(category_map_C)
    transformed_df['PRES12_DIDNT_VOTE'] = df['PRES12'].map(category_map_D)
    transformed_df['PRES12_DONT_KNOW'] = df['PRES12'].map(DONT_KNOW_map)

    transformed_df['PRES16_CLINTON'] = df['PRES16'].map(category_map_A)
    transformed_df['PRES16_TRUMP'] = df['PRES16'].map(category_map_B)
    transformed_df['PRES16_OTHER'] = df['PRES16'].map(category_map_C)
    transformed_df['PRES16_DIDNT_VOTE'] = df['PRES16'].map(category_map_D)
    transformed_df['PRES16_DONT_KNOW'] = df['PRES16'].map(DONT_KNOW_map)

    transformed_df['PRES20_BIDEN'] = df['PRES20'].map(category_map_A)
    transformed_df['PRES20_TRUMP'] = df['PRES20'].map(category_map_B)
    transformed_df['PRES20_OTHER'] = df['PRES20'].map(category_map_C)
    transformed_df['PRES20_DIDNT_VOTE'] = df['PRES20'].map(category_map_D)
    transformed_df['PRES20_DONT_KNOW'] = df['PRES20'].map(DONT_KNOW_map)

    transformed_df['IF68WHO_HUMPHREY'] = df['IF68WHO'].map(category_map_A)
    transformed_df['IF68WHO_NIXON'] = df['IF68WHO'].map(category_map_B)
    transformed_df['IF68WHO_WALLACE'] = df['IF68WHO'].map(category_map_C)
    transformed_df['IF68WHO_OTHER'] = df['IF68WHO'].map(category_map_D)
    transformed_df['IF68WHO_WLDNT_VT_RELIG'] = df['IF68WHO'].map(category_map_E)
    transformed_df['IF68WHO_DONT_KNOW'] = df['IF68WHO'].map(DONT_KNOW_map)

    transformed_df['IF72WHO_MCGOVERN'] = df['IF72WHO'].map(category_map_A)
    transformed_df['IF72WHO_NIXON'] = df['IF72WHO'].map(category_map_B)
    transformed_df['IF72WHO_OTHER'] = df['IF72WHO'].map(category_map_C)
    transformed_df['IF72WHO_REFUSED'] = df['IF72WHO'].map(category_map_D)
    transformed_df['IF72WHO_WOULDNT_VOTE'] = df['IF72WHO'].map(category_map_E)
    transformed_df['IF72WHO_WLDNT_VT_RELIG'] = df['IF72WHO'].map(category_map_F)
    transformed_df['IF72WHO_DONT_KNOW'] = df['IF72WHO'].map(DONT_KNOW_map)

    transformed_df['IF76WHO_CARTER'] = df['IF76WHO'].map(category_map_A)
    transformed_df['IF76WHO_FORD'] = df['IF76WHO'].map(category_map_B)
    transformed_df['IF76WHO_OTHER'] = df['IF76WHO'].map(category_map_C)
    transformed_df['IF76WHO_REFUSED'] = df['IF76WHO'].map(category_map_D)
    transformed_df['IF76WHO_WOULDNT_VOTE'] = df['IF76WHO'].map(category_map_E)
    transformed_df['IF76WHO_DONT_KNOW'] = df['IF76WHO'].map(DONT_KNOW_map)

    transformed_df['IF80WHO_CARTER'] = df['IF80WHO'].map(category_map_A)
    transformed_df['IF80WHO_REAGAN'] = df['IF80WHO'].map(category_map_B)
    transformed_df['IF80WHO_ANDERSON'] = df['IF80WHO'].map(category_map_C)
    transformed_df['IF80WHO_OTHER'] = df['IF80WHO'].map(category_map_D)
    transformed_df['IF80WHO_WOULDNT_VOTE'] = df['IF80WHO'].map(category_map_E)
    transformed_df['IF80WHO_REFUSED'] = df['IF80WHO'].map(category_map_F)
    transformed_df['IF80WHO_DONT_KNOW'] = df['IF80WHO'].map(DONT_KNOW_map)

    transformed_df['IF84WHO_MONDALE'] = df['IF84WHO'].map(category_map_A)
    transformed_df['IF84WHO_REAGAN'] = df['IF84WHO'].map(category_map_B)
    transformed_df['IF84WHO_OTHER'] = df['IF84WHO'].map(category_map_C)
    transformed_df['IF84WHO_WOULDNT_VOTE'] = df['IF84WHO'].map(category_map_D)
    transformed_df['IF84WHO_DONT_KNOW'] = df['IF84WHO'].map(DONT_KNOW_map)

    transformed_df['IF88WHO_DUKAKIS'] = df['IF88WHO'].map(category_map_A)
    transformed_df['IF88WHO_BUSH'] = df['IF88WHO'].map(category_map_B)
    transformed_df['IF88WHO_OTHER'] = df['IF88WHO'].map(category_map_C)
    transformed_df['IF88WHO_DONT_KNOW'] = df['IF88WHO'].map(DONT_KNOW_map)

    transformed_df['IF92WHO_CLINTON'] = df['IF92WHO'].map(category_map_A)
    transformed_df['IF92WHO_BUSH'] = df['IF92WHO'].map(category_map_B)
    transformed_df['IF92WHO_PEROT'] = df['IF92WHO'].map(category_map_C)
    transformed_df['IF92WHO_OTHER'] = df['IF92WHO'].map(category_map_D)
    transformed_df['IF92WHO_DONT_KNOW'] = df['IF92WHO'].map(DONT_KNOW_map)

    transformed_df['IF96WHO_CLINTON'] = df['IF96WHO'].map(category_map_A)
    transformed_df['IF96WHO_DOLE'] = df['IF96WHO'].map(category_map_B)
    transformed_df['IF96WHO_PEROT'] = df['IF96WHO'].map(category_map_C)
    transformed_df['IF96WHO_OTHER'] = df['IF96WHO'].map(category_map_D)
    transformed_df['IF96WHO_DONT_KNOW'] = df['IF96WHO'].map(DONT_KNOW_map)

    transformed_df['IF00WHO_GORE'] = df['IF00WHO'].map(category_map_A)
    transformed_df['IF00WHO_BUSH'] = df['IF00WHO'].map(category_map_B)
    transformed_df['IF00WHO_NADER'] = df['IF00WHO'].map(category_map_C)
    transformed_df['IF00WHO_OTHER'] = df['IF00WHO'].map(category_map_D)
    transformed_df['IF00WHO_DONT_KNOW'] = df['IF00WHO'].map(DONT_KNOW_map)

    transformed_df['IF04WHO_KERRY'] = df['IF04WHO'].map(category_map_A)
    transformed_df['IF04WHO_BUSH'] = df['IF04WHO'].map(category_map_B)
    transformed_df['IF04WHO_NADER'] = df['IF04WHO'].map(category_map_C)
    transformed_df['IF04WHO_DONT_KNOW'] = df['IF04WHO'].map(DONT_KNOW_map)

    transformed_df['IF08WHO_OBAMA'] = df['IF08WHO'].map(category_map_A)
    transformed_df['IF08WHO_MCCAIN'] = df['IF08WHO'].map(category_map_B)
    transformed_df['IF08WHO_OTHER'] = df['IF08WHO'].map(category_map_C)
    transformed_df['IF08WHO_DONT_KNOW'] = df['IF08WHO'].map(DONT_KNOW_map)

    transformed_df['IF12WHO_OBAMA'] = df['IF12WHO'].map(category_map_A)
    transformed_df['IF12WHO_ROMNEY'] = df['IF12WHO'].map(category_map_B)
    transformed_df['IF12WHO_OTHER'] = df['IF12WHO'].map(category_map_C)
    transformed_df['IF12WHO_DONT_KNOW'] = df['IF12WHO'].map(DONT_KNOW_map)

    transformed_df['IF16WHO_CLINTON'] = df['IF16WHO'].map(category_map_A)
    transformed_df['IF16WHO_TRUMP'] = df['IF16WHO'].map(category_map_B)
    transformed_df['IF16WHO_OTHER'] = df['IF16WHO'].map(category_map_C)
    transformed_df['IF16WHO_CANT_REMEMBER'] = df['IF16WHO'].map(category_map_D)
    transformed_df['IF16WHO_DONT_KNOW'] = df['IF16WHO'].map(DONT_KNOW_map)

    transformed_df['IF20WHO_BIDEN'] = df['IF20WHO'].map(category_map_A)
    transformed_df['IF20WHO_TRUMP'] = df['IF20WHO'].map(category_map_B)
    transformed_df['IF20WHO_OTHER'] = df['IF20WHO'].map(category_map_C)
    transformed_df['IF20WHO_CANT_REMEMBER'] = df['IF20WHO'].map(category_map_D)
    transformed_df['IF20WHO_DONT_KNOW'] = df['IF20WHO'].map(DONT_KNOW_map)

    transformed_df['POLVIEWS'] = df['POLVIEWS'].map(POLVIEWS_map)

    transformed_df['NATSPAC'] = df['NATSPAC'].map(NAT_map)
    transformed_df['NATENVIR'] = df['NATENVIR'].map(NAT_map)
    transformed_df['NATHEAL'] = df['NATHEAL'].map(NAT_map)
    transformed_df['NATCITY'] = df['NATCITY'].map(NAT_map)
    transformed_df['NATCRIME'] = df['NATCRIME'].map(NAT_map)
    transformed_df['NATDRUG'] = df['NATDRUG'].map(NAT_map)
    transformed_df['NATEDUC'] = df['NATEDUC'].map(NAT_map)
    transformed_df['NATRACE'] = df['NATRACE'].map(NAT_map)
    transformed_df['NATARMS'] = df['NATARMS'].map(NAT_map)
    transformed_df['NATAID'] = df['NATAID'].map(NAT_map)
    transformed_df['NATFARE'] = df['NATFARE'].map(NAT_map)
    transformed_df['NATROAD'] = df['NATROAD'].map(NAT_map)
    transformed_df['NATSOC'] = df['NATSOC'].map(NAT_map)
    transformed_df['NATMASS'] = df['NATMASS'].map(NAT_map)
    transformed_df['NATPARK'] = df['NATPARK'].map(NAT_map)
    transformed_df['NATCHLD'] = df['NATCHLD'].map(NAT_map)
    transformed_df['NATSCI'] = df['NATSCI'].map(NAT_map)
    transformed_df['NATENRGY'] = df['NATENRGY'].map(NAT_map)

    transformed_df['NATSPACY'] = df['NATSPACY'].map(NAT_map)
    transformed_df['NATENVIY'] = df['NATENVIY'].map(NAT_map)
    transformed_df['NATHEALY'] = df['NATHEALY'].map(NAT_map)
    transformed_df['NATCITYY'] = df['NATCITYY'].map(NAT_map)
    transformed_df['NATCRIMY'] = df['NATCRIMY'].map(NAT_map)
    transformed_df['NATDRUGY'] = df['NATDRUGY'].map(NAT_map)
    transformed_df['NATEDUCY'] = df['NATEDUCY'].map(NAT_map)
    transformed_df['NATRACEY'] = df['NATRACEY'].map(NAT_map)
    transformed_df['NATARMSY'] = df['NATARMSY'].map(NAT_map)
    transformed_df['NATAIDY'] = df['NATAIDY'].map(NAT_map)
    transformed_df['NATFAREY'] = df['NATFAREY'].map(NAT_map)

    transformed_df['EQWLTH'] = df['EQWLTH'].map(EQWLTH_map)

    transformed_df['SPKATH'] = df['SPKATH'].map(SPK_map)
    transformed_df['SPKRAC'] = df['SPKRAC'].map(SPK_map)
    transformed_df['SPKCOM'] = df['SPKCOM'].map(SPK_map)
    transformed_df['SPKMIL'] = df['SPKMIL'].map(SPK_map)
    transformed_df['SPKHOMO'] = df['SPKHOMO'].map(SPK_map)
    transformed_df['SPKMSLM'] = df['SPKMSLM'].map(SPK_map)

    transformed_df['COLATH'] = df['COLATH'].map(COLATH_map)
    transformed_df['COLRAC'] = df['COLRAC'].map(COLATH_map)
    transformed_df['COLCOM'] = df['COLCOM'].map(COLATH_map)
    transformed_df['COLMIL'] = df['COLMIL'].map(COLATH_map)
    transformed_df['COLHOMO'] = df['COLHOMO'].map(COLATH_map)
    transformed_df['COLMSLM'] = df['COLMSLM'].map(COLATH_map)

    transformed_df['LIBATH'] = df['LIBATH'].map(LIB_map)
    transformed_df['LIBRAC'] = df['LIBRAC'].map(LIB_map)
    transformed_df['LIBCOM'] = df['LIBCOM'].map(LIB_map)
    transformed_df['LIBMIL'] = df['LIBMIL'].map(LIB_map)
    transformed_df['LIBHOMO'] = df['LIBHOMO'].map(LIB_map)
    transformed_df['LIBMSLM'] = df['LIBMSLM'].map(LIB_map)

    transformed_df['CAPPUN'] = df['CAPPUN'].map(POL1_map)
    transformed_df['GUNLAW'] = df['GUNLAW'].map(POL1_map)

    transformed_df['COURTS'] = df['COURTS'].map(COURTS_map)

    transformed_df['GRASS'] = df['GRASS'].map(GRASS_map)

    transformed_df['RELIG_PROT'] = df['RELIG'].map(category_map_A)
    transformed_df['RELIG_CATHOLIC'] = df['RELIG'].map(category_map_B)
    transformed_df['RELIG_JEWISH'] = df['RELIG'].map(category_map_C)
    transformed_df['RELIG_NONE'] = df['RELIG'].map(category_map_D)
    transformed_df['RELIG_OTHER'] = df['RELIG'].map(category_map_E)
    transformed_df['RELIG_BUDDHISM'] = df['RELIG'].map(category_map_F)
    transformed_df['RELIG_HINDUISM'] = df['RELIG'].map(category_map_G)
    transformed_df['RELIG_OTHER_EASTERN'] = df['RELIG'].map(category_map_H)  # Adjusted for clarity
    transformed_df['RELIG_MUSLIM_ISLAM'] = df['RELIG'].map(category_map_I)
    transformed_df['RELIG_ORTHODOX_CHRISTIAN'] = df['RELIG'].map(category_map_J)
    transformed_df['RELIG_CHRISTIAN'] = df['RELIG'].map(category_map_K)
    transformed_df['RELIG_NATIVE_AMERICAN'] = df['RELIG'].map(category_map_L)
    transformed_df['RELIG_INTER_NONDENOMINATIONAL'] = df['RELIG'].map(category_map_M)

    transformed_df['ATTEND'] = df['ATTEND'].map(ATTEND_map)

    transformed_df['RELITEN'] = df['RELITEN'].map(RELITEN_map)

    transformed_df['POSTLIFE'] = df['POSTLIFE'].map(POSTLIFE_map)

    transformed_df['PRAYER'] = df['PRAYER'].map(PRAYER_map)

    transformed_df['RACOPEN'] = df['RACOPEN'].map(RACOPEN_map)

    transformed_df['AFFRMACT'] = df['AFFRMACT'].map(AFFRMACT_map)

    transformed_df['WRKWAYUP'] = df['WRKWAYUP'].map(WRKWAYUP_map)

    transformed_df['HELPFUL'] = df['HELPFUL'].map(HELPFUL_map)

    transformed_df['FAIR'] = df['FAIR'].map(FAIR_map)

    transformed_df['TRUST'] = df['TRUST'].map(TRUST_map)

    transformed_df['CONFINAN'] = df['CONFINAN'].map(CON_map)
    transformed_df['CONBUS'] = df['CONBUS'].map(CON_map)
    transformed_df['CONCLERG'] = df['CONCLERG'].map(CON_map)
    transformed_df['CONEDUC'] = df['CONEDUC'].map(CON_map)
    transformed_df['CONFED'] = df['CONFED'].map(CON_map)
    transformed_df['CONLABOR'] = df['CONLABOR'].map(CON_map)
    transformed_df['CONPRESS'] = df['CONPRESS'].map(CON_map)
    transformed_df['CONMEDIC'] = df['CONMEDIC'].map(CON_map)
    transformed_df['CONTV'] = df['CONTV'].map(CON_map)
    transformed_df['CONJUDGE'] = df['CONJUDGE'].map(CON_map)
    transformed_df['CONSCI'] = df['CONSCI'].map(CON_map)
    transformed_df['CONLEGIS'] = df['CONLEGIS'].map(CON_map)
    transformed_df['CONARMY'] = df['CONARMY'].map(CON_map)

    transformed_df['OBEY'] = df['OBEY'].map(KID_map)
    transformed_df['POPULAR'] = df['POPULAR'].map(KID_map)
    transformed_df['THNKSELF'] = df['THNKSELF'].map(KID_map)
    transformed_df['WORKHARD'] = df['WORKHARD'].map(KID_map)
    transformed_df['HELPOTH'] = df['HELPOTH'].map(KID_map)

    transformed_df['GETAHEAD'] = df['GETAHEAD'].map(GETAHEAD_map)

    transformed_df['FEPOL'] = df['FEPOL'].map(FEPOL_map)

    transformed_df['ABDEFECT'] = df['ABDEFECT'].map(AB_map)
    transformed_df['ABNOMORE'] = df['ABNOMORE'].map(AB_map)
    transformed_df['ABHLTH'] = df['ABHLTH'].map(AB_map)
    transformed_df['ABPOOR'] = df['ABPOOR'].map(AB_map)
    transformed_df['ABRAPE'] = df['ABRAPE'].map(AB_map)
    transformed_df['ABSINGLE'] = df['ABSINGLE'].map(AB_map)
    transformed_df['ABANY'] = df['ABANY'].map(AB_map)


    transformed_df['SEXEDUC'] = df['SEXEDUC'].map(SEXEDUC_map)

    transformed_df['DIVLAW'] = df['DIVLAW'].map(DIVLAW_map)

    transformed_df['PREMARSX'] = df['PREMARSX'].map(SEX_map)
    transformed_df['TEENSEX'] = df['TEENSEX'].map(SEX_map)
    transformed_df['XMARSEX'] = df['XMARSEX'].map(SEX_map)
    transformed_df['HOMOSEX'] = df['HOMOSEX'].map(SEX_map)

    transformed_df['PORNLAW'] = df['PORNLAW'].map(PORNLAW_map)

    transformed_df['SPANKING'] = df['SPANKING'].map(SPANKING_map)

    transformed_df['LETDIE1'] = df['LETDIE1'].map(DEATH_map)
    transformed_df['SUICIDE1'] = df['SUICIDE1'].map(DEATH_map)
    transformed_df['SUICIDE2'] = df['SUICIDE2'].map(DEATH_map)

    transformed_df['POLHITOK'] = df['POLHITOK'].map(POLICE_map)
    transformed_df['POLABUSE'] = df['POLABUSE'].map(POLICE_map)
    transformed_df['POLMURDR'] = df['POLMURDR'].map(POLICE_map)
    transformed_df['POLESCAP'] = df['POLESCAP'].map(POLICE_map)
    transformed_df['POLATTAK'] = df['POLATTAK'].map(POLICE_map)

    transformed_df['NEWS'] = df['NEWS'].map(NEWS_map)

    transformed_df['TVHOURS'] = df['TVHOURS'].map(TVHOURS_map)

    transformed_df['FECHLD'] = df['FECHLD'].map(FE_map)
    transformed_df['FEPRESCH'] = df['FEPRESCH'].map(FE_map)
    transformed_df['FEFAM'] = df['FEFAM'].map(FE_map)

    transformed_df['RACDIF1'] = df['RACDIF1'].map(RACDIF_map)
    transformed_df['RACDIF2'] = df['RACDIF2'].map(RACDIF_map)
    transformed_df['RACDIF3'] = df['RACDIF3'].map(RACDIF_map)
    transformed_df['RACDIF4'] = df['RACDIF4'].map(RACDIF_map)

    transformed_df['HELPPOOR'] = df['HELPPOOR'].map(HELP_map)
    transformed_df['HELPNOT'] = df['HELPNOT'].map(HELP_map)
    transformed_df['HELPBLK'] = df['HELPBLK'].map(HELP_map)

    transformed_df['MARHOMO'] = df['MARHOMO'].map(MARHOMO_map)
    # endregion
    
    if combine_variants:
        for variant, original in variants.items():
            combined_column = transformed_df[original].combine_first(transformed_df[variant])
            transformed_df[original] = combined_column
    
    transformed_df = transformed_df.drop(variants.keys(), axis=1)

    ### Uncomment this to print select columns to a text file.
    # with open('partyid.txt', 'w') as f:
    #     for index, row in df.iterrows():
    #         f.write(str(row['MARHOMO']) + ' ' + str(row['MARHOMO (mapped)']) + ' ' + str(row['PRES68 (dont know)']) + '\n')

    return transformed_df