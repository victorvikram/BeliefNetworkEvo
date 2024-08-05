import pandas as pd
import pyreadstat as prs
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np



def make_variable_summary(df):
    """
    this function goes year by year and for each ballot puts in the percentage of non nan answers to the question
    """
    counts = df.groupby(['YEAR', 'BALLOT'], dropna=False).count()

    notnull_df = df.notna()
    notnull_df['YEAR'] = df['YEAR']
    notnull_df['BALLOT'] = df['BALLOT']

    pcts = notnull_df.groupby(['YEAR', 'BALLOT'], dropna=False).mean()


    partially_complete_ballot_mask = (0 < pcts) & (pcts < 0.8)
    pc_row, pc_col = np.where(partially_complete_ballot_mask)
    partially_complete_ballot = [(partially_complete_ballot_mask.index[row_ind], partially_complete_ballot_mask.columns[col_ind]) for (row_ind, col_ind) in zip(pc_row, pc_col)]


    return counts, pcts, partially_complete_ballot

def make_vote_supernodes(df, meta_df, varnames=["VOTE{year}", "PRES{year}_NONCONFORM", "PRES{year}_DEMREP"]):
    
    year_order = ["68", "72", "76", "80", "84", "88", "92", "96", "00", "04", "08", "12", "16", "20"]
    new_df = df.copy()
    new_meta_df = meta_df.copy()

    for var in varnames:
        cols = [var.format(year=year) for year in year_order]
        cols_present = [col for col in cols if col in df.columns]
        sub_df = df.loc[:, cols_present]

        new_col = var.format(year="LAST")
        sub_df[new_col] = np.nan
        sub_df = sub_df.ffill(axis=1)

        new_df[new_col] = sub_df[new_col]

        min_val = np.nanmin(new_df[new_col])
        max_val = np.nanmax(new_df[new_col])
        new_meta_df.loc[new_col, :] = {"min": min_val, "max": max_val}


    return new_df, new_meta_df

def normalize_columns(df, meta_df, exclude=["YEAR", "BALLOT", "ID"]):
    """
    transforms the df so that the range of each column is 1. signed variables go from [-0.5, 0.5], others go from [0, 1]
    """

    cols_to_use = [col for col in df.columns if col not in exclude]

    range_of_values = meta_df["max"] - meta_df["min"]
    normalized_df = df.loc[:,cols_to_use].div(range_of_values[cols_to_use])
    
    for col in exclude:
        if col in df.columns:
            normalized_df[col] = df[col]

    return normalized_df



def check_for_duplicate_column_names(df):
    """
    this function takes in a dataframe and returns a list of column names that are duplicated
    """
    return df.columns[df.columns.duplicated()].tolist()


# The purpose of this script is to read in the GSS dataset and perform some basic data cleaning and transformation.
# The dataset is a SAS7BDAT file, so we will use the pandas library to read in the data.
# The basic idea here is to manually write the mappings for the variables in the dataset and then apply them to the dataset.

def transform_dataframe_1(df, combine_variants=True):

    """

    note: if value is not in map it goes to NaN
    """
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
        "NATENVIY": "NATENVIR",
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

    #####################################################################################################################################################################
    #####################################################################################################################################################################

    # Naturally binary variables
    """Here we catch all the variables that can be naturally mapped to "anti-belief" (-1) and "belief" (1)"""

    """They are: = ["VOTE68, PRES68, IF68WHO, VOTE72, PRES72, IF72WHO, VOTE76, PRES76, IF76WHO,	VOTE80,	PRES80,	IF80WHO, VOTE84, PRES84, IF84WHO,	
    VOTE88,	PRES88,	IF88WHO, VOTE92, PRES92, IF92WHO, VOTE96, PRES96, IF96WHO, VOTE00, PRES00, IF00WHO, VOTE04,	PRES04,	IF04WHO, VOTE08, PRES08, IF08WHO,	
    VOTE12,	PRES12,	IF12WHO, VOTE16, PRES16, IF16WHO, VOTE20, PRES20, IF20WHO, 
    SPKATH,	COLATH,	LIBATH,	SPKRAC,	COLRAC,	LIBRAC,	SPKCOM,	COLCOM,	LIBCOM,	SPKMIL,	COLMIL,	LIBMIL,	SPKHOMO, COLHOMO, LIBHOMO, SPKMSLM,	COLMSLM, LIBMSLM,
    CAPPUN,	GUNLAW,	GRASS, RELIG, POSTLIFE, PRAYER,	FEPOL, ABDEFECT, ABNOMORE, ABHLTH, ABPOOR, ABRAPE, ABSINGLE, ABANY,	SEXEDUC, LETDIE1, SUICIDE1,
    SUICIDE2, POLHITOK, POLABUSE, POLMURDR,	POLESCAP, POLATTAK,	RACDIF1, RACDIF2, RACDIF3, RACDIF4]
    """
    # VOTE68, VOTE72, VOTE76, VOTE80, VOTE84, VOTE88, VOTE92, VOTE96, VOTE00, VOTE04, VOTE08, VOTE12, VOTE16, VOTE20
    VOTE_map = {1: 1, 2: -1}
    ELIGIBLE_map = {1: 1, 2: 1, 3: -1}
    DONT_KNOW_map = {-98: 1, 1: -1, 2: -1, 3: -1, 4: -1, 5: -1, 6: -1, 7: -1}

    # PRES68, PRES72, PRES76, PRES80, PRES84, PRES88, PRES92, PRES96, PRES00, PRES04, PRES08, PRES12, PRES16...
    category_map_A = {1: 1, 2: -1, 3: -1, 4: -1, 5: -1, 6: -1} # For mapping to the first option
    category_map_B = {1: -1, 2: 1, 3: -1, 4: -1, 5: -1, 6: -1} # For mapping to the second option...
    category_map_C = {1: -1, 2: -1, 3: 1, 4: -1, 5: -1, 6: -1}
    category_map_D = {1: -1, 2: -1, 3: -1, 4: 1, 5: -1, 6: -1}
    category_map_E = {1: -1, 2: -1, 3: -1, 4: -1, 5: 1, 6: -1}
    category_map_F = {1: -1, 2: -1, 3: -1, 4: -1, 5: -1, 6: 1}

    # Speaking variables: SPKATH, SPKRAC, SPKCOM, SPKMIL, SPKHOMO, SPKMSLM
    SPK_map = {1: 1, 2: -1}

    # Teaching variables: COLATH, COLRAC, COLCOM, COLMIL, COLHOMO, COLMSLM
    COLATH_map = {4: 1, 5: -1}

    # Library book variables: LIBATH, LIBRAC, LIBCOM, LIBMIL, LIBHOMO, LIBMSLM
    LIB_map = {1: 1, 2: -1}

    # CAPPUN and GUNLAW
    POL1_map = {1: 1, 2: -1}

    # GRASS: Should the use of marijuana should be made legal or not?
    GRASS_map = {1: 1, 2: -1}

    # RELIG: What is your religious preference?
    category_map_A = {1: 1, 2: -1, 3: -1, 4: -1, 5: -1, 6: -1, 7: -1, 8: -1, 9: -1, 1-1: -1, 11: -1, 12: -1, 13: -1}
    category_map_B = {1: -1, 2: 1, 3: -1, 4: -1, 5: -1, 6: -1, 7: -1, 8: -1, 9: -1, 1-1: -1, 11: -1, 12: -1, 13: -1}
    category_map_C = {1: -1, 2: -1, 3: 1, 4: -1, 5: -1, 6: -1, 7: -1, 8: -1, 9: -1, 1-1: -1, 11: -1, 12: -1, 13: -1}
    category_map_D = {1: -1, 2: -1, 3: -1, 4: 1, 5: -1, 6: -1, 7: -1, 8: -1, 9: -1, 1-1: -1, 11: -1, 12: -1, 13: -1}
    category_map_E = {1: -1, 2: -1, 3: -1, 4: -1, 5: 1, 6: -1, 7: -1, 8: -1, 9: -1, 1-1: -1, 11: -1, 12: -1, 13: -1}
    category_map_F = {1: -1, 2: -1, 3: -1, 4: -1, 5: -1, 6: 1, 7: -1, 8: -1, 9: -1, 1-1: -1, 11: -1, 12: -1, 13: -1}
    category_map_G = {1: -1, 2: -1, 3: -1, 4: -1, 5: -1, 6: -1, 7: 1, 8: -1, 9: -1, 1-1: -1, 11: -1, 12: -1, 13: -1}
    category_map_H = {1: -1, 2: -1, 3: -1, 4: -1, 5: -1, 6: -1, 7: -1, 8: 1, 9: -1, 1-1: -1, 11: -1, 12: -1, 13: -1}
    category_map_I = {1: -1, 2: -1, 3: -1, 4: -1, 5: -1, 6: -1, 7: -1, 8: -1, 9: 1, 1-1: -1, 11: -1, 12: -1, 13: -1}
    category_map_J = {1: -1, 2: -1, 3: -1, 4: -1, 5: -1, 6: -1, 7: -1, 8: -1, 9: -1, 1-1: 1, 11: -1, 12: -1, 13: -1}
    category_map_K = {1: -1, 2: -1, 3: -1, 4: -1, 5: -1, 6: -1, 7: -1, 8: -1, 9: -1, 1-1: -1, 11: 1, 12: -1, 13: -1}
    category_map_L = {1: -1, 2: -1, 3: -1, 4: -1, 5: -1, 6: -1, 7: -1, 8: -1, 9: -1, 1-1: -1, 11: -1, 12: 1, 13: -1}
    category_map_M = {1: -1, 2: -1, 3: -1, 4: -1, 5: -1, 6: -1, 7: -1, 8: -1, 9: -1, 1-1: -1, 11: -1, 12: -1, 13: 1}


    # POSTLIFE: Do you believe in life after death?
    POSTLIFE_map = {1: 1, 2: -1}

    # PRAYER: Do you approve or disapprove that mandatory prayer is banned in public schools?
    PRAYER_map = {1: 1, 2: -1}

    # FEPOL: Most men are better suited emotionally for politics than are most women.
    FEPOL_map = {1: 1, 2: -1}

    # Abortion questions: ABDEFECT, ABNOMORE, ABHLTH, ABPOOR, ABRAPE, ABSINGLE, ABANY
    AB_map = {1: 1, 2: -1}

    # SEXEDUC: Sex education in the public schools?
    SEXEDUC_map = {1: 1, 2: -1}

    # Death: LETDIE1, SUICIDE1, SUICIDE2
    DEATH_map = {1: 1, 2: -1}

    # Police: POLHITOK, POLABUSE, POLMURDR, POLESCAP, POLATTAK
    POLICE_map = {1: 1, 2: -1}

    # Racial differences: RACDIF1, RACDIF2, RACDIF3, RACDIF4
    RACDIF_map = {1: 1, 2: -1}

    #####################################################################################################################################################################
    #####################################################################################################################################################################

    # Naturally signed variables
    """Here we catch all the variables that can be naturally mapped to "anti-belief" (-1) --- "lack-of-belief / middle-ground-stance" (0) and "belief" (1)"""

    """They are: = [PARTYID, POLVIEWS, NATSPAC, NATENVIR, NATHEAL, NATCITY, NATCRIME, NATDRUG, NATEDUC, NATRACE, NATAID, NATFARE, NATROAD, NATSOC, NATMASS, NATPARK, NATCHLD, NATSCI, NATSPACY, NATENVIY,
        NATHEALY, NATCITYY, NATCRIMY,NATDRUGY, NATEDUCY, NATRACEY, NATARMSY, NATAIDY, NATFAREY, COURTS,	RACOPEN, AFFRMACT, WRKWAYUP, HELPFUL, FAIR,	TRUST, GETAHEAD, DIVLAW, SPANKING,
        FECHLD,	FEPRESCH, FEFAM, HELPPOOR, HELPNOT, HELPBLK, MARHOMO]
    """

    # PARTYID: Generally speaking, do you usually think of yourself as a Republican, Democrat, Independent, or what?
    PARTYID_map = {0: -3, 1: -2, 2: -1, 3: 0, 4: 1, 5: 2, 6: 3}
    other_map = {0: -1, 1: -1, 2: -1, 3: -1, 4: -1, 5: -1, 6: -1, 7: 1}

    # POLVIEWS: Does r think of self as liberal or conservative?
    POLVIEWS_map = {1: -3, 2: -2, 3: -1, 4: 0, 5: 1, 6: 2, 7: 3} 

    # NATSPAC, NATENVIR, NATHEAL, NATCITY, NATCRIME, NATDRUG, NATEDUC, NATRACE, NATARMS, NATAID, NATFARE, NATROAD, NATSOC, NATMASS, NATPARK, NATCHLD, NATSCI, NATENRGY, NATSPACY, NATENVIY, NATHEALY, NATCITYY, NATCRIMY, NATDRUGY, NATEDUCY, NATRACEY, NATARMSY, NATAIDY, NATFAREY
    NAT_map = {1: -1, 2: 0, 3: 1}

    # COURTS: Courts deal too harshly or not harshly enough with criminals?
    COURTS_map = {1: -1, 2: 0, 3: 1}

    # RACOPEN
    RACOPEN_map = {1: -1, 2: 1, 3: 0, -98: 0}

    # AFFRMACT: Are you for or against preferential hiring and promotion of blacks? 
    AFFRMACT_map = {1: 2, 2: 1, 3: -1, 4: -2}

    # WRKWAYUP: Irish, Italians, Jewish and many other minorities overcame prejudice and worked their way up. 
    WRKWAYUP_map = {1: 2, 2: 1, 3: 0, 4: -1, 5: -2}

    # HELPFUL: Do you think people try to be helpful, or that they are mostly just looking out for themselves?
    HELPFUL_map = {1: 1, 2: -1, 3: 0}

    # FAIR: Do you think people would try to take advantage of you if they got a chance, or would they try to be fair?
    FAIR_map = {2: 1, 1: -1, 3: 0}

    # TRUST: Can or can't most people be trusted?
    TRUST_map = {1: 1, 2: -1, 3: 0}

    # GETAHEAD: Does hardwork or luck get you ahead in life?
    GETAHEAD_map = {1: -1, 2: 0, 3: 1}
    other_GETAHEAD_map = {1: -1, 2: -1, 3: -1, 4: 1}

    # DIVLAW: Should divorce in this country be easier or more difficult to obtain than it is now?
    DIVLAW_map = {1: 1, 2: 0, 3: -1}

    # SPANKING: It is sometimes necessary to discipline a child with a good, hard spanking.
    SPANKING_map = {1: 2, 2: 1, 3: -1, 4: -2}

    # female: FECHLD, FEPRESCH, and FEFAM
    FE_map = {1: 2, 2: 1, 3: -1, 4: -2}

    # Help people: HELPPOOR, HELPNOT, HELPBLK
    HELP_map = {1: 2, 2: 1, 3: 0, 4: -1, 5: -2}

    # MARHOMO: Homosexual couples should have the right to marry one another.
    MARHOMO_map = {1: 2, 2: 1, 3: 0, 4: -1, 5: -2}


    #####################################################################################################################################################################
    #####################################################################################################################################################################

    # Nuanced cases: median approach for setting up ordinality
    """See clean_data_2.py for how we treat these variables. The following mappings are sufficient for computing correlations, but not for stress calculations."""
    
    # EQWLTH: Should govt reduce income differences?
    # Set the mapping around the median value
    EQWLTH_map = {1: 3, 2: 2, 3: 1, 4: 0, 5: -1, 6: -2, 7: -3}
    #DONT_KNOW_map works here

    # ATTEND: How often do you attend religious services? 
    ATTEND_map = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8}
    
    # RELITEN: Would you call yourself a strongly religious or a not very strong?
    RELITEN_map = {1: 3, 2: 2, 3: 1, 4: 0}
    NO_ANSWER_map = {-99: 1, 1: -1, 2: -1, 3: -1, 4: -1, 5: -1, 6: -1, 7: -1}

    # Confidence variables: CONFINAN, CONBUS, CONCLERG, CONEDUC, CONFED, CONLABOR, CONPRESS, CONMEDIC, CONTV, CONJUDGE, CONSCI, CONLEGIS, CONARMY
    CON_map = {1: 2, 2: 1, 3: 0}
    
    # Kid learning ranking: OBEY, POPULAR, THNKSELF, WORKHARD, HELPOTH
    KID_map = {1: 4, 2: 3, 3: 2, 4: 1, 5: 0}
    
    # SEX: PREMARSX, TEENSEX, XMARSEX, HOMOSEX
    SEX_map = {1: 3, 2: 2, 3: 1, 4: 0}
    
    # PORNLAW: Should laws forbid porn?
    PORNLAW_map = {1: 2, 2: 1, 3: 0}
    
    # NEWS: How often do you read the newspaper?
    NEWS_map = {1: 4, 2: 3, 3: 2, 4: 1, 5: 0}
    
    # TVHOURS: On average, how many hours per day on TV?
    TVHOURS_map = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10, 11:11, 12:12, 13:13, 14:14, 15:15, 16:16, 17:17, 18:18, 19:19, 20:20, 21:21, 22:22, 23:23, 24:24}
    
    #####################################################################################################################################################################
    #####################################################################################################################################################################

    # Derived beliefs -- stuff that we've made up, constructed from the data.

    DEMREP_map = {1: -1, 2: 1}
    NONCONFORM_map = {1: -1, 2: -1, 3: 1} # in years where there is no significant third party, nonconforming is just other
    NONCONFORM3P_map = {1: -1, 2: -1, 3: 1, 4: 1} # in years whre there is a significant third party, nonconforming is the third party or other


    transformations_to_do = [
        # PARTYID
        {"source": "PARTYID", "dest": "PARTYID", "map": PARTYID_map},
        {"source": "PARTYID", "dest": "OTHER_PARTY", "map": other_map},

        # VOTE__
        {"source": "VOTE68", "dest": "VOTE68", "map": VOTE_map},
        {"source": "VOTE72", "dest": "VOTE72", "map": VOTE_map},
        {"source": "VOTE76", "dest": "VOTE76", "map": VOTE_map},
        {"source": "VOTE80", "dest": "VOTE80", "map": VOTE_map},
        {"source": "VOTE84", "dest": "VOTE84", "map": VOTE_map},
        {"source": "VOTE88", "dest": "VOTE88", "map": VOTE_map},
        {"source": "VOTE92", "dest": "VOTE92", "map": VOTE_map},
        {"source": "VOTE96", "dest": "VOTE96", "map": VOTE_map},
        {"source": "VOTE00", "dest": "VOTE00", "map": VOTE_map},
        {"source": "VOTE04", "dest": "VOTE04", "map": VOTE_map},
        {"source": "VOTE08", "dest": "VOTE08", "map": VOTE_map},
        {"source": "VOTE12", "dest": "VOTE12", "map": VOTE_map},
        {"source": "VOTE16", "dest": "VOTE16", "map": VOTE_map},
        {"source": "VOTE20", "dest": "VOTE20", "map": VOTE_map},

        {"source": "VOTE68", "dest": "VOTE68_ELIGIBLE", "map": ELIGIBLE_map},
        {"source": "VOTE72", "dest": "VOTE72_ELIGIBLE", "map": ELIGIBLE_map},
        {"source": "VOTE76", "dest": "VOTE76_ELIGIBLE", "map": ELIGIBLE_map},
        {"source": "VOTE80", "dest": "VOTE80_ELIGIBLE", "map": ELIGIBLE_map},
        {"source": "VOTE84", "dest": "VOTE84_ELIGIBLE", "map": ELIGIBLE_map},
        {"source": "VOTE88", "dest": "VOTE88_ELIGIBLE", "map": ELIGIBLE_map},
        {"source": "VOTE92", "dest": "VOTE92_ELIGIBLE", "map": ELIGIBLE_map},
        {"source": "VOTE96", "dest": "VOTE96_ELIGIBLE", "map": ELIGIBLE_map},
        {"source": "VOTE00", "dest": "VOTE00_ELIGIBLE", "map": ELIGIBLE_map},
        {"source": "VOTE04", "dest": "VOTE04_ELIGIBLE", "map": ELIGIBLE_map},
        {"source": "VOTE08", "dest": "VOTE08_ELIGIBLE", "map": ELIGIBLE_map},
        {"source": "VOTE12", "dest": "VOTE12_ELIGIBLE", "map": ELIGIBLE_map},
        {"source": "VOTE16", "dest": "VOTE16_ELIGIBLE", "map": ELIGIBLE_map},
        {"source": "VOTE20", "dest": "VOTE20_ELIGIBLE", "map": ELIGIBLE_map},

        {"source": "VOTE68", "dest": "VOTE68_DONT_KNOW", "map": DONT_KNOW_map},
        {"source": "VOTE72", "dest": "VOTE72_DONT_KNOW", "map": DONT_KNOW_map},
        {"source": "VOTE76", "dest": "VOTE76_DONT_KNOW", "map": DONT_KNOW_map},
        {"source": "VOTE80", "dest": "VOTE80_DONT_KNOW", "map": DONT_KNOW_map},
        {"source": "VOTE84", "dest": "VOTE84_DONT_KNOW", "map": DONT_KNOW_map},
        {"source": "VOTE88", "dest": "VOTE88_DONT_KNOW", "map": DONT_KNOW_map},
        {"source": "VOTE92", "dest": "VOTE92_DONT_KNOW", "map": DONT_KNOW_map},
        {"source": "VOTE96", "dest": "VOTE96_DONT_KNOW", "map": DONT_KNOW_map},
        {"source": "VOTE00", "dest": "VOTE00_DONT_KNOW", "map": DONT_KNOW_map},
        {"source": "VOTE04", "dest": "VOTE04_DONT_KNOW", "map": DONT_KNOW_map},
        {"source": "VOTE08", "dest": "VOTE08_DONT_KNOW", "map": DONT_KNOW_map},
        {"source": "VOTE12", "dest": "VOTE12_DONT_KNOW", "map": DONT_KNOW_map},
        {"source": "VOTE16", "dest": "VOTE16_DONT_KNOW", "map": DONT_KNOW_map},
        {"source": "VOTE20", "dest": "VOTE20_DONT_KNOW", "map": DONT_KNOW_map},

        {"source": "PRES68", "dest": "PRES68_HUMPHREY", "map": category_map_A},
        {"source": "PRES68", "dest": "PRES68_NIXON", "map": category_map_B},
        {"source": "PRES68", "dest": "PRES68_WALLACE", "map": category_map_C},
        {"source": "PRES68", "dest": "PRES68_OTHER", "map": category_map_D},
        {"source": "PRES68", "dest": "PRES68_REFUSED", "map": category_map_E},
        {"source": "PRES68", "dest": "PRES68_DEMREP", "map": DEMREP_map},
        {"source": "PRES68", "dest": "PRES68_NONCONFORM", "map": NONCONFORM3P_map},

        {"source": "PRES72", "dest": "PRES72_MCGOVERN", "map": category_map_A},
        {"source": "PRES72", "dest": "PRES72_NIXON", "map": category_map_B},
        {"source": "PRES72", "dest": "PRES72_OTHER", "map": category_map_C},
        {"source": "PRES72", "dest": "PRES72_REFUSED", "map": category_map_D},
        {"source": "PRES72", "dest": "PRES72_WOULDNT_VOTE", "map": category_map_E},
        {"source": "PRES72", "dest": "PRES72_DONT_KNOW", "map": DONT_KNOW_map},
        {"source": "PRES72", "dest": "PRES72_DEMREP", "map": DEMREP_map},
        {"source": "PRES72", "dest": "PRES72_NONCONFORM", "map": NONCONFORM_map},

        # PRES76 transformations
        {"source": "PRES76", "dest": "PRES76_CARTER", "map": category_map_A},
        {"source": "PRES76", "dest": "PRES76_FORD", "map": category_map_B},
        {"source": "PRES76", "dest": "PRES76_OTHER", "map": category_map_C},
        {"source": "PRES76", "dest": "PRES76_REFUSED", "map": category_map_D},
        {"source": "PRES76", "dest": "PRES76_NO_PRES_VOTE", "map": category_map_E},
        {"source": "PRES76", "dest": "PRES76_DONT_KNOW", "map": DONT_KNOW_map},
        {"source": "PRES76", "dest": "PRES76_DEMREP", "map": DEMREP_map},
        {"source": "PRES76", "dest": "PRES76_NONCONFORM", "map": NONCONFORM_map},

        # PRES80 transformations
        {"source": "PRES80", "dest": "PRES80_CARTER", "map": category_map_A},
        {"source": "PRES80", "dest": "PRES80_REAGAN", "map": category_map_B},
        {"source": "PRES80", "dest": "PRES80_ANDERSON", "map": category_map_C},
        {"source": "PRES80", "dest": "PRES80_OTHER", "map": category_map_D},
        {"source": "PRES80", "dest": "PRES80_REFUSED", "map": category_map_E},
        {"source": "PRES80", "dest": "PRES80_DIDNT_VOTE", "map": category_map_F},
        {"source": "PRES80", "dest": "PRES80_DONT_KNOW", "map": DONT_KNOW_map},
        {"source": "PRES80", "dest": "PRES80_DEMREP", "map": DEMREP_map},
        {"source": "PRES80", "dest": "PRES80_NONCONFORM", "map": NONCONFORM3P_map},

        # PRES84 transformations
        {"source": "PRES84", "dest": "PRES84_MONDALE", "map": category_map_A},
        {"source": "PRES84", "dest": "PRES84_REAGAN", "map": category_map_B},
        {"source": "PRES84", "dest": "PRES84_OTHER", "map": category_map_C},
        {"source": "PRES84", "dest": "PRES84_REFUSED", "map": category_map_D},
        {"source": "PRES84", "dest": "PRES84_NO_PRES_VOTE", "map": category_map_E},
        {"source": "PRES84", "dest": "PRES84_DONT_KNOW", "map": DONT_KNOW_map},
        {"source": "PRES84", "dest": "PRES84_DEMREP", "map": DEMREP_map},
        {"source": "PRES84", "dest": "PRES84_NONCONFORM", "map": NONCONFORM_map},

        # PRES88 transformations
        {"source": "PRES88", "dest": "PRES88_DUKAKIS", "map": category_map_A},
        {"source": "PRES88", "dest": "PRES88_BUSH", "map": category_map_B},
        {"source": "PRES88", "dest": "PRES88_OTHER", "map": category_map_C},
        {"source": "PRES88", "dest": "PRES88_REFUSED", "map": category_map_D},
        {"source": "PRES88", "dest": "PRES88_NO_PRES_VOTE", "map": category_map_E},
        {"source": "PRES88", "dest": "PRES88_DONT_KNOW", "map": DONT_KNOW_map},
        {"source": "PRES88", "dest": "PRES88_DEMREP", "map": DEMREP_map},
        {"source": "PRES88", "dest": "PRES88_NONCONFORM", "map": NONCONFORM_map},

        # PRES92 transformations
        {"source": "PRES92", "dest": "PRES92_CLINTON", "map": category_map_A},
        {"source": "PRES92", "dest": "PRES92_BUSH", "map": category_map_B},
        {"source": "PRES92", "dest": "PRES92_PEROT", "map": category_map_C},
        {"source": "PRES92", "dest": "PRES92_OTHER", "map": category_map_D},
        {"source": "PRES92", "dest": "PRES92_NO_PRES_VOTE", "map": category_map_E},
        {"source": "PRES92", "dest": "PRES92_DONT_KNOW", "map": DONT_KNOW_map},
        {"source": "PRES92", "dest": "PRES92_DEMREP", "map": DEMREP_map},
        {"source": "PRES92", "dest": "PRES92_NONCONFORM", "map": NONCONFORM3P_map},
        
        # PRES96 transformations
        {"source": "PRES96", "dest": "PRES96_CLINTON", "map": category_map_A},
        {"source": "PRES96", "dest": "PRES96_DOLE", "map": category_map_B},
        {"source": "PRES96", "dest": "PRES96_PEROT", "map": category_map_C},
        {"source": "PRES96", "dest": "PRES96_OTHER", "map": category_map_D},
        {"source": "PRES96", "dest": "PRES96_DIDNT_VOTE", "map": category_map_E},
        {"source": "PRES96", "dest": "PRES96_DONT_KNOW", "map": DONT_KNOW_map},
        {"source": "PRES96", "dest": "PRES96_DEMREP", "map": DEMREP_map},
        {"source": "PRES96", "dest": "PRES96_NONCONFORM", "map": NONCONFORM3P_map},
        
        # PRES00 transformations
        {"source": "PRES00", "dest": "PRES00_GORE", "map": category_map_A},
        {"source": "PRES00", "dest": "PRES00_BUSH", "map": category_map_B},
        {"source": "PRES00", "dest": "PRES00_NADER", "map": category_map_C},
        {"source": "PRES00", "dest": "PRES00_OTHER", "map": category_map_D},
        {"source": "PRES00", "dest": "PRES00_DIDNT_VOTE", "map": category_map_E},
        {"source": "PRES00", "dest": "PRES00_DONT_KNOW", "map": DONT_KNOW_map},
        {"source": "PRES00", "dest": "PRES00_DEMREP", "map": DEMREP_map},
        {"source": "PRES00", "dest": "PRES00_NONCONFORM", "map": NONCONFORM3P_map},
        
        # PRES04 transformations
        {"source": "PRES04", "dest": "PRES04_KERRY", "map": category_map_A},
        {"source": "PRES04", "dest": "PRES04_BUSH", "map": category_map_B},
        {"source": "PRES04", "dest": "PRES04_NADER", "map": category_map_C},
        {"source": "PRES04", "dest": "PRES04_NO_PRES_VOTE", "map": category_map_D},
        {"source": "PRES04", "dest": "PRES04_DONT_KNOW", "map": DONT_KNOW_map},
        {"source": "PRES04", "dest": "PRES04_DEMREP", "map": DEMREP_map},
        {"source": "PRES04", "dest": "PRES04_NONCONFORM", "map": NONCONFORM_map},
        
        # PRES08 transformations
        {"source": "PRES08", "dest": "PRES08_OBAMA", "map": category_map_A},
        {"source": "PRES08", "dest": "PRES08_MCCAIN", "map": category_map_B},
        {"source": "PRES08", "dest": "PRES08_OTHER", "map": category_map_C},
        {"source": "PRES08", "dest": "PRES08_DIDNT_VOTE", "map": category_map_D},
        {"source": "PRES08", "dest": "PRES08_DONT_KNOW", "map": DONT_KNOW_map},
        {"source": "PRES08", "dest": "PRES08_DEMREP", "map": DEMREP_map},
        {"source": "PRES08", "dest": "PRES08_NONCONFORM", "map": NONCONFORM_map},

        # PRES12 transformations
        {"source": "PRES12", "dest": "PRES12_OBAMA", "map": category_map_A},
        {"source": "PRES12", "dest": "PRES12_ROMNEY", "map": category_map_B},
        {"source": "PRES12", "dest": "PRES12_OTHER", "map": category_map_C},
        {"source": "PRES12", "dest": "PRES12_DIDNT_VOTE", "map": category_map_D},
        {"source": "PRES12", "dest": "PRES12_DONT_KNOW", "map": DONT_KNOW_map},
        {"source": "PRES12", "dest": "PRES12_DEMREP", "map": DEMREP_map},
        {"source": "PRES12", "dest": "PRES12_NONCONFORM", "map": NONCONFORM_map},
        
        # PRES16 transformations
        {"source": "PRES16", "dest": "PRES16_CLINTON", "map": category_map_A},
        {"source": "PRES16", "dest": "PRES16_TRUMP", "map": category_map_B},
        {"source": "PRES16", "dest": "PRES16_OTHER", "map": category_map_C},
        {"source": "PRES16", "dest": "PRES16_DIDNT_VOTE", "map": category_map_D},
        {"source": "PRES16", "dest": "PRES16_DONT_KNOW", "map": DONT_KNOW_map},
        {"source": "PRES16", "dest": "PRES16_DEMREP", "map": DEMREP_map},
        {"source": "PRES16", "dest": "PRES16_NONCONFORM", "map": NONCONFORM_map},
        
        # PRES20 transformations
        {"source": "PRES20", "dest": "PRES20_BIDEN", "map": category_map_A},
        {"source": "PRES20", "dest": "PRES20_TRUMP", "map": category_map_B},
        {"source": "PRES20", "dest": "PRES20_OTHER", "map": category_map_C},
        {"source": "PRES20", "dest": "PRES20_DIDNT_VOTE", "map": category_map_D},
        {"source": "PRES20", "dest": "PRES20_DONT_KNOW", "map": DONT_KNOW_map},
        {"source": "PRES20", "dest": "PRES20_DEMREP", "map": DEMREP_map},
        {"source": "PRES20", "dest": "PRES20_NONCONFORM", "map": NONCONFORM_map},

        # IF68WHO transformations
        {"source": "IF68WHO", "dest": "IF68WHO_HUMPHREY", "map": category_map_A},
        {"source": "IF68WHO", "dest": "IF68WHO_NIXON", "map": category_map_B},
        {"source": "IF68WHO", "dest": "IF68WHO_WALLACE", "map": category_map_C},
        {"source": "IF68WHO", "dest": "IF68WHO_OTHER", "map": category_map_D},
        {"source": "IF68WHO", "dest": "IF68WHO_WLDNT_VT_RELIG", "map": category_map_E},
        {"source": "IF68WHO", "dest": "IF68WHO_DONT_KNOW", "map": DONT_KNOW_map},
        
        # IF72WHO transformations
        {"source": "IF72WHO", "dest": "IF72WHO_MCGOVERN", "map": category_map_A},
        {"source": "IF72WHO", "dest": "IF72WHO_NIXON", "map": category_map_B},
        {"source": "IF72WHO", "dest": "IF72WHO_OTHER", "map": category_map_C},
        {"source": "IF72WHO", "dest": "IF72WHO_REFUSED", "map": category_map_D},
        {"source": "IF72WHO", "dest": "IF72WHO_WOULDNT_VOTE", "map": category_map_E},
        {"source": "IF72WHO", "dest": "IF72WHO_WLDNT_VT_RELIG", "map": category_map_F},
        {"source": "IF72WHO", "dest": "IF72WHO_DONT_KNOW", "map": DONT_KNOW_map},
        
        # IF76WHO transformations
        {"source": "IF76WHO", "dest": "IF76WHO_CARTER", "map": category_map_A},
        {"source": "IF76WHO", "dest": "IF76WHO_FORD", "map": category_map_B},
        {"source": "IF76WHO", "dest": "IF76WHO_OTHER", "map": category_map_C},
        {"source": "IF76WHO", "dest": "IF76WHO_REFUSED", "map": category_map_D},
        {"source": "IF76WHO", "dest": "IF76WHO_WOULDNT_VOTE", "map": category_map_E},
        {"source": "IF76WHO", "dest": "IF76WHO_DONT_KNOW", "map": DONT_KNOW_map},
        
        # IF80WHO transformations
        {"source": "IF80WHO", "dest": "IF80WHO_CARTER", "map": category_map_A},
        {"source": "IF80WHO", "dest": "IF80WHO_REAGAN", "map": category_map_B},
        {"source": "IF80WHO", "dest": "IF80WHO_ANDERSON", "map": category_map_C},
        {"source": "IF80WHO", "dest": "IF80WHO_OTHER", "map": category_map_D},
        {"source": "IF80WHO", "dest": "IF80WHO_WOULDNT_VOTE", "map": category_map_E},
        {"source": "IF80WHO", "dest": "IF80WHO_REFUSED", "map": category_map_F},
        {"source": "IF80WHO", "dest": "IF80WHO_DONT_KNOW", "map": DONT_KNOW_map},
        
        # IF84WHO transformations
        {"source": "IF84WHO", "dest": "IF84WHO_MONDALE", "map": category_map_A},
        {"source": "IF84WHO", "dest": "IF84WHO_REAGAN", "map": category_map_B},
        {"source": "IF84WHO", "dest": "IF84WHO_OTHER", "map": category_map_C},
        {"source": "IF84WHO", "dest": "IF84WHO_WOULDNT_VOTE", "map": category_map_D},
        {"source": "IF84WHO", "dest": "IF84WHO_DONT_KNOW", "map": DONT_KNOW_map},
        
        # IF88WHO transformations
        {"source": "IF88WHO", "dest": "IF88WHO_DUKAKIS", "map": category_map_A},
        {"source": "IF88WHO", "dest": "IF88WHO_BUSH", "map": category_map_B},
        {"source": "IF88WHO", "dest": "IF88WHO_OTHER", "map": category_map_C},
        {"source": "IF88WHO", "dest": "IF88WHO_DONT_KNOW", "map": DONT_KNOW_map},

        # IF92WHO transformations
        {"source": "IF92WHO", "dest": "IF92WHO_CLINTON", "map": category_map_A},
        {"source": "IF92WHO", "dest": "IF92WHO_BUSH", "map": category_map_B},
        {"source": "IF92WHO", "dest": "IF92WHO_PEROT", "map": category_map_C},
        {"source": "IF92WHO", "dest": "IF92WHO_OTHER", "map": category_map_D},
        {"source": "IF92WHO", "dest": "IF92WHO_DONT_KNOW", "map": DONT_KNOW_map},
        
        # IF96WHO transformations
        {"source": "IF96WHO", "dest": "IF96WHO_CLINTON", "map": category_map_A},
        {"source": "IF96WHO", "dest": "IF96WHO_DOLE", "map": category_map_B},
        {"source": "IF96WHO", "dest": "IF96WHO_PEROT", "map": category_map_C},
        {"source": "IF96WHO", "dest": "IF96WHO_OTHER", "map": category_map_D},
        {"source": "IF96WHO", "dest": "IF96WHO_DONT_KNOW", "map": DONT_KNOW_map},
        
        # IF00WHO transformations
        {"source": "IF00WHO", "dest": "IF00WHO_GORE", "map": category_map_A},
        {"source": "IF00WHO", "dest": "IF00WHO_BUSH", "map": category_map_B},
        {"source": "IF00WHO", "dest": "IF00WHO_NADER", "map": category_map_C},
        {"source": "IF00WHO", "dest": "IF00WHO_OTHER", "map": category_map_D},
        {"source": "IF00WHO", "dest": "IF00WHO_DONT_KNOW", "map": DONT_KNOW_map},
        
        # IF04WHO transformations
        {"source": "IF04WHO", "dest": "IF04WHO_KERRY", "map": category_map_A},
        {"source": "IF04WHO", "dest": "IF04WHO_BUSH", "map": category_map_B},
        {"source": "IF04WHO", "dest": "IF04WHO_NADER", "map": category_map_C},
        {"source": "IF04WHO", "dest": "IF04WHO_DONT_KNOW", "map": DONT_KNOW_map},
        
        # IF08WHO transformations
        {"source": "IF08WHO", "dest": "IF08WHO_OBAMA", "map": category_map_A},
        {"source": "IF08WHO", "dest": "IF08WHO_MCCAIN", "map": category_map_B},
        {"source": "IF08WHO", "dest": "IF08WHO_OTHER", "map": category_map_C},
        {"source": "IF08WHO", "dest": "IF08WHO_DONT_KNOW", "map": DONT_KNOW_map},
        
        # IF12WHO transformations
        {"source": "IF12WHO", "dest": "IF12WHO_OBAMA", "map": category_map_A},
        {"source": "IF12WHO", "dest": "IF12WHO_ROMNEY", "map": category_map_B},
        {"source": "IF12WHO", "dest": "IF12WHO_OTHER", "map": category_map_C},
        {"source": "IF12WHO", "dest": "IF12WHO_DONT_KNOW", "map": DONT_KNOW_map},

        # IF16WHO transformations
        {"source": "IF16WHO", "dest": "IF16WHO_CLINTON", "map": category_map_A},
        {"source": "IF16WHO", "dest": "IF16WHO_TRUMP", "map": category_map_B},
        {"source": "IF16WHO", "dest": "IF16WHO_OTHER", "map": category_map_C},
        {"source": "IF16WHO", "dest": "IF16WHO_CANT_REMEMBER", "map": category_map_D},
        {"source": "IF16WHO", "dest": "IF16WHO_DONT_KNOW", "map": DONT_KNOW_map},
        
        # IF20WHO transformations
        {"source": "IF20WHO", "dest": "IF20WHO_BIDEN", "map": category_map_A},
        {"source": "IF20WHO", "dest": "IF20WHO_TRUMP", "map": category_map_B},
        {"source": "IF20WHO", "dest": "IF20WHO_OTHER", "map": category_map_C},
        {"source": "IF20WHO", "dest": "IF20WHO_CANT_REMEMBER", "map": category_map_D},
        {"source": "IF20WHO", "dest": "IF20WHO_DONT_KNOW", "map": DONT_KNOW_map},
        
        # POLVIEWS transformation
        {"source": "POLVIEWS", "dest": "POLVIEWS", "map": POLVIEWS_map},
        
        # NAT mappings
        {"source": "NATSPAC", "dest": "NATSPAC", "map": NAT_map},
        {"source": "NATENVIR", "dest": "NATENVIR", "map": NAT_map},
        {"source": "NATHEAL", "dest": "NATHEAL", "map": NAT_map},
        {"source": "NATCITY", "dest": "NATCITY", "map": NAT_map},
        {"source": "NATCRIME", "dest": "NATCRIME", "map": NAT_map},
        {"source": "NATDRUG", "dest": "NATDRUG", "map": NAT_map},
        {"source": "NATEDUC", "dest": "NATEDUC", "map": NAT_map},
        {"source": "NATRACE", "dest": "NATRACE", "map": NAT_map},
        {"source": "NATARMS", "dest": "NATARMS", "map": NAT_map},
        {"source": "NATAID", "dest": "NATAID", "map": NAT_map},
        {"source": "NATFARE", "dest": "NATFARE", "map": NAT_map},
        {"source": "NATROAD", "dest": "NATROAD", "map": NAT_map},
        {"source": "NATSOC", "dest": "NATSOC", "map": NAT_map},
        {"source": "NATMASS", "dest": "NATMASS", "map": NAT_map},
        {"source": "NATPARK", "dest": "NATPARK", "map": NAT_map},
        {"source": "NATCHLD", "dest": "NATCHLD", "map": NAT_map},
        {"source": "NATSCI", "dest": "NATSCI", "map": NAT_map},
        {"source": "NATENRGY", "dest": "NATENRGY", "map": NAT_map},
    
        {"source": "NATSPACY", "dest": "NATSPACY", "map": NAT_map},
        {"source": "NATENVIY", "dest": "NATENVIY", "map": NAT_map},
        {"source": "NATHEALY", "dest": "NATHEALY", "map": NAT_map},
        {"source": "NATCITYY", "dest": "NATCITYY", "map": NAT_map},
        {"source": "NATCRIMY", "dest": "NATCRIMY", "map": NAT_map},
        {"source": "NATDRUGY", "dest": "NATDRUGY", "map": NAT_map},
        {"source": "NATEDUCY", "dest": "NATEDUCY", "map": NAT_map},
        {"source": "NATRACEY", "dest": "NATRACEY", "map": NAT_map},
        {"source": "NATARMSY", "dest": "NATARMSY", "map": NAT_map},
        {"source": "NATAIDY", "dest": "NATAIDY", "map": NAT_map},
        {"source": "NATFAREY", "dest": "NATFAREY", "map": NAT_map},
        
        {"source": "EQWLTH", "dest": "EQWLTH", "map": EQWLTH_map},
        
        {"source": "SPKATH", "dest": "SPKATH", "map": SPK_map},
        {"source": "SPKRAC", "dest": "SPKRAC", "map": SPK_map},
        {"source": "SPKCOM", "dest": "SPKCOM", "map": SPK_map},
        {"source": "SPKMIL", "dest": "SPKMIL", "map": SPK_map},
        {"source": "SPKHOMO", "dest": "SPKHOMO", "map": SPK_map},
        {"source": "SPKMSLM", "dest": "SPKMSLM", "map": SPK_map},
        
        {"source": "COLATH", "dest": "COLATH", "map": COLATH_map},
        {"source": "COLRAC", "dest": "COLRAC", "map": COLATH_map},
        {"source": "COLCOM", "dest": "COLCOM", "map": COLATH_map},
        {"source": "COLMIL", "dest": "COLMIL", "map": COLATH_map},
        {"source": "COLHOMO", "dest": "COLHOMO", "map": COLATH_map},
        {"source": "COLMSLM", "dest": "COLMSLM", "map": COLATH_map},
        
        {"source": "LIBATH", "dest": "LIBATH", "map": LIB_map},
        {"source": "LIBRAC", "dest": "LIBRAC", "map": LIB_map},
        {"source": "LIBCOM", "dest": "LIBCOM", "map": LIB_map},
        {"source": "LIBMIL", "dest": "LIBMIL", "map": LIB_map},
        {"source": "LIBHOMO", "dest": "LIBHOMO", "map": LIB_map},
        {"source": "LIBMSLM", "dest": "LIBMSLM", "map": LIB_map},

        {"source": "CAPPUN", "dest": "CAPPUN", "map": POL1_map},
        {"source": "GUNLAW", "dest": "GUNLAW", "map": POL1_map},

        {"source": "COURTS", "dest": "COURTS", "map": COURTS_map},

        {"source": "GRASS", "dest": "GRASS", "map": GRASS_map},

        {"source": "RELIG", "dest": "RELIG_PROT", "map": category_map_A},
        {"source": "RELIG", "dest": "RELIG_CATHOLIC", "map": category_map_B},
        {"source": "RELIG", "dest": "RELIG_JEWISH", "map": category_map_C},
        {"source": "RELIG", "dest": "RELIG_NONE", "map": category_map_D},
        {"source": "RELIG", "dest": "RELIG_OTHER", "map": category_map_E},
        {"source": "RELIG", "dest": "RELIG_BUDDHISM", "map": category_map_F},
        {"source": "RELIG", "dest": "RELIG_HINDUISM", "map": category_map_G},
        {"source": "RELIG", "dest": "RELIG_OTHER_EASTERN", "map": category_map_H}, # adjusted for clarity
        {"source": "RELIG", "dest": "RELIG_MUSLIM_ISLAM", "map": category_map_I},
        {"source": "RELIG", "dest": "RELIG_ORTHODOX_CHRISTIAN", "map": category_map_J},
        {"source": "RELIG", "dest": "RELIG_CHRISTIAN", "map": category_map_K},
        {"source": "RELIG", "dest": "RELIG_NATIVE_AMERICAN", "map": category_map_L},
        {"source": "RELIG", "dest": "RELIG_INTER_NONDENOMINATIONAL", "map": category_map_M},

        {"source": "ATTEND", "dest": "ATTEND", "map": ATTEND_map},

        {"source": "RELITEN", "dest": "RELITEN", "map": RELITEN_map},

        {"source": "POSTLIFE", "dest": "POSTLIFE", "map": POSTLIFE_map},

        {"source": "PRAYER", "dest": "PRAYER", "map": PRAYER_map},

        {"source": "RACOPEN", "dest": "RACOPEN", "map": RACOPEN_map},

        {"source": "AFFRMACT", "dest": "AFFRMACT", "map": AFFRMACT_map},

        {"source": "WRKWAYUP", "dest": "WRKWAYUP", "map": WRKWAYUP_map},

        {"source": "HELPFUL", "dest": "HELPFUL", "map": HELPFUL_map},

        {"source": "FAIR", "dest": "FAIR", "map": FAIR_map},

        {"source": "TRUST", "dest": "TRUST", "map": TRUST_map},

        {"source": "CONFINAN", "dest": "CONFINAN", "map": CON_map},
        {"source": "CONBUS", "dest": "CONBUS", "map": CON_map},
        {"source": "CONCLERG", "dest": "CONCLERG", "map": CON_map},
        {"source": "CONEDUC", "dest": "CONEDUC", "map": CON_map},
        {"source": "CONFED", "dest": "CONFED", "map": CON_map},
        {"source": "CONLABOR", "dest": "CONLABOR", "map": CON_map},
        {"source": "CONPRESS", "dest": "CONPRESS", "map": CON_map},
        {"source": "CONMEDIC", "dest": "CONMEDIC", "map": CON_map},
        {"source": "CONTV", "dest": "CONTV", "map": CON_map},
        {"source": "CONJUDGE", "dest": "CONJUDGE", "map": CON_map},
        {"source": "CONSCI", "dest": "CONSCI", "map": CON_map},
        {"source": "CONLEGIS", "dest": "CONLEGIS", "map": CON_map},
        {"source": "CONARMY", "dest": "CONARMY", "map": CON_map},

        {"source": "OBEY", "dest": "OBEY", "map": KID_map},
        {"source": "POPULAR", "dest": "POPULAR", "map": KID_map},
        {"source": "THNKSELF", "dest": "THNKSELF", "map": KID_map},
        {"source": "WORKHARD", "dest": "WORKHARD", "map": KID_map},
        {"source": "HELPOTH", "dest": "HELPOTH", "map": KID_map},

        {"source": "GETAHEAD", "dest": "GETAHEAD", "map": GETAHEAD_map},

        {"source": "FEPOL", "dest": "FEPOL", "map": FEPOL_map},

        {"source": "ABDEFECT", "dest": "ABDEFECT", "map": AB_map},
        {"source": "ABNOMORE", "dest": "ABNOMORE", "map": AB_map},
        {"source": "ABHLTH", "dest": "ABHLTH", "map": AB_map},
        {"source": "ABPOOR", "dest": "ABPOOR", "map": AB_map},
        {"source": "ABRAPE", "dest": "ABRAPE", "map": AB_map},
        {"source": "ABSINGLE", "dest": "ABSINGLE", "map": AB_map},
        {"source": "ABANY", "dest": "ABANY", "map": AB_map},

        {"source": "SEXEDUC", "dest": "SEXEDUC", "map": SEXEDUC_map},

        {"source": "DIVLAW", "dest": "DIVLAW", "map": DIVLAW_map},

        {"source": "PREMARSX", "dest": "PREMARSX", "map": SEX_map},
        {"source": "TEENSEX", "dest": "TEENSEX", "map": SEX_map},
        {"source": "XMARSEX", "dest": "XMARSEX", "map": SEX_map},
        {"source": "HOMOSEX", "dest": "HOMOSEX", "map": SEX_map},

        {"source": "PORNLAW", "dest": "PORNLAW", "map": PORNLAW_map},

        {"source": "SPANKING", "dest": "SPANKING", "map": SPANKING_map},

        {"source": "LETDIE1", "dest": "LETDIE1", "map": DEATH_map},
        {"source": "SUICIDE1", "dest": "SUICIDE1", "map": DEATH_map},
        {"source": "SUICIDE2", "dest": "SUICIDE2", "map": DEATH_map},

        {"source": "POLHITOK", "dest": "POLHITOK", "map": POLICE_map},
        {"source": "POLABUSE", "dest": "POLABUSE", "map": POLICE_map},
        {"source": "POLMURDR", "dest": "POLMURDR", "map": POLICE_map},
        {"source": "POLESCAP", "dest": "POLESCAP", "map": POLICE_map},
        {"source": "POLATTAK", "dest": "POLATTAK", "map": POLICE_map},

        {"source": "NEWS", "dest": "NEWS", "map": NEWS_map},

        {"source": "TVHOURS", "dest": "TVHOURS", "map": TVHOURS_map},

        {"source": "FECHLD", "dest": "FECHLD", "map": FE_map},
        {"source": "FEPRESCH", "dest": "FEPRESCH", "map": FE_map},
        {"source": "FEFAM", "dest": "FEFAM", "map": FE_map},

        {"source": "RACDIF1", "dest": "RACDIF1", "map": RACDIF_map},
        {"source": "RACDIF2", "dest": "RACDIF2", "map": RACDIF_map},
        {"source": "RACDIF3", "dest": "RACDIF3", "map": RACDIF_map},
        {"source": "RACDIF4", "dest": "RACDIF4", "map": RACDIF_map},

        {"source": "HELPPOOR", "dest": "HELPPOOR", "map": HELP_map},
        {"source": "HELPNOT", "dest": "HELPNOT", "map": HELP_map},
        {"source": "HELPBLK", "dest": "HELPBLK", "map": HELP_map},

        {"source": "MARHOMO", "dest": "MARHOMO", "map": MARHOMO_map}

    ]

    transformed_data = {}
    metadata_data = {}

    preserved_columns = ["YEAR", "ID", "BALLOT"]
    for col in preserved_columns:
        transformed_data[col] = df[col]

    for transformation in transformations_to_do:
        source_col = transformation["source"]
        dest_col = transformation["dest"]
        column_map = transformation["map"]

        transformed_data[dest_col] = df[source_col].map(column_map)
        metadata_data[dest_col] = {"min": min(column_map.values()), "max": max(column_map.values())}

    transformed_df = pd.DataFrame(transformed_data)
    metadata_df = pd.DataFrame.from_dict(metadata_data, orient='index')
    
    transformed_df, metadata_df = make_vote_supernodes(transformed_df, metadata_df, varnames=["VOTE{year}", "PRES{year}_NONCONFORM", "PRES{year}_DEMREP"])

    if combine_variants:
        for variant, original in variants.items():
            combined_column = transformed_df[original].combine_first(transformed_df[variant])
            transformed_df[original] = combined_column
    
    transformed_df = transformed_df.drop(variants.keys(), axis=1)


    if check_for_duplicate_column_names(transformed_df):
        print("Warning. There are duplicate column names in the transformed dataframe.")
        # And print the duplicate column names
        print(check_for_duplicate_column_names(transformed_df))


    return transformed_df, metadata_df