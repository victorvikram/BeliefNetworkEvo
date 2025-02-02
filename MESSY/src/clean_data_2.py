import pandas as pd
import pyreadstat as prs
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np


# def make_variable_summary(df):
#     """
#     this function goes year by year and for each ballot puts in the percentage of non nan answers to the question
#     """
#     counts = df.groupby(['YEAR', 'BALLOT'], dropna=False).count()

#     notnull_df = df.notna()
#     notnull_df['YEAR'] = df['YEAR']
#     notnull_df['BALLOT'] = df['BALLOT']

#     pcts = notnull_df.groupby(['YEAR', 'BALLOT'], dropna=False).mean()


#     partially_complete_ballot_mask = (0 < pcts) & (pcts < 0.8)
#     pc_row, pc_col = np.where(partially_complete_ballot_mask)
#     partially_complete_ballot = [(partially_complete_ballot_mask.index[row_ind], partially_complete_ballot_mask.columns[col_ind]) for (row_ind, col_ind) in zip(pc_row, pc_col)]

#     return counts, pcts, partially_complete_ballot


# def normalize_columns(df, meta_df, exclude=["YEAR", "BALLOT", "ID"]):
#     """
#     transforms the df so that the range of each column is 1. signed variables go from [-0.5, 0.5], others go from [0, 1]
#     """

#     cols_to_use = [col for col in df.columns if col not in exclude]

#     range_of_values = meta_df["max"] - meta_df["min"]
#     normalized_df = df.loc[:,cols_to_use].div(range_of_values[cols_to_use])
    
#     for col in exclude:
#         if col in df.columns:
#             normalized_df[col] = df[col]

#     return normalized_df


def get_median_response(df, time_frame, variable):
    """
    this function takes in the dataframe, a set of relavent years (list of years), and a variable of interest
    and returns the median response for that variable across the year of interest
    """    
    median_response = df.loc[df['YEAR'].isin(time_frame), variable].median()
    return median_response

def contains_nans(mapping):
    """Check if a mapping contains NaNs."""
    return any(pd.isna(v) for v in mapping.values())

def check_for_duplicate_column_names(df):
    """
    this function takes in a dataframe and returns a list of column names that are duplicated
    """
    return df.columns[df.columns.duplicated()].tolist()



# The purpose of this script is to transform the dataframe into a format that can be used for the ising model.
# We want to make all variables range from -1 to 1, with -1 being the anti-belief, 0 being the "neutral" or lack-of belief, and 1 being belief.
# Some variables have a natual -1/1 intepretation (binary beliefs).
# Others have an ordinal scale with a natural 0 point. 
# But some variables are tricky -- there is no natural intepretation for a zero point. For these, we will set the zero point at the median value of the variable.  

def transform_dataframe_2(df, time_frame, combine_variants=True):

    """
    note: if value is not in map it goes to NaN
    """

    # Limit the df to the time frame of interest
    df = df.loc[df['YEAR'].isin(time_frame)]
    

    # All variables. Keep in mind that these are the variables AFTER the initial transformation, hence the many presidential variables.
    columns = ['YEAR', 'ID', 'BALLOT', 'PARTYID', 'OTHER_PARTY', 'VOTE68', 'VOTE72', 'VOTE76', 'VOTE80', 'VOTE84', 'VOTE88', 'VOTE92', 'VOTE96', 'VOTE00', 'VOTE04', 'VOTE08', 
               'VOTE12', 'VOTE16', 'VOTE20', 'VOTE68_ELIGIBLE', 'VOTE72_ELIGIBLE', 'VOTE76_ELIGIBLE', 'VOTE80_ELIGIBLE', 'VOTE84_ELIGIBLE', 'VOTE88_ELIGIBLE', 'VOTE92_ELIGIBLE', 
               'VOTE96_ELIGIBLE', 'VOTE00_ELIGIBLE', 'VOTE04_ELIGIBLE', 'VOTE08_ELIGIBLE', 'VOTE12_ELIGIBLE', 'VOTE16_ELIGIBLE', 'VOTE20_ELIGIBLE', 'VOTE68_DONT_KNOW', 
               'VOTE72_DONT_KNOW', 'VOTE76_DONT_KNOW', 'VOTE80_DONT_KNOW', 'VOTE84_DONT_KNOW', 'VOTE88_DONT_KNOW', 'VOTE92_DONT_KNOW', 'VOTE96_DONT_KNOW', 'VOTE00_DONT_KNOW', 
               'VOTE04_DONT_KNOW', 'VOTE08_DONT_KNOW', 'VOTE12_DONT_KNOW', 'VOTE16_DONT_KNOW', 'VOTE20_DONT_KNOW', 'PRES68_HUMPHREY', 'PRES68_NIXON', 'PRES68_WALLACE',
               'PRES68_OTHER', 'PRES68_REFUSED', 'PRES68_DEMREP', 'PRES68_NONCONFORM', 'PRES72_MCGOVERN', 'PRES72_NIXON', 'PRES72_OTHER', 'PRES72_REFUSED', 
               'PRES72_WOULDNT_VOTE', 'PRES72_DONT_KNOW', 'PRES72_DEMREP', 'PRES72_NONCONFORM', 'PRES76_CARTER', 'PRES76_FORD', 'PRES76_OTHER', 'PRES76_REFUSED', 
               'PRES76_NO_PRES_VOTE', 'PRES76_DONT_KNOW', 'PRES76_DEMREP', 'PRES76_NONCONFORM', 'PRES80_CARTER', 'PRES80_REAGAN', 'PRES80_ANDERSON', 'PRES80_OTHER', 
               'PRES80_REFUSED', 'PRES80_DIDNT_VOTE', 'PRES80_DONT_KNOW', 'PRES80_DEMREP', 'PRES80_NONCONFORM', 'PRES84_MONDALE', 'PRES84_REAGAN', 'PRES84_OTHER',
               'PRES84_REFUSED', 'PRES84_NO_PRES_VOTE', 'PRES84_DONT_KNOW', 'PRES84_DEMREP', 'PRES84_NONCONFORM', 'PRES88_DUKAKIS', 'PRES88_BUSH', 'PRES88_OTHER',
               'PRES88_REFUSED', 'PRES88_NO_PRES_VOTE', 'PRES88_DONT_KNOW', 'PRES88_DEMREP', 'PRES88_NONCONFORM', 'PRES92_CLINTON', 'PRES92_BUSH', 'PRES92_PEROT', 
               'PRES92_OTHER', 'PRES92_NO_PRES_VOTE', 'PRES92_DONT_KNOW', 'PRES92_DEMREP', 'PRES92_NONCONFORM', 'PRES96_CLINTON', 'PRES96_DOLE', 'PRES96_PEROT', 
               'PRES96_OTHER', 'PRES96_DIDNT_VOTE', 'PRES96_DONT_KNOW', 'PRES96_DEMREP', 'PRES96_NONCONFORM', 'PRES00_GORE', 'PRES00_BUSH', 'PRES00_NADER', 'PRES00_OTHER', 
               'PRES00_DIDNT_VOTE', 'PRES00_DONT_KNOW', 'PRES00_DEMREP', 'PRES00_NONCONFORM', 'PRES04_KERRY', 'PRES04_BUSH', 'PRES04_NADER', 'PRES04_NO_PRES_VOTE',
               'PRES04_DONT_KNOW', 'PRES04_DEMREP', 'PRES04_NONCONFORM', 'PRES08_OBAMA', 'PRES08_MCCAIN', 'PRES08_OTHER', 'PRES08_DIDNT_VOTE', 'PRES08_DONT_KNOW', 
               'PRES08_DEMREP', 'PRES08_NONCONFORM', 'PRES12_OBAMA', 'PRES12_ROMNEY', 'PRES12_OTHER', 'PRES12_DIDNT_VOTE', 'PRES12_DONT_KNOW', 'PRES12_DEMREP',
               'PRES12_NONCONFORM', 'PRES16_CLINTON', 'PRES16_TRUMP', 'PRES16_OTHER', 'PRES16_DIDNT_VOTE', 'PRES16_DONT_KNOW', 'PRES16_DEMREP', 'PRES16_NONCONFORM', 
               'PRES20_BIDEN', 'PRES20_TRUMP', 'PRES20_OTHER', 'PRES20_DIDNT_VOTE', 'PRES20_DONT_KNOW', 'PRES20_DEMREP', 'PRES20_NONCONFORM', 'IF68WHO_HUMPHREY', 
               'IF68WHO_NIXON', 'IF68WHO_WALLACE', 'IF68WHO_OTHER', 'IF68WHO_WLDNT_VT_RELIG', 'IF68WHO_DONT_KNOW', 'IF72WHO_MCGOVERN', 'IF72WHO_NIXON', 'IF72WHO_OTHER', 
               'IF72WHO_REFUSED', 'IF72WHO_WOULDNT_VOTE', 'IF72WHO_WLDNT_VT_RELIG', 'IF72WHO_DONT_KNOW', 'IF76WHO_CARTER', 'IF76WHO_FORD', 'IF76WHO_OTHER', 'IF76WHO_REFUSED', 
               'IF76WHO_WOULDNT_VOTE', 'IF76WHO_DONT_KNOW', 'IF80WHO_CARTER', 'IF80WHO_REAGAN', 'IF80WHO_ANDERSON', 'IF80WHO_OTHER', 'IF80WHO_WOULDNT_VOTE', 'IF80WHO_REFUSED', 
               'IF80WHO_DONT_KNOW', 'IF84WHO_MONDALE', 'IF84WHO_REAGAN', 'IF84WHO_OTHER', 'IF84WHO_WOULDNT_VOTE', 'IF84WHO_DONT_KNOW', 'IF88WHO_DUKAKIS', 'IF88WHO_BUSH', 
               'IF88WHO_OTHER', 'IF88WHO_DONT_KNOW', 'IF92WHO_CLINTON', 'IF92WHO_BUSH', 'IF92WHO_PEROT', 'IF92WHO_OTHER', 'IF92WHO_DONT_KNOW', 'IF96WHO_CLINTON', 'IF96WHO_DOLE', 
               'IF96WHO_PEROT', 'IF96WHO_OTHER', 'IF96WHO_DONT_KNOW', 'IF00WHO_GORE', 'IF00WHO_BUSH', 'IF00WHO_NADER', 'IF00WHO_OTHER', 'IF00WHO_DONT_KNOW', 'IF04WHO_KERRY',
                 'IF04WHO_BUSH', 'IF04WHO_NADER', 'IF04WHO_DONT_KNOW', 'IF08WHO_OBAMA', 'IF08WHO_MCCAIN', 'IF08WHO_OTHER', 'IF08WHO_DONT_KNOW', 'IF12WHO_OBAMA', 'IF12WHO_ROMNEY', 
                 'IF12WHO_OTHER', 'IF12WHO_DONT_KNOW', 'IF16WHO_CLINTON', 'IF16WHO_TRUMP', 'IF16WHO_OTHER', 'IF16WHO_CANT_REMEMBER', 'IF16WHO_DONT_KNOW', 'IF20WHO_BIDEN', 
                 'IF20WHO_TRUMP', 'IF20WHO_OTHER', 'IF20WHO_CANT_REMEMBER', 'IF20WHO_DONT_KNOW', 'POLVIEWS', 'NATSPAC', 'NATENVIR', 'NATHEAL', 'NATCITY', 'NATCRIME', 'NATDRUG', 
                 'NATEDUC', 'NATRACE', 'NATARMS', 'NATAID', 'NATFARE', 'NATROAD', 'NATSOC', 'NATMASS', 'NATPARK', 'NATCHLD', 'NATSCI', 'NATENRGY', 'EQWLTH', 'SPKATH', 'SPKRAC', 
                 'SPKCOM', 'SPKMIL', 'SPKHOMO', 'SPKMSLM', 'COLATH', 'COLRAC', 'COLCOM', 'COLMIL', 'COLHOMO', 'COLMSLM', 'LIBATH', 'LIBRAC', 'LIBCOM', 'LIBMIL', 'LIBHOMO', 
                 'LIBMSLM', 'CAPPUN', 'GUNLAW', 'COURTS', 'GRASS', 'RELIG_PROT', 'RELIG_CATHOLIC', 'RELIG_JEWISH', 'RELIG_NONE', 'RELIG_OTHER', 'RELIG_BUDDHISM', 'RELIG_HINDUISM', 
                 'RELIG_OTHER_EASTERN', 'RELIG_MUSLIM_ISLAM', 'RELIG_ORTHODOX_CHRISTIAN', 'RELIG_CHRISTIAN', 'RELIG_NATIVE_AMERICAN', 'RELIG_INTER_NONDENOMINATIONAL', 'ATTEND', 
                 'RELITEN', 'POSTLIFE', 'PRAYER', 'RACOPEN', 'AFFRMACT', 'WRKWAYUP', 'HELPFUL', 'FAIR', 'TRUST', 'CONFINAN', 'CONBUS', 'CONCLERG', 'CONEDUC', 'CONFED', 'CONLABOR',
                   'CONPRESS', 'CONMEDIC', 'CONTV', 'CONJUDGE', 'CONSCI', 'CONLEGIS', 'CONARMY', 'OBEY', 'POPULAR', 'THNKSELF', 'WORKHARD', 'HELPOTH', 'GETAHEAD', 'FEPOL',
                     'ABDEFECT', 'ABNOMORE', 'ABHLTH', 'ABPOOR', 'ABRAPE', 'ABSINGLE', 'ABANY', 'SEXEDUC', 'DIVLAW', 'PREMARSX', 'TEENSEX', 'XMARSEX', 'HOMOSEX', 'PORNLAW', 
                     'SPANKING', 'LETDIE1', 'SUICIDE1', 'SUICIDE2', 'POLHITOK', 'POLABUSE', 'POLMURDR', 'POLESCAP', 'POLATTAK', 'NEWS', 'TVHOURS', 'FECHLD', 'FEPRESCH', 'FEFAM',
                       'RACDIF1', 'RACDIF2', 'RACDIF3', 'RACDIF4', 'HELPPOOR', 'HELPNOT', 'HELPBLK', 'MARHOMO', 'VOTELAST', 'PRESLAST_NONCONFORM', 'PRESLAST_DEMREP']


    #####################################################################################################################################################################
    #####################################################################################################################################################################

    # Nuanced cases: median approach for setting up ordinality
    """Here we address the special case of variables that ARE ordinal but have no natural way to set a zero point. For stress calculations in ising models, the zero point is very important to get right.
    The simplest yet principled way to do this is to set the zero point at the median value of the variable. The basic intepretation is that the median value is the best guess at the stance least involved in
    increasing or decreasing stress. 
    
    Here we map the median response of variables to "lack-of-belief / middle-ground-stance" (0) and then map the rest of the values to "anti-belief" (-1) and "belief" (1), evenly spaced about zero"""
    
    """They are: = [EQWLTH,	ATTEND,	RELITEN,	CONFINAN,	CONBUS,	CONCLERG,	CONEDUC,	CONFED,	CONLABOR,	CONPRESS,	CONMEDIC,	CONTV,	CONJUDGE,	CONSCI,	CONLEGIS,	CONARMY,	OBEY,	POPULAR,
        	THNKSELF,	WORKHARD,	HELPOTH,	PREMARSX,	TEENSEX,	XMARSEX,	HOMOSEX,	NEWS,	TVHOURS]"""
    
    # EQWLTH: Should govt reduce income differences?
    # Set the mapping around the median value
    original_EQWLTH_map = {1: 3, 2: 2, 3: 1, 4: 0, 5: -1, 6: -2, 7: -3}
    median = get_median_response(df, time_frame, "EQWLTH")
    EQWLTH_map = {key: (key - median) for key in original_EQWLTH_map}
    #DONT_KNOW_map works here

    # ATTEND: How often do you attend religious services? 
    original_ATTEND_map = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8}
    median = get_median_response(df, time_frame, "ATTEND")
    ATTEND_map = {key: (key - median) for key in original_ATTEND_map}
    
    # RELITEN: Would you call yourself a strongly religious or a not very strong?
    original_RELITEN_map = {1: 3, 2: 2, 3: 1, 4: 0}
    median = get_median_response(df, time_frame, "RELITEN")
    RELITEN_map = {key: (key - median) for key in original_RELITEN_map}   
    NO_ANSWER_map = {-99: 1, 1: -1, 2: -1, 3: -1, 4: -1, 5: -1, 6: -1, 7: -1}

    # Confidence variables: CONFINAN, CONBUS, CONCLERG, CONEDUC, CONFED, CONLABOR, CONPRESS, CONMEDIC, CONTV, CONJUDGE, CONSCI, CONLEGIS, CONARMY
    original_CON_map = {1: 2, 2: 1, 3: 0}
    for ID in ["CONFINAN", "CONBUS", "CONCLERG", "CONEDUC", "CONFED", "CONLABOR", "CONPRESS", "CONMEDIC", "CONTV", "CONJUDGE", "CONSCI", "CONLEGIS", "CONARMY"]:
        median = get_median_response(df, time_frame, ID)
        new_map = {key: (key - median) for key in original_CON_map}
        globals()[f"{ID}_map"] = new_map

    # Kid learning ranking: OBEY, POPULAR, THNKSELF, WORKHARD, HELPOTH
    original_KID_map = {1: 4, 2: 3, 3: 2, 4: 1, 5: 0}
    for ID in ["OBEY", "POPULAR", "THNKSELF", "WORKHARD", "HELPOTH"]:
        median = get_median_response(df, time_frame, ID)
        new_map = {key: (key - median) for key in original_KID_map}
        globals()[f"{ID}_map"] = new_map
    
    # SEX: PREMARSX, TEENSEX, XMARSEX, HOMOSEX
    original_SEX_map = {1: 3, 2: 2, 3: 1, 4: 0}
    for ID in ["PREMARSX", "TEENSEX", "XMARSEX", "HOMOSEX"]:
        median = get_median_response(df, time_frame, ID)
        new_map = {key: (key - median) for key in original_SEX_map}
        globals()[f"{ID}_map"] = new_map
    
    # PORNLAW: Should laws forbid porn?
    original_PORNLAW_map = {1: 2, 2: 1, 3: 0}
    median = get_median_response(df, time_frame, "PORNLAW")
    PORNLAW_map = {key: (key - median) for key in original_PORNLAW_map}

    # NEWS: How often do you read the newspaper?
    original_NEWS_map = {1: 4, 2: 3, 3: 2, 4: 1, 5: 0}
    median = get_median_response(df, time_frame, "NEWS")
    NEWS_map = {key: (key - median) for key in original_NEWS_map}

    # TVHOURS: On average, how many hours per day on TV?
    original_TVHOURS_map = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10, 11:11, 12:12, 13:13, 14:14, 15:15, 16:16, 17:17, 18:18, 19:19, 20:20, 21:21, 22:22, 23:23, 24:24}
    median = get_median_response(df, time_frame, "TVHOURS")
    TVHOURS_map = {key: (key - median) for key in original_TVHOURS_map}

    #####################################################################################################################################################################
    #####################################################################################################################################################################

    # make a dataframe that for each variable gives the min, max, and step
    metadata_df = pd.DataFrame({"min": [], "max": []})


    mappings = [EQWLTH_map,ATTEND_map,RELITEN_map,CONFINAN_map,CONBUS_map,CONCLERG_map,CONEDUC_map,CONFED_map,CONLABOR_map,CONPRESS_map,CONMEDIC_map,CONTV_map,CONJUDGE_map,
                 CONSCI_map,CONLEGIS_map,CONARMY_map,OBEY_map,POPULAR_map,THNKSELF_map,WORKHARD_map,HELPOTH_map,PREMARSX_map,TEENSEX_map,XMARSEX_map,HOMOSEX_map,PORNLAW_map,
                 NEWS_map,TVHOURS_map]
    columns_to_map = ["EQWLTH","ATTEND","RELITEN","CONFINAN","CONBUS","CONCLERG","CONEDUC","CONFED","CONLABOR","CONPRESS","CONMEDIC","CONTV","CONJUDGE",
                 "CONSCI","CONLEGIS","CONARMY","OBEY","POPULAR","THNKSELF","WORKHARD","HELPOTH","PREMARSX","TEENSEX","XMARSEX","HOMOSEX","PORNLAW",
                 "NEWS","TVHOURS"]
    
    transformations_to_do = [
        {"source": var_name, "dest": var_name, "map": map_name} for var_name, map_name in zip(columns_to_map, mappings) if not contains_nans(map_name)
    ]

    transformed_data = {}
    metadata_data = {}

    # Preserved columns should be all columns that are not transformed
    all_columns = df.columns
    preserved_columns = [col for col in all_columns if col not in [transformation["source"] for transformation in transformations_to_do]]
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

    transformed_df = transformed_df.copy()

    # If there are variables that have all NaN, print a warning and delete them
    # variables_with_all_nan = transformed_df.columns[transformed_df.isna().all()]
    # if len(variables_with_all_nan) > 0:
    #     print(f"{len(variables_with_all_nan)} variables have all NaN values. They will be removed from the dataframe.")
    #     print(variables_with_all_nan)
    #     transformed_df = transformed_df.drop(columns=variables_with_all_nan)


    return transformed_df, metadata_df