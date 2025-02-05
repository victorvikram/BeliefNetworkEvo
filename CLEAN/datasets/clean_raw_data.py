"""
GSS Data Cleaning Script
-----------------------

This script provides functions to clean and transform GSS data in two ways:
1. Regular version: Variables normalized to [-1, 1] with natural zero points preserved
2. Median-centered version: Variables normalized to [-1, 1] and centered around their median


For future reference, the column codes that we will use are:
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



The script handles:
Case 1
- Symmetric questions: mapping to [-1, 1] with natural zero point preserved
- Asymmetric questions: mapping to [0, 1] with natural zero point preserved

Case 2
- Symmetric questions: mapping to [-1, 1] with *median response* as zero point
- Asymmetric questions: mapping to [0, 1] with natural zero point preserved
"""

#------------------------------------------------------------------------------
# Imports and Setup
#------------------------------------------------------------------------------

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Union
from dataclasses import dataclass
import warnings
import sys
from pathlib import Path
import os

pd.set_option('future.no_silent_downcasting', True) # Should suppress a warning message relating to line 255.

# Add the parent directory to Python path
sys.path.append(str(Path(__file__).parent))

from import_gss import import_dataset

#------------------------------------------------------------------------------
# Data Configuration Class
#------------------------------------------------------------------------------

@dataclass
class DataConfig:
    """Configuration for data cleaning and transformation.
    
    The GSS variables are grouped into two main categories:
    
    symmetric_questions = ['PARTYID', 'POLVIEWS', 'NATSPAC', 'NATENVIR', 'NATHEAL', 'NATCITY', 'NATCRIME', 'NATDRUG', 
                       'NATEDUC', 'NATRACE', 'NATARMS', 'NATAID', 'NATFARE', 'NATROAD', 'NATSOC', 'NATMASS', 'NATPARK', 
                       'NATCHLD', 'NATSCI', 'NATENRGY', 'NATSPACY', 'NATENVIY', 'NATHEALY', 'NATCITYY', 'NATCRIMY', 
                       'NATDRUGY', 'NATEDUCY', 'NATRACEY', 'NATARMSY', 'NATAIDY', 'NATFAREY', 'EQWLTH', 'SPKATH', 
                       'COLATH', 'LIBATH', 'SPKRAC', 'COLRAC', 'LIBRAC', 'SPKCOM', 'COLCOM', 'LIBCOM', 'SPKMIL', 
                       'COLMIL', 'LIBMIL', 'SPKHOMO', 'COLHOMO', 'LIBHOMO', 'SPKMSLM', 'COLMSLM', 'LIBMSLM', 'CAPPUN', 
                       'GUNLAW', 'COURTS', 'GRASS', 'POSTLIFE', 'PRAYER', 'AFFRMACT', 'WRKWAYUP', 'HELPFUL', 
                       'FAIR', 'TRUST', 'CONFINAN', 'CONBUS', 'CONCLERG', 'CONEDUC', 'CONFED', 'CONLABOR', 'CONPRESS', 
                       'CONMEDIC', 'CONTV', 'CONJUDGE', 'CONSCI', 'CONLEGIS', 'CONARMY', 'OBEY', 'POPULAR', 'THNKSELF', 
                       'WORKHARD', 'HELPOTH', 'GETAHEAD', 'FEPOL', 'ABDEFECT', 'ABNOMORE', 'ABHLTH', 'ABPOOR', 'ABRAPE', 
                       'ABSINGLE', 'ABANY', 'SEXEDUC', 'DIVLAW', 'PREMARSX', 'TEENSEX', 'XMARSEX', 'HOMOSEX', 'PORNLAW', 
                       'SPANKING', 'LETDIE1', 'SUICIDE1', 'SUICIDE2', 'POLHITOK', 'POLABUSE', 'POLMURDR', 'POLESCAP', 
                       'POLATTAK', 'FECHLD', 'FEPRESCH', 'FEFAM', 'RACDIF1', 'RACDIF2', 'RACDIF3', 'RACDIF4', 
                       'HELPPOOR', 'HELPNOT', 'HELPBLK', 'MARHOMO']

    complicated_questions = ['VOTE68', 'PRES68', 'IF68WHO', 'VOTE72', 'PRES72', 'IF72WHO', 'VOTE76', 'PRES76', 'IF76WHO', 
                         'VOTE80', 'PRES80', 'IF80WHO', 'VOTE84', 'PRES84', 'IF84WHO', 'VOTE88', 'PRES88', 'IF88WHO', 
                         'VOTE92', 'PRES92', 'IF92WHO', 'VOTE96', 'PRES96', 'IF96WHO', 'VOTE00', 'PRES00', 'IF00WHO', 
                         'VOTE04', 'PRES04', 'IF04WHO', 'VOTE08', 'PRES08', 'IF08WHO', 'VOTE12', 'PRES12', 'IF12WHO', 
                         'VOTE16', 'PRES16', 'IF16WHO', 'VOTE20', 'PRES20', 'IF20WHO', 'RELIG', 'ATTEND', 'RACOPEN', 
                         'NEWS', 'TVHOURS', 'RELITEN']
    """

    # Columns that should not be transformed
    EXCLUDE_COLS = ["YEAR", "BALLOT", "ID"]
    
    #--------------------------------------------------------------------------
    # Symmetric variables Mappings and Categories
    #--------------------------------------------------------------------------
    
    # Variables mapped from {0, 1, 2, 3, 4, 5, 6} -> {-1, -0.66666667, -0.33333333, 0, 0.33333333, 0.66666667, 1}
    ordinal_0_to_6 = {0: -1, 1: -0.66666667, 2: -0.33333333, 3: 0, 4: 0.33333333, 5: 0.66666667, 6: 1}
    VARS_B = ["PARTYID"]
    #VARS_B = VARS_B + ["SEXFREQ"]

    # Variables mapped from {1, 2, 3, 4, 5, 6 ,7} -> {-1, -0.66666667, -0.33333333, 0, 0.33333333, 0.66666667, 1}
    ordinal_1_to_7 = {1: -1, 2: -0.66666667, 3: -0.33333333, 4: 0, 5: 0.33333333, 6: 0.66666667, 7: 1}
    VARS_C = ["POLVIEWS", "EQWLTH"]

    # Variables mapped from {1, 2, 3} -> {-1, 0, 1}
    ordinal_1_to_3 = {1: -1, 2: 0, 3: 1}
    VARS_D = ['NATSPAC', 'NATENVIR', 'NATHEAL', 'NATCITY', 'NATCRIME', 'NATDRUG', 'NATEDUC', 'NATRACE', 
              'NATARMS', 'NATAID', 'NATFARE', 'NATROAD', 'NATSOC', 'NATMASS', 'NATPARK', 'NATCHLD', 
              'NATSCI', 'NATENRGY', 'NATSPACY', 'NATENVIY', 'NATHEALY', 'NATCITYY', 'NATCRIMY', 
              'NATDRUGY', 'NATEDUCY', 'NATRACEY', 'NATARMSY', 'NATAIDY', 'NATFAREY', 'PORNLAW']

    # Variables mapped from {1, 'D', 2} -> {1, 0, -1}
    ordinal_1_to_2_with_98 = {1: 1, 'D': 0, 2: -1}
    VARS_E = ['SPKATH', 'LIBATH', 'SPKRAC', 'LIBRAC', 'SPKCOM', 'LIBCOM', 'SPKMIL', 'LIBMIL', 
              'SPKHOMO', 'LIBHOMO', 'SPKMSLM', 'LIBMSLM', 'CAPPUN', 'GUNLAW', 'GRASS', 'POSTLIFE', 
              'PRAYER', 'FEPOL', 'ABDEFECT', 'ABNOMORE', 'ABHLTH', 'ABPOOR', 'ABRAPE', 'ABSINGLE',
              'ABANY', 'LETDIE1', 'SUICIDE1', 'SUICIDE2', 'POLHITOK', 'POLABUSE', 'POLMURDR', 
              'POLESCAP', 'POLATTAK', 'RACDIF1', 'RACDIF2', 'RACDIF3', 'RACDIF4']

    # Variables mapped from {1, 2, 3} -> {1, -1, 0}
    ordinal_1_to_3_alt_order = {1: 1, 2: -1, 3: 0}
    VARS_F = ['COURTS', 'HELPFUL', 'FAIR', 'TRUST', 'SEXEDUC', 'DIVLAW']

    # Variables mapped from {1, 2, 'D', 3, 4} -> {1, 0.5, 0, -0.5, -1}
    ordinal_1_to_4_with_98 = {1: 1, 2: 0.5, 'D': 0, 3: -0.5, 4: -1}
    VARS_G = ['AFFRMACT', 'PREMARSX', 'TEENSEX', 'XMARSEX', 'HOMOSEX', 'SPANKING', 'FECHLD','FEPRESCH', 'FEFAM']

    # Variables mapped from {1, 2, 3, 4, 5} -> {1, 0.5, 0, -0.5, -1}
    ordinal_1_to_5 = {1: 1, 2: 0.5, 3: 0, 4: -0.5, 5: -1}
    VARS_H = ['WRKWAYUP', 'OBEY', 'POPULAR', 'THNKSELF', 'WORKHARD', 'HELPOTH', 'HELPPOOR', 'HELPNOT', 'HELPBLK', 'MARHOMO']

    # Variables mapped from {1, 2, 3} -> {1, 0, -1}
    ordinal_1_to_3 = {1: 1, 2: 0, 3: -1}
    VARS_I = ['CONFINAN', 'CONBUS', 'CONCLERG', 'CONEDUC', 'CONFED', 'CONLABOR', 'CONPRESS', 
              'CONMEDIC', 'CONTV', 'CONJUDGE', 'CONSCI', 'CONLEGIS', 'CONARMY', 'GETAHEAD']

    # Variables mapped from {4, 'D', 5} -> {1, 0, -1}
    ordinal_4_to_5_with_98 = {4: 1, 'D': 0, 5: -1}
    VARS_J = ['COLATH', 'COLRAC', 'COLCOM', 'COLMIL', 'COLMSLM', 'COLHOMO']

    #--------------------------------------------------------------------------
    # Special case variables mappings and categories
    #--------------------------------------------------------------------------
    """
    special_questions = ['VOTE68', 'PRES68', 'IF68WHO', 'VOTE72', 'PRES72', 'IF72WHO', 'VOTE76', 'PRES76', 'IF76WHO', 
                         'VOTE80', 'PRES80', 'IF80WHO', 'VOTE84', 'PRES84', 'IF84WHO', 'VOTE88', 'PRES88', 'IF88WHO', 
                         'VOTE92', 'PRES92', 'IF92WHO', 'VOTE96', 'PRES96', 'IF96WHO', 'VOTE00', 'PRES00', 'IF00WHO', 
                         'VOTE04', 'PRES04', 'IF04WHO', 'VOTE08', 'PRES08', 'IF08WHO', 'VOTE12', 'PRES12', 'IF12WHO', 
                         'VOTE16', 'PRES16', 'IF16WHO', 'VOTE20', 'PRES20', 'IF20WHO', 'RELIG', 'ATTEND', 'RACOPEN', 
                         'NEWS', 'TVHOURS', 'RELITEN']
    """

    # Plan is: make the following derived variables: PRESLAST_DEMREP, PRESLAST_NONCONFORM, WOULDVOTELAST_DEMREP, WOULDVOTELAST_NONCONFORM, DIDVOTELAST
    # and symmetrise all the other variables via various coping strategies.

    # In effect -- all variables will be made symmetric.

    preslast_demrep_map = {1: -1, 2: 1}
    preslast_nonconform_3_options_map = {1: -1, 2: -1, 3: 1, 4: 1, 5: -1}           # Used for 1968, 1980, 1992, 1996, 2000, 2004.
    preslast_nonconform_regular_map = {1: -1, 2: -1, 3: 1, 4: -1}                   
    pres_vars = ['PRES68', 'PRES72', 'PRES76', 'PRES80', 'PRES84', 'PRES88', 'PRES92', 'PRES96', 'PRES00', 'PRES04', 'PRES08', 'PRES12', 'PRES16', 'PRES20']
    derived_preslast_vars = ['PRESLAST_DEMREP', 'PRESLAST_NONCONFORM']


    wouldvotelast_demrep_map = {1: -1, 2: 1}
    wouldvotelast_nonconform_3_options_map = {1: -1, 2: -1, 3: 1, 4: 1, 5: -1}      # Used for 1968, 1980, 1992, 1996, 2000, 2004.
    wouldvotelast_nonconform_regular_map = {1: -1, 2: -1, 3: 1, 4: -1}
    if_vars = ['IF68WHO', 'IF72WHO', 'IF76WHO', 'IF80WHO', 'IF84WHO', 'IF88WHO', 'IF92WHO', 'IF96WHO', 'IF00WHO', 'IF04WHO', 'IF08WHO', 'IF12WHO', 'IF16WHO', 'IF20WHO']
    derived_wouldvotelast_vars = ['WOULDVOTELAST_DEMREP', 'WOULDVOTELAST_NONCONFORM']

    didvotelast_map = {1: 1, 2: -1}
    vote_vars = ['VOTE68', 'VOTE72', 'VOTE76', 'VOTE80', 'VOTE84', 'VOTE88', 'VOTE92', 'VOTE96', 'VOTE00', 'VOTE04', 'VOTE08', 'VOTE12', 'VOTE16', 'VOTE20']
    derived_didvotelast_vars = ['DIDVOTELAST']

    # Religion maps (1 is "holds religion", -1 is "does not hold religion")
    relig_map_Protestant                    = {1: 1, 2: -1, 3: -1, 4: -1, 5: -1, 6: -1, 7: -1, 8: -1, 9: -1, 10: -1, 11: -1, 12: -1, 13: -1}
    relig_map_Catholic                      = {1: -1, 2: 1, 3: -1, 4: -1, 5: -1, 6: -1, 7: -1, 8: -1, 9: -1, 10: -1, 11: -1, 12: -1, 13: -1}
    relig_map_Jewish                        = {1: -1, 2: -1, 3: 1, 4: -1, 5: -1, 6: -1, 7: -1, 8: -1, 9: -1, 10: -1, 11: -1, 12: -1, 13: -1}
    relig_map_None                          = {1: -1, 2: -1, 3: -1, 4: 1, 5: -1, 6: -1, 7: -1, 8: -1, 9: -1, 10: -1, 11: -1, 12: -1, 13: -1}
    relig_map_Other                         = {1: -1, 2: -1, 3: -1, 4: -1, 5: 1, 6: -1, 7: -1, 8: -1, 9: -1, 10: -1, 11: -1, 12: -1, 13: -1}
    relig_map_Buddhism                      = {1: -1, 2: -1, 3: -1, 4: -1, 5: -1, 6: 1, 7: -1, 8: -1, 9: -1, 10: -1, 11: -1, 12: -1, 13: -1}
    relig_map_Hinduism                      = {1: -1, 2: -1, 3: -1, 4: -1, 5: -1, 6: -1, 7: 1, 8: -1, 9: -1, 10: -1, 11: -1, 12: -1, 13: -1}
    relig_map_Other_eastern_religions       = {1: -1, 2: -1, 3: -1, 4: -1, 5: -1, 6: -1, 7: -1, 8: 1, 9: -1, 10: -1, 11: -1, 12: -1, 13: -1}
    relig_map_Muslim                        = {1: -1, 2: -1, 3: -1, 4: -1, 5: -1, 6: -1, 7: -1, 8: -1, 9: 1, 10: -1, 11: -1, 12: -1, 13: -1}
    relig_map_Orthodox_christian	        = {1: -1, 2: -1, 3: -1, 4: -1, 5: -1, 6: -1, 7: -1, 8: -1, 9: -1, 10: 1, 11: -1, 12: -1, 13: -1}
    relig_map_Christian                     = {1: -1, 2: -1, 3: -1, 4: -1, 5: -1, 6: -1, 7: -1, 8: -1, 9: -1, 10: -1, 11: 1, 12: -1, 13: -1}
    relig_map_Native_american               = {1: -1, 2: -1, 3: -1, 4: -1, 5: -1, 6: -1, 7: -1, 8: -1, 9: -1, 10: -1, 11: -1, 12: 1, 13: -1}
    relig_map_Inter_nondenominational	    = {1: -1, 2: -1, 3: -1, 4: -1, 5: -1, 6: -1, 7: -1, 8: -1, 9: -1, 10: -1, 11: -1, 12: -1, 13: 1}
    relig_vars = ['RELIG']

    # RELITEN map: RELITEN will be mapped to a [-1, 0, 1] scale (Not very strong and somewhat strong will be collapsed into 0). The intepretation is that "I believe X things related to religion vs I do not believe in X things related to religion".
    reliten_map = {1: -1, 2: 0, 3: 0, 4: -1}
    reliten_vars = ['RELITEN']

    # RACOPEN will be mapped to a [-1, 0, 1] scale (First law is 1, second law is -1, and third law is 0). 
    racopen_map = {1: 1, 2: -1, 3: 0}
    racopen_vars = ['RACOPEN']

    # ATTEND, NEWS, and TVHOURS will face the median method
    # TO BE COMPLETED AT A LATER DATE!!   
    median_vars = ['ATTEND', 'NEWS', 'TVHOURS']

    #--------------------------------------------------------------------------
    # Helper Properties
    #--------------------------------------------------------------------------

    @property
    def all_questions(self) -> List[str]:
        """Get all question variables that should be included."""
        all_vars = (self.VARS_B + self.VARS_C + self.VARS_D + self.VARS_E + 
                   self.VARS_F + self.VARS_G + self.VARS_H + self.VARS_I +
                   self.VARS_J)
        return list(set(all_vars))  # Remove any duplicates
    
    @property
    def all_mappings(self) -> Dict[str, Dict]:
        """Get mapping dictionary for all variables."""
        mappings = {}
        
        for var in self.VARS_B:
            mappings[var] = self.ordinal_0_to_6
        for var in self.VARS_C:
            mappings[var] = self.ordinal_1_to_7
        for var in self.VARS_D:
            mappings[var] = self.ordinal_1_to_3
        for var in self.VARS_E:
            mappings[var] = self.ordinal_1_to_2_with_98
        for var in self.VARS_F:
            mappings[var] = self.ordinal_1_to_3_alt_order
        for var in self.VARS_G:
            mappings[var] = self.ordinal_1_to_4_with_98
        for var in self.VARS_H:
            mappings[var] = self.ordinal_1_to_5
        for var in self.VARS_I:
            mappings[var] = self.ordinal_1_to_3
        for var in self.VARS_J:
            mappings[var] = self.ordinal_4_to_5_with_98
            
        return mappings

#------------------------------------------------------------------------------
# Transformation Functions
#------------------------------------------------------------------------------

def transform_column(series: pd.Series, mapping: Dict) -> pd.Series:
    """
    Transform a single column using the provided mapping.
    
    Args:
        series: Input pandas series
        mapping: Dictionary mapping original values to new values
    
    Returns:
        Transformed pandas series with unmapped values and 'I' values set to NaN
    """
    # Replace 'I', 'N', and 'Y' with NaN before mapping
    series = series.replace(['I', 'N', 'Y'], np.nan)
    return series.map(mapping)

def transform_regular() -> pd.DataFrame:
    """
    Transform GSS data using regular normalization (preserving natural zero points).
    
    Returns:
        Transformed DataFrame
    """
    # Load raw data
    df, _ = import_dataset()
    
    # Initialize config
    config = DataConfig()
    
    # Get columns to keep including special case variables
    columns_to_keep = (
        config.EXCLUDE_COLS 
        + config.all_questions 
        + config.pres_vars 
        + config.if_vars 
        + config.vote_vars 
        + config.relig_vars 
        + config.reliten_vars 
        + config.racopen_vars 
        + config.median_vars
    )
    columns_to_keep = list(set(columns_to_keep))  # Remove duplicates
    
    # Filter columns
    df_filtered = df[columns_to_keep].copy()
    
    # Transform each column according to its mapping
    mappings = config.all_mappings
    
    for col, mapping in mappings.items():
        if col in df_filtered.columns:
            df_filtered[col] = transform_column(df_filtered[col], mapping)
    
    # Process PRES variables to create PRESLAST_DEMREP and PRESLAST_NONCONFORM
    #df_filtered['PRESLAST_DEMREP'] = np.nan
    #df_filtered['PRESLAST_NONCONFORM'] = np.nan

    # Test optimisations
    preslast_cols = pd.DataFrame({
        'PRESLAST_DEMREP': np.nan,
        'PRESLAST_NONCONFORM': np.nan
    }, index=df_filtered.index)
    df_filtered = pd.concat([df_filtered, preslast_cols], axis=1)

    for pres_var in config.pres_vars:
        if pres_var not in df_filtered.columns:
            continue
        year_str = pres_var.replace('PRES', '')
        election_year = 1900 + int(year_str) if int(year_str) >= 68 else 2000 + int(year_str)
        if election_year in [1968, 1980, 1992, 1996, 2000, 2004]:
            nonconform_map = config.preslast_nonconform_3_options_map
        else:
            nonconform_map = config.preslast_nonconform_regular_map
        
        demrep_col = df_filtered[pres_var].map(config.preslast_demrep_map)
        nonconform_col = df_filtered[pres_var].map(nonconform_map)
        
        mask = demrep_col.notna()
        df_filtered.loc[mask, 'PRESLAST_DEMREP'] = demrep_col[mask]
        mask = nonconform_col.notna()
        df_filtered.loc[mask, 'PRESLAST_NONCONFORM'] = nonconform_col[mask]
    
    # Process IF variables to create WOULDVOTELAST_DEMREP and WOULDVOTELAST_NONCONFORM
    #df_filtered['WOULDVOTELAST_DEMREP'] = np.nan
    #df_filtered['WOULDVOTELAST_NONCONFORM'] = np.nan
    
    # Test optimisations
    wouldvote_cols = pd.DataFrame({
        'WOULDVOTELAST_DEMREP': np.nan,
        'WOULDVOTELAST_NONCONFORM': np.nan
    }, index=df_filtered.index)
    df_filtered = pd.concat([df_filtered, wouldvote_cols], axis=1)
    
    for if_var in config.if_vars:
        if if_var not in df_filtered.columns:
            continue
        year_str = if_var.replace('IF', '').replace('WHO', '')
        election_year = 1900 + int(year_str) if int(year_str) >= 68 else 2000 + int(year_str)
        if election_year in [1968, 1980, 1992, 1996, 2000, 2004]:
            nonconform_map = config.wouldvotelast_nonconform_3_options_map
        else:
            nonconform_map = config.wouldvotelast_nonconform_regular_map
        
        demrep_col = df_filtered[if_var].map(config.wouldvotelast_demrep_map)
        nonconform_col = df_filtered[if_var].map(nonconform_map)
        
        mask = demrep_col.notna()
        df_filtered.loc[mask, 'WOULDVOTELAST_DEMREP'] = demrep_col[mask]
        mask = nonconform_col.notna()
        df_filtered.loc[mask, 'WOULDVOTELAST_NONCONFORM'] = nonconform_col[mask]
    
    # Process VOTE variables to create DIDVOTELAST
    #df_filtered['DIDVOTELAST'] = np.nan
    
    # Test optimisations
    didvote_col = pd.DataFrame({'DIDVOTELAST': np.nan}, index=df_filtered.index)
    df_filtered = pd.concat([df_filtered, didvote_col], axis=1)
   
    for vote_var in config.vote_vars:
        if vote_var not in df_filtered.columns:
            continue
        vote_col = df_filtered[vote_var].map(config.didvotelast_map)
        mask = vote_col.notna()
        df_filtered.loc[mask, 'DIDVOTELAST'] = vote_col[mask]
    
    # Process RELIG into binary variables
    religion_derivations = [
        ('Protestant', config.relig_map_Protestant),
        ('Catholic', config.relig_map_Catholic),
        ('Jewish', config.relig_map_Jewish),
        ('None', config.relig_map_None),
        ('Other', config.relig_map_Other),
        ('Buddhism', config.relig_map_Buddhism),
        ('Hinduism', config.relig_map_Hinduism),
        ('Other_eastern_religions', config.relig_map_Other_eastern_religions),
        ('Muslim', config.relig_map_Muslim),
        ('Orthodox_christian', config.relig_map_Orthodox_christian),
        ('Christian', config.relig_map_Christian),
        ('Native_american', config.relig_map_Native_american),
        ('Inter_nondenominational', config.relig_map_Inter_nondenominational)
    ]

    #for suffix, mapping in religion_derivations:
    #    new_col = f'RELIG_{suffix}'
    #    df_filtered[new_col] = df_filtered['RELIG'].map(mapping)
    
    # Test optimisations
    religion_data = {}
    for suffix, mapping in religion_derivations:
        new_col = f'RELIG_{suffix}'
        religion_data[new_col] = df_filtered['RELIG'].map(mapping)
    religion_df = pd.DataFrame(religion_data)
    df_filtered = pd.concat([df_filtered, religion_df], axis=1)
    df_filtered = df_filtered.copy() # Defrag

    # Process RELITEN
    df_filtered['RELITEN'] = df_filtered['RELITEN'].map(config.reliten_map)
    
    # Process RACOPEN
    df_filtered['RACOPEN'] = df_filtered['RACOPEN'].map(config.racopen_map)
    
    
    # Drop original special case columns
    columns_to_drop = (
        config.pres_vars + config.if_vars + config.vote_vars 
        + config.relig_vars + config.reliten_vars + config.racopen_vars 
        + config.median_vars
    )
    df_filtered.drop(columns=columns_to_drop, errors='ignore', inplace=True)
    
    return df_filtered

#------------------------------------------------------------------------------
# Main Function
#------------------------------------------------------------------------------

def clean_datasets() -> pd.DataFrame:
    """
    Clean and transform GSS data.
    
    Returns:
        Regular version of the transformed dataset
    """
    # Initialize config
    config = DataConfig()
    
    # Create regular version
    df_clean = transform_regular()
    
    # Cache the cleaned datasets
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.append(project_root)
    cache_dir = os.path.join(project_root, 'datasets', 'cached_data')
    os.makedirs(cache_dir, exist_ok=True)
    cache_file_1 = os.path.join(cache_dir, 'df_clean.pkl')
    
    # Save the dataset as a pickle file
    df_clean.to_pickle(cache_file_1)

    return df_clean 

#------------------------------------------------------------------------------
# For Testing (if run as script)
#------------------------------------------------------------------------------

if __name__ == '__main__':
    df_clean = clean_datasets()
    
    
    # Add the project root directory to the Python path
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.append(project_root)
    
    
    # Cache the cleaned datasets
    cache_dir = os.path.join(project_root, 'datasets', 'cached_data')
    os.makedirs(cache_dir, exist_ok=True)
    
    
    
    # Cache each version separately
    cache_file_1 = os.path.join(cache_dir, 'df_clean.pkl')
    
    # Save the datasets
    df_clean.to_pickle(cache_file_1)

    # Print list of all variables
    # print(list(df_clean.columns))
