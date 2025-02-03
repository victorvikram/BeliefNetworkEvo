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

    asymmetric_questions = ['VOTE68', 'PRES68', 'IF68WHO', 'VOTE72', 'PRES72', 'IF72WHO', 'VOTE76', 'PRES76', 'IF76WHO', 
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
    VARS_E = ['SPKATH', 'LIBATH', 'SPKRAC', 'LIBRAC', 'SPKCOM', 'LIBCOM', 
              'SPKMIL', 'LIBMIL', 'SPKHOMO', 'LIBHOMO', 'SPKMSLM', 'LIBMSLM',
              'CAPPUN', 'GUNLAW', 'GRASS', 'POSTLIFE', 'PRAYER', 'FEPOL', 'ABDEFECT', 'ABNOMORE', 'ABHLTH', 'ABPOOR', 'ABRAPE', 
              'ABSINGLE', 'ABANY', 'LETDIE1', 'SUICIDE1', 'SUICIDE2', 'POLHITOK', 'POLABUSE', 'POLMURDR', 'POLESCAP', 
              'POLATTAK', 'RACDIF1', 'RACDIF2', 'RACDIF3', 'RACDIF4']

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
    # Asymmetric variables Mappings and Categories
    #--------------------------------------------------------------------------
    """
    asymmetric_questions = ['VOTE68', 'PRES68', 'IF68WHO', 'VOTE72', 'PRES72', 'IF72WHO', 'VOTE76', 'PRES76', 'IF76WHO', 
                         'VOTE80', 'PRES80', 'IF80WHO', 'VOTE84', 'PRES84', 'IF84WHO', 'VOTE88', 'PRES88', 'IF88WHO', 
                         'VOTE92', 'PRES92', 'IF92WHO', 'VOTE96', 'PRES96', 'IF96WHO', 'VOTE00', 'PRES00', 'IF00WHO', 
                         'VOTE04', 'PRES04', 'IF04WHO', 'VOTE08', 'PRES08', 'IF08WHO', 'VOTE12', 'PRES12', 'IF12WHO', 
                         'VOTE16', 'PRES16', 'IF16WHO', 'VOTE20', 'PRES20', 'IF20WHO', 'RELIG', 'ATTEND', 'RACOPEN', 
                         'NEWS', 'TVHOURS', 'RELITEN']
    """
    
    # Variables mapped from {1, 2} -> {1, 0} for basic voting participation
    binary_1_to_2 = {1: 1, 2: 0}
    VARS_K = ['VOTE68', 'VOTE72', 'VOTE76', 'VOTE80', 'VOTE84', 'VOTE88', 'VOTE92', 
              'VOTE96', 'VOTE00', 'VOTE04', 'VOTE08', 'VOTE12', 'VOTE16', 'VOTE20']

    # Variables 
    category_pres_vars_option_A = {1: 1, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 'I': 0, 'D': 0, 'Y': 0, 'N': 0}
    category_pres_vars_option_B = {1: 0, 2: 1, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 'I': 0, 'D': 0, 'Y': 0, 'N': 0}
    category_pres_vars_option_C = {1: 1, 2: 0, 3: 1, 4: 0, 5: 0, 6: 0, 7: 0, 'I': 0, 'D': 0, 'Y': 0, 'N': 0}
    category_pres_vars_option_D = {1: 0, 2: 0, 3: 0, 4: 1, 5: 0, 6: 0, 7: 0, 'I': 0, 'D': 0, 'Y': 0, 'N': 0}
    category_pres_vars_option_E = {1: 0, 2: 0, 3: 0, 4: 0, 5: 1, 6: 0, 7: 0, 'I': 0, 'D': 0, 'Y': 0, 'N': 0}
    category_pres_vars_option_F = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 1, 7: 0, 'I': 0, 'D': 0, 'Y': 0, 'N': 0}
    category_pres_vars_option_G = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 1, 'I': 0, 'D': 0, 'Y': 0, 'N': 0}

    derived_pres_vars = ['PRES68_HUMPHREY', 'PRES68_NIXON', 'PRES68_WALLACE', 'PRES68_OTHER', 'PRES68_NOVOTE',
                         'PRES72_MCGOVERN', 'PRES72_NIXON', 'PRES72_OTHER', 'PRES72_NOVOTE',
                         'PRES76_CARTER', 'PRES76_FORD', 'PRES76_OTHER', 'PRES76_NOVOTE',
                         'PRES80_CARTER', 'PRES80_REAGAN', 'PRES80_ANDERSON', 'PRES80_OTHER', 'PRES80_NOVOTE',
                         'PRES84_MONDALE', 'PRES84_REAGAN', 'PRES84_OTHER', 'PRES84_NOVOTE',
                         'PRES88_DUKAKIS', 'PRES88_BUSH', 'PRES88_OTHER', 'PRES88_NOVOTE',
                         'PRES92_CLINTON', 'PRES92_BUSH', 'PRES92_PEROT', 'PRES92_OTHER', 'PRES92_NOVOTE',
                         'PRES96_CLINTON', 'PRES96_DOLE', 'PRES96_PEROT', 'PRES96_OTHER', 'PRES96_NOVOTE',
                         'PRES00_GORE', 'PRES00_BUSH', 'PRES00_NADER', 'PRES00_OTHER', 'PRES00_NOVOTE',
                         'PRES04_KERRY', 'PRES04_BUSH', 'PRES04_NADER', 'PRES04_OTHER', 'PRES04_NOVOTE',
                         'PRES08_OBAMA', 'PRES08_McCain', 'PRES08_OTHER', 'PRES08_NOVOTE',
                         'PRES12_OBAMA', 'PRES12_ROMNEY', 'PRES12_OTHER', 'PRES12_NOVOTE',
                         'PRES16_CLINTON', 'PRES16_TRUMP', 'PRES16_OTHER', 'PRES16_NOVOTE',
                         'PRES20_BIDEN', 'PRES20_TRUMP', 'PRES20_OTHER', 'PRES20_NOVOTE']


    category_ifwho_vars = {1: 1, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 'I': 0, 'D': 0, 'Y': 0, 'N': 0}

    derived_ifwho_vars = ['IF68WHO_HUMPHREY', 'IF68WHO_NIXON', 'IF68WHO_OTHER', 
                          'IF72WHO_MCGOVERN', 'IF72WHO_NIXON', 'IF72WHO_OTHER', 
                          'IF76WHO_CARTER', 'IF76WHO_FORD', 'IF76WHO_OTHER', 
                          'IF80WHO_CARTER', 'IF80WHO_REAGAN', 'IF80WHO_ANDERSON', 'IF80WHO_OTHER', 
                          'IF84WHO_MONDALE', 'IF84WHO_REAGAN', 'IF84WHO_OTHER', 
                          'IF88WHO_DUKAKIS', 'IF88WHO_BUSH', 'IF88WHO_OTHER', 
                          'IF92WHO_CLINTON', 'IF92WHO_BUSH', 'IF92WHO_PEROT', 'IF92WHO_OTHER', 
                          'IF96WHO_CLINTON', 'IF96WHO_DOLE', 'IF96WHO_PEROT', 'IF96WHO_OTHER', 
                          'IF00WHO_GORE', 'IF00WHO_BUSH', 'IF00WHO_NADER', 'IF00WHO_OTHER', 
                          'IF04WHO_KERRY', 'IF04WHO_BUSH', 'IF04WHO_NADER', 'IF04WHO_OTHER', 
                          'IF08WHO_OBAMA', 'IF08WHO_McCain', 'IF08WHO_OTHER', 
                          'IF12WHO_OBAMA', 'IF12WHO_ROMNEY', 'IF12WHO_OTHER', 
                          'IF16WHO_CLINTON', 'IF16WHO_TRUMP', 'IF16WHO_OTHER', 
                          'IF20WHO_BIDEN', 'IF20WHO_TRUMP', 'IF20WHO_OTHER', 'IF20WHO_NOVOTE']
    
    

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

    #--------------------------------------------------------------------------
    # Presidential Vote Variables and Mappings
    #--------------------------------------------------------------------------
    
    def create_vote_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create binary indicator variables for presidential votes.
        For each PRES* variable, creates separate columns for:
        - Democratic candidate (1 if voted for, 0 otherwise)
        - Republican candidate (1 if voted for, 0 otherwise)
        - Other candidate (1 if voted for third party/other, 0 otherwise)
        - No vote (1 if didn't vote/refused, 0 otherwise)
        """
        df = df.copy()
        
        # Get all PRES* columns
        pres_cols = [col for col in df.columns if col.startswith('PRES') and col.endswith(('68','72','76','80','84','88','92','96','00','04','08','12','16','20'))]
        
        for col in pres_cols:
            year = col[-2:]  # Get year from column name
            
            # Create indicator variables
            df[f'{col}_DEM'] = (df[col] == 1).astype(int)  # Democratic candidate
            df[f'{col}_REP'] = (df[col] == 2).astype(int)  # Republican candidate
            df[f'{col}_OTHER'] = (df[col] == 3).astype(int)  # Other/third party
            df[f'{col}_NOVOTE'] = df[col].isin([4, 5]).astype(int)  # No vote/refused
            
            # Handle missing values
            missing_mask = df[col].isin([-100, -99, -98, -70])
            new_cols = [f'{col}_{suffix}' for suffix in ['DEM', 'REP', 'OTHER', 'NOVOTE']]
            df.loc[missing_mask, new_cols] = np.nan
            
        return df

    def create_ifwho_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create binary indicator variables for hypothetical vote preference.
        Similar structure to create_vote_indicators but for IF*WHO variables.
        """
        df = df.copy()
        
        # Get all IF*WHO columns
        ifwho_cols = [col for col in df.columns if col.startswith('IF') and col.endswith('WHO')]
        
        for col in ifwho_cols:
            # Create indicator variables
            df[f'{col}_DEM'] = (df[col] == 1).astype(int)  # Democratic candidate
            df[f'{col}_REP'] = (df[col] == 2).astype(int)  # Republican candidate
            df[f'{col}_OTHER'] = (df[col] == 3).astype(int)  # Other/third party
            
            # Handle missing values
            missing_mask = df[col].isin([-100, -99, -98, -70])
            new_cols = [f'{col}_{suffix}' for suffix in ['DEM', 'REP', 'OTHER']]
            df.loc[missing_mask, new_cols] = np.nan
            
        return df

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
    
    # Get columns to keep
    columns_to_keep = config.EXCLUDE_COLS + config.all_questions
    
    # Filter columns
    df_filtered = df[columns_to_keep].copy()
    
    # Transform each column according to its mapping
    mappings = config.all_mappings
    for col, mapping in mappings.items():
        if col in df_filtered.columns:
            df_filtered[col] = transform_column(df_filtered[col], mapping)
    
    # Create binary indicators for presidential votes and hypothetical votes
    df_filtered = config.create_vote_indicators(df_filtered)
    df_filtered = config.create_ifwho_indicators(df_filtered)
    
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
    df_regular = transform_regular()
    
    return df_regular 

