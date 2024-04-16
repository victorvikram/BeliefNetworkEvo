import pandas as pd
import pyreadstat as prs

def load_gss(filename, vars_to_load, filetype="sav"):
    if filetype == "sav":
        df, meta = prs.read_sav(filename, usecols=vars_to_load)

    return df, meta



def load_anes(filename, vars_to_load):
    return pd.read_csv(filename, use_cols=columns_to_load)

