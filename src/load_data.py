import pandas as pd

def load_gss(filename, vars_to_load):
    return pd.read_stata(filename, columns=vars_to_load)

def load_anes(filename, vars_to_load):
    return pd.read_csv(filename, use_cols=columns_to_load)

