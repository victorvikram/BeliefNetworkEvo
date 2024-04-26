import pandas as pd
import pyreadstat as prs

def load_gss_sas(filename, vars_to_load, metadataonly=False):
    """
    loads the gss from a sas file
    """
    df, meta = prs.read_sas7bdat(filename, usecols=vars_to_load, metadataonly=metadataonly)

    return df, meta


def make_variable_df(data_file, variable_list):
    """
    makes a dataframe with the variable names and their descriptions
    """
    gss_df, meta = load_gss_sas(data_file, variable_list, metadataonly=True)
    variable_dict = meta.column_names_to_labels
    variable_df = pd.DataFrame(list(variable_dict.items()), columns=['var_name', 'var_desc'])
    return variable_df


if __name__ == "__main__":
    gss_file = "C:/Users/vicvi/big-datasets/social_values/GSS_sas/gss7222_r3.sas7bdat"
    variable_list = ["YEAR", "BALLOT"]

    df = make_variable_df(gss_file, variable_list)