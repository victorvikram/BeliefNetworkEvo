""" BORING!!! """
import numpy as np
import pyreadstat as prs
from clean_data_1 import transform_dataframe_1
from clean_data_2 import transform_dataframe_2
from make_belief_network import make_belief_network
from calc_stress import stress


def belief_to_stress_histogram(df, variables, start_year, duration, plot=True):
    """ Takes in a dataframe cleaned via transform_dataframe_1 and returns a histogram of the stress values of the belief vectors """
    """ It can also return the stress vector and the variables list """

    ### Making a belief network ###

    """ Now we specify a time-frame and a set of variables, and make a belief network """

    timeframe = list(range(start_year, start_year+duration))

    _, variables_list, correlation_matrix_partial = make_belief_network(df, variables, timeframe, method="spearman", is_partial=True, threshold=0.0, sample_threshold=0, regularisation=0.2)

    """ In order to calculate the stress of beleif vectors, we need to belief vectors """
    """ We first clear the dataset again to implement the median solution """

    df_b_vecs, _ = transform_dataframe_2(df, timeframe)

    """ We can then cut the data down to only the variables in the belief network """

    df_b_vecs = df_b_vecs[variables_list]

    """ And then get an array of the belief vectors """

    belief_vectors = df_b_vecs.to_numpy()

    """ Finally we want to set the NaNs to zero, and normalise the vectors such that they span -1 and 1 """
    belief_vectors[np.isnan(belief_vectors)] = 0
    belief_vectors = 2*(belief_vectors - np.min(belief_vectors, axis=0))/(np.max(belief_vectors, axis=0) - np.min(belief_vectors, axis=0)) - 1

    """ Okay, and now we calculate the stress of the belief vectors """

    correlation_matrix_noDiag = correlation_matrix_partial - np.eye(correlation_matrix_partial.shape[0])
    stress_vec = []
    stress_vec= [stress(belief_vectors[i,:], correlation_matrix_noDiag) for i in range(belief_vectors.shape[0])]


    if plot:
        import matplotlib.pyplot as plt
        plt.hist(stress_vec, bins=100)
        plt.xlabel("Stress")
        plt.ylabel("Frequency")
        plt.show()

    return stress_vec, variables_list
