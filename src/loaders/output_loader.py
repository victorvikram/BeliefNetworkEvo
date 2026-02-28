"""
This file has functions to help load outputs from other parts of the code
Often it is loading partial correlation networks as saved in outputs
"""

import os 
import numpy as np
import csv

def get_sorted_adj_mat_and_var_list(path):
    """
    This function takes the file path of one of the output folders and returns the sorted adjacency matrix and variable list,
    where both are sorted alphabetically according to the variables
    """
    ind_list, var_list = csv_to_sorted_list(os.path.join(path, "variables_list.csv"))
    l = len(var_list)
    adj_mat = np.genfromtxt(os.path.join(path, "correlation_matrix_partial.csv"), delimiter=',')
    adj_mat = adj_mat[ind_list, :][:, ind_list] - np.identity(l)
    
    return adj_mat, var_list

def csv_to_sorted_list(path):
    """
    This function reads a csv file and returns a list where each row of the csv is stored as a string. E.g.

    hello, goodbye, tomorrow
    1, 2, 3
    "past", "present", "future"

    would be stored as ["hello,goodbye,tomorrow", "1,2,3", "past,present,future"]

    and CSVs with one entry per line 

    hello
    goodbye
    tomorrow

    would be stored as ["hello", "goodbye", "tomorrow"]
    """
    # Initialize an empty list to store the strings
    lines = []

    # Open the CSV file and read it
    with open(path, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            # Join the row into a single string separated by commas
            line = ','.join(row)
            lines.append(line)
    
    indexed_list = list(enumerate(lines))
    sorted_indexed_list = sorted(indexed_list, key=lambda x: x[1]) # sort based on the value
    index_list = [ind for ind, val in sorted_indexed_list]
    value_list = [val for ind, val in sorted_indexed_list]

    return index_list, value_list