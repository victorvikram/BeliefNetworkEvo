import numpy as np
from scipy.optimize import linprog

def find_nonneg_sol(A, b):

    """
    `A` : [m x n matrix], `b` : [m x 1 vector] -> 
    `x` : [n x 1 vector], `s` : [m x 1 vector], `success` : bool

    finds a non-negative solution to the equation `Ax = b` if it exists. 
    If not, finds {x | Ax <= b} and ||Ax - b||_1 is minimized.

    Returns `x`, the optimal solution, 
            `s = b - Ax`, and
            `success`, whether we could find an equation with 0 (`s = 0`)
    """
    m = A.shape[0]
    n = A.shape[1]
    I = np.identity((m, m))
    A_ext = np.block([[A, I]])
    c = np.concatenate(np.zeros((n,)), np.ones((m,)))

    res = linprog(c, A_eq=A_ext, b_eq=b, bounds=(0, None))
    
    if res.status != 0:
        return res.status
    
    x = res.x[:n]
    s = res.x[n:]
    success = (s == 0).all()

<<<<<<< HEAD
    return x, s, success
=======
    return x, s, success

def bin_time_periods(time_period_df, num_bins=None):
    
        