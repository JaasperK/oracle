import numpy as np
from numba import njit

@njit("float64[:,:](float64[:], float64[:,:], float64[:])", cache=True)
def initial_tableau(c, A, b):
    t = np.vstack((-np.ascontiguousarray(c).reshape(1, c.size), A))  # numba hack
    b = np.concatenate((np.zeros(1), b))
    return np.column_stack((t, b.reshape(b.size, 1)))


@njit(fastmath=True, cache=True)
def simplex(c, A, b):
    """
    Implementation of the tableau simplex algorithm. Solves linear programming
    problems in standard form, meaning all inequalities have been converted to
    equalities and c and A contain the logical/slack variables:
    ```
    max z = c @ x
    s.t. Ax = b
    where x >= 0
    ``` 

    Parameters
    ----------
    c : np.ndarray
        Coefficients of the objective function.
    A : np.ndarray
        Constraint matrix.
    b : np.ndarray
        Right-hand side of constraints.

    Returns
    -------
    z : float
        Value of the objective function.
    """
    tab = initial_tableau(c, A, b)
    tab_rows = np.arange(tab.shape[0])

    while np.any(tab[0] < 0.0):  # solution is not optimal
        pivot_col = np.argmin(tab[0, :-1])
        r1 = tab[1:, pivot_col]
        r1 = np.where(r1 <= 0, -1, r1)  # avoid division by 0
        r2 = tab[1:, -1]
        ratio = np.where(r1 > 0, r2 / r1, np.inf)
        pivot_row = np.argmin(ratio) + 1  # +1 to account for the z-row that was not included in r1 and r2

        # update tableau, probably not necessary since pivot element is always 1 in our special case
        # tab[pivot_row] = tab[pivot_row] / tab[pivot_row, pivot_col]
        
        for i in tab_rows[tab_rows != pivot_row]:
            tab[i] = tab[i] - tab[i, pivot_col] * tab[pivot_row]
    
    return tab[0][-1]
