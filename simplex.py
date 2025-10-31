import numpy as np

def initial_tableau(c, A, b):
    t = np.vstack([-c, A])
    b = np.hstack([0, b])
    return np.column_stack([t, b])

def simplex(c, A, b):
    """
    Solves linear programming problems in standard form, meaning all inequalities have been converted to equalities:
    ```
    max z = c @ x
    s.t. Ax = b
    where x >= 0
    ```

    c and A are assumed to include slack variables. 

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
    x : np.ndarray
        Optimal feasible solution ``x`` for which the maximum value of the
        objective function is reached.
    z : float
        Value of the objective function.
    """
    tab = initial_tableau(c, A, b)
    m = A.shape[0]
    n = A.shape[1]
    basis = np.arange(n - m, n)
    while np.any(tab[0] < 0):   # solution is not optimal
        pivot_col = np.argmin(tab[0, :-1])
        r1 = tab[1:, pivot_col]
        r1 = np.where(r1 <= 0, -1, r1)  # avoid division by 0
        r2 = tab[1:, -1]
        ratio = np.where(r1 > 0, r2 / r1, np.inf)
        pivot_row = np.argmin(ratio) + 1        # +1 to account for the z-row that was not included in r1 and r2

        # update basis
        basis[basis == pivot_row + 1] = pivot_col

        # update tableau
        tab[pivot_row] = tab[pivot_row] / tab[pivot_row, pivot_col]
        
        rows = np.arange(tab.shape[0])
        for i in rows[rows != pivot_row]:
            tab[i] = tab[i] - tab[i, pivot_col] * tab[pivot_row]

        print(tab)
    # find x using basis
    x = np.zeros(A.shape[1])
    for i in basis:
        x[i] = tab[:, i] @ tab[:, -1]
    
    return x, tab[0][-1]
