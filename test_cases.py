import numpy as np
from simplex import simplex

def case1():
#   max   z = 4x_1 + 3x_2
#
#   s.t.  2x_1 + 3x_2 + s_1                   = 6
#        -3x_1 + 2x_2 +     + s_2             = 3
#                2x_2 +           + s_3       = 5
#         2x_1 +  x_2                   + s_4 = 4
#   x_1, x_2, s_1, s_2, s_3, s_4 >= 0.      (Non-negativity)
    c = np.array([4, 3, 0, 0, 0, 0]).astype(np.float64)
    A = np.array([
        [ 2, 3, 1, 0, 0, 0],
        [-3, 2, 0, 1, 0, 0],
        [ 0, 2, 0, 0, 1, 0],
        [ 2, 1, 0, 0, 0, 1]
    ]).astype(np.float64)
    b = np.array([6, 3, 5, 4]).astype(np.float64)    
    return c, A, b

def case2():
    c = np.array([7, 6, 0, 0]).astype(np.float64)
    A = np.array([
        [2, 4, 1, 0],
        [3, 2, 0, 1]
    ]).astype(np.float64)
    b = np.array([16, 12])
    return c, A, b

def case3():
    c = np.array([40, 30, 0, 0]).astype(np.float64)
    A = np.array([
        [1, 1, 1, 0],
        [2, 1, 0, 1]
    ]).astype(np.float64)
    b = np.array([12, 16])
    return c, A, b

def case4():
    c = np.array([3, 2, 0, 0, 0, 0]).astype(np.float64)
    A = np.array([
        [1, 1, 1, 0, 0, 0],
        [3, 1, 0, 1, 0, 0],
        [1, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 1]
    ]).astype(np.float64)
    b = np.array([9, 18, 7, 6])
    return c, A, b

def case5():
    c = np.array([6, 9, 0, 0,]).astype(np.float64)
    A = np.array([
        [2, 3, 1, 0],
        [1, 1, 0, 1]
    ]).astype(np.float64)
    b = np.array([12, 5])
    return c, A, b

def case6():
    c = np.array([-6, -9, 0, 0,]).astype(np.float64)
    A = np.array([
        [2, 3, 1, 0],
        [1, 1, 0, 1]
    ]).astype(np.float64)
    b = np.array([12, 5])
    return c, A, b

def case7():
    c = np.array([3, 4, 0, 0,]).astype(np.float64)
    A = np.array([
        [1, 1, -1, 0],
        [2, 1, 0, -1]
    ]).astype(np.float64)
    b = np.array([450, 600])
    return c, A, b


def main():
    cases = [case1, case2, case3, case4, case5, case6, case7]
    for case in cases:
        c, A, b = case()
        x, z = simplex(c, A, b)
        print(f"Solution x: {x}\nCosts: {c}\nObjective value z = {z}.\n")
    
if __name__ == "__main__":
    main()
