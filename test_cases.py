import numpy as np

from claspy.data_loader import load_tssb_dataset
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
    return c, A, b, "max", 9.0

def case2():
    c = np.array([7, 6, 0, 0]).astype(np.float64)
    A = np.array([
        [2, 4, 1, 0],
        [3, 2, 0, 1]
    ]).astype(np.float64)
    b = np.array([16, 12])
    return c, A, b, "max", 32.0

def case3():
    c = np.array([40, 30, 0, 0]).astype(np.float64)
    A = np.array([
        [1, 1, 1, 0],
        [2, 1, 0, 1]
    ]).astype(np.float64)
    b = np.array([12, 16])
    return c, A, b, "max", 400.0

def case4():
    c = np.array([3, 2, 0, 0, 0, 0]).astype(np.float64)
    A = np.array([
        [1, 1, 1, 0, 0, 0],
        [3, 1, 0, 1, 0, 0],
        [1, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 1]
    ]).astype(np.float64)
    b = np.array([9, 18, 7, 6])
    return c, A, b, "max", 22.5

def case5():
    c = np.array([6, 9, 0, 0,]).astype(np.float64)
    A = np.array([
        [2, 3, 1, 0],
        [1, 1, 0, 1]
    ]).astype(np.float64)
    b = np.array([12, 5])
    return c, A, b, "max", 36.0

def case6():
    c = np.array([6, 9, 0, 0,]).astype(np.float64)
    A = np.array([
        [2, 3, 1, 0],
        [1, 1, 0, 1]
    ]).astype(np.float64)
    b = np.array([12, 5])
    return c, A, b, "min", 0.0


def case7():
    c = np.array([7, 3, 0]).astype(np.float64)
    A = np.array([
        [2, -1, 0],
        [1, 1, 1]
    ]).astype(np.float64)
    b = np.array([5, 9])
    return c, A, b, "max", 45.669

def load_dataset(dataset):
    tssb = load_tssb_dataset([dataset])
    idx, (dataset, window_size, cps, ts) = list(tssb.iterrows())[0]

    window_size = 3
    offset = 1

    window1 = np.abs(ts[:window_size])               # is supplier, sum is ~4.1
    window2 = np.abs(ts[offset:window_size+offset])  # is consumer, sum is ~3.1

    print(window1)
    print(f"\t {window2}")

    c = np.zeros((window_size * window_size) + window_size, dtype=np.float64)
    for i in range(window_size):
        for j in range(window_size):
            c[i * window_size + j] = np.abs(i - (offset + j))
    print(c)

    b_eq = window2
    A_eq = np.zeros((window_size, window_size * window_size + window_size), dtype=np.float64)
    for j in range(window_size):
        for i in range(window_size):
            A_eq[i, j * window_size + i] = 1.0

    b_ub = window1
    A_ub = np.zeros((window_size, window_size * window_size + window_size), dtype=np.float64)
    for i in range(window_size):
        for j in range(window_size):
            A_ub[i, i * window_size + j] = 1.0
        A_ub[i, window_size * window_size + i] = 1.0

    b = np.hstack([b_eq, b_ub])
    A = np.vstack([A_eq, A_ub])

    return c, A, b, "min", -5.116696


def main():
    c, A, b, optimize, z_true = load_dataset("Adiac")
    x, z_pred = simplex(c, A, b, optimize)
    z_pred = np.round(z_pred, 6)
    print(f"Solution x: {x}\nCosts: {c}\nObjective value z: {z_pred} \nDiff of pred and true z = {z_pred - z_true}.\n")
    
if __name__ == "__main__":
    import shutil
    width = shutil.get_terminal_size().columns
    np.set_printoptions(linewidth=width)

    main()
