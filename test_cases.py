import numpy as np
from numba import njit

from claspy.data_loader import load_tssb_dataset
from simplex import simplex

import cProfile
import pstats

import matplotlib.pyplot as plt

@njit(fastmath=True, cache=True)
def sliding_csum_abs(time_series, window_size):
    """
    Computes the sliding cumulative sum of absolute values of each time series subsequence
    with a specified window size.

    Parameters
    ----------
    time_series: numpy.ndarray
        A 1-dimensional numpy array containing the time series data.
    window_size: int
        The size of the sliding window.

    Returns
    -------
    time_series : numpy.ndarray
        A 1-dimensional numpy array containing the time series data.
    csum_abs : numpy.ndarray
        A 1-dimensional numpy array containing the cumulative sum of absolute values of each
        time series subsequence.
    """
    csum = np.concatenate((np.zeros(1, dtype=np.float64), np.cumsum(np.abs(time_series))))
    return csum[window_size:] - csum[:-window_size]


@njit(cache=True)
def cost_vector(window_size):
    c = np.zeros((window_size * window_size) + window_size, dtype=np.float64)
    for i in range(window_size):
        for j in range(window_size):
            c[i * window_size + j] = np.abs(i - (window_size + j))
    return c


@njit(cache=True)
def constraint_matrix(window_size):
    A_eq = np.zeros((window_size, window_size * window_size + window_size), dtype=np.float64)
    for j in range(window_size):
        for i in range(window_size):
            A_eq[i, j * window_size + i] = 1.0

    A_ub = np.zeros((window_size, window_size * window_size + window_size), dtype=np.float64)
    for i in range(window_size):
        for j in range(window_size):
            A_ub[i, i * window_size + j] = 1.0
        A_ub[i, window_size * window_size + i] = 1.0
    
    return np.vstack((A_eq, A_ub))


def load_dataset(dataset):
    tssb = load_tssb_dataset([dataset])
    idx, (dataset, window_size, cps, ts) = list(tssb.iterrows())[0]
    return ts, window_size


def lp_params(idx, jdx, window_size, time_series, csum_abs):
    if csum_abs[idx] >= csum_abs[jdx]:
        supplier_idx = idx  # s1
        consumer_idx = jdx  # s2
    else:
        supplier_idx = jdx  # s1
        consumer_idx = idx  # s2

    b_eq = np.abs(time_series[consumer_idx : consumer_idx + window_size])
    b_ub = np.abs(time_series[supplier_idx : supplier_idx + window_size])
    b = np.concatenate((b_eq, b_ub))

    return b


def main():
    idx = 1000
    dataset = "Beef"
    
    time_series, window_size = load_dataset(dataset)
    csum_abs = sliding_csum_abs(time_series, window_size)
    
    dist = np.zeros(csum_abs.shape[0])
    A = constraint_matrix(window_size)  # the same for each run
    c = cost_vector(window_size)

    for jdx in range(len(csum_abs)):
        if idx == jdx:
            continue
        b = lp_params(idx, jdx, window_size, time_series, csum_abs)
        z = simplex(c, A, b)
        dist[jdx] = z
    
    dist = np.round(dist, 4)
    
    # plt.plot(dist)
    # plt.title(f"{dataset}")
    # plt.xlabel("Subsequences si")
    # plt.ylabel(f"Distance to s{idx}")
    # plt.savefig("./plots/dist.png")
    

if __name__ == "__main__":
    with cProfile.Profile() as pr:
        main()

    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.CUMULATIVE).print_stats(10)

