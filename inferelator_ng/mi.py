import numpy as np
import pandas as pd
import multiprocessing
from . import utils

DEFAULT_CORES = 10
DEFAULT_NUM_BINS = 10
POOL_CHUNKSIZE = 1000
CLR_DDOF = 1
DEFAULT_LOG_TYPE = np.log


class MIDriver:
    cores = DEFAULT_CORES
    bins = DEFAULT_NUM_BINS

    def __init__(self, cores=None, bins=None):

        if cores is not None:
            self.cores = cores
        if bins is not None:
            self.bins = bins

    def run(self, x_df, y_df, cores=None, bins=None, logtype=DEFAULT_LOG_TYPE, set_autoregulation_to_0=True):
        """
        Wrapper to calculate the CLR and MI for two data sets that have common condition columns
        :param x_df: pd.DataFrame
        :param y_df: pd.DataFrame
        :param logtype: np.log func
        :param cores: int
            Number of cores for multiprocessing
        :param bins: int
            Number of bins for discretizing continuous variables
        :param set_autoregulation_to_0: bool

        :return clr, mi: pd.DataFrame, pd.DataFrame
            CLR and MI DataFrames
        """

        if cores is not None:
            self.cores = cores

        if bins is not None:
            self.bins = bins

        mi = mutual_information(y_df, x_df, self.bins, cores=self.cores, logtype=logtype)
        mi_bg = mutual_information(x_df, x_df, self.bins, cores=self.cores, logtype=logtype)
        if set_autoregulation_to_0:
            clr = calc_mixed_clr(utils.df_set_diag(mi, 0), utils.df_set_diag(mi_bg, 0))
        else:
            clr = calc_mixed_clr(mi, mi_bg)

        return clr, mi


def mutual_information(X, Y, bins, cores=1, logtype=DEFAULT_LOG_TYPE):
    """
    Calculate the mutual information matrix between two data matrices, where the columns are equivalent conditions

    :param X: pd.DataFrame (m1 x n)
        The data from m1 variables across n conditions
    :param Y: pd.DataFrame (m2 x n)
        The data from m2 variables across n conditions
    :param bins: int
        Number of bins to discretize continuous data into for the generation of a contingency table
    :param cores: int
        Number of cores to use for this process
    :param logtype: np.log func
        Which type of log function should be used (log2 results in MI bits, log results in MI nats, log10... is weird)

    :return mi: pd.DataFrame (m2 x m1)
        The mutual information between variables m1 and m2
    """
    assert X.shape[1] == Y.shape[1]
    assert (X.columns == Y.columns).all()

    # Create dense output matrix and copy the inputs
    mi = np.zeros((X.shape[0], Y.shape[0]), dtype=np.dtype(float))
    mi_r = X.index
    mi_c = Y.index

    X = X.values
    Y = Y.values

    # Discretize the input matrixes
    X = _make_array_discrete(X, bins, axis=1).transpose()
    Y = _make_array_discrete(Y, bins, axis=1).transpose()

    # Run _calc_mi on every pairwise combination of features
    if cores == 1:
        for mi_data in _mi_gen(X, Y, bins, logtype=logtype):
            i, j, mi_val = _mi_mp_1d(mi_data)
            mi[i, j] = mi_val

    # Run _calc_mi on every pairwise combination of features using Pool to multiprocess
    else:
        # Create child processes with Pool. Spawn instead of os.fork() if possible.
        # For python 3.6 this works nicely to fix memory overallocation in jobs
        # For python 2.7 the only real solution now is to limit the number of cores
        # TODO: Do something smart to fix this problem with 2.7
        try:
            mp_pool = multiprocessing.Pool(processes=cores, context='spawn')
        except TypeError:
            mp_pool = multiprocessing.Pool(processes=cores)
            
        try:
            for mi_data in mp_pool.imap(_mi_mp_1d, _mi_gen(X, Y, bins, logtype=logtype), chunksize=POOL_CHUNKSIZE):
                i, j, mi_val = mi_data
                mi[i, j] = mi_val
        except KeyboardInterrupt:
            # In theory this should SIGTERM all the children and then clean them up before reraising the interrupt
            # In practice it'll probably just hang
            mp_pool.terminate()
            mp_pool.join()
            raise

        # Clean up the remaining child processes
        # This should happen on return but why take the chance
        mp_pool.close()
        mp_pool.join()

    mi_p = pd.DataFrame(mi, index=mi_r, columns=mi_c)
    return mi_p


def calc_mixed_clr(mi, mi_bg):
    """
    Calculate the context likelihood of relatedness from the dynamic data mi and a static background mi

    :param mi: pd.DataFrame
        Mutual information dataframe
    :param mi_bg: pd.DataFrame
        Background mutual information dataframe
    :return clr: pd.DataFrame
        Context likelihood of relatedness dataframe
    """
    # Calculate the zscore for columns
    z_col = mi.copy().round(8)
    z_col = z_col.subtract(mi_bg.mean(axis=0))
    z_col = z_col.divide(mi_bg.std(axis=0, ddof=CLR_DDOF))
    z_col[z_col < 0] = 0

    # Calculate the zscore for rows
    z_row = mi.copy().round(8)
    z_row = z_row.subtract(mi.mean(axis=1), axis=0)
    z_row = z_row.divide(mi.std(axis=1, ddof=CLR_DDOF), axis=0)
    z_row[z_row < 0] = 0

    clr = np.sqrt(np.square(z_col) + np.square(z_row))
    return clr


def _mi_gen(X, Y, bins, logtype=np.log):
    """
    Generator that yields a packed tuple of indices i & j, the column slices Xi and Yj, and passes through bins and log
    This allows for easy use of map

    :param X: np.ndarray
    :param Y: np.ndarray
    :param bins: int
    :param logtype: np.log func

    :yield: int, int, np.ndarray, np.ndarray, int, np.log func
    """

    for i in range(X.shape[1]):
        X_slice = X[:, i]
        for j in range(Y.shape[1]):
            yield (i, j, X_slice, Y[:, j], bins, logtype)


def _mi_mp_1d(data):
    """
    Wrappers _calc_mi and unpacks _mi_gen tuples. Just so that the generator can be used with map.
    """
    i, j, x, y, bins, logtype = data
    ctable = _make_table(x, y, bins)
    mi_val = _calc_mi(ctable, logtype=logtype)
    return i, j, mi_val


def _make_array_discrete(array, num_bins, axis=0):
    """
    Applies _make_discrete to a 2d array
    """
    return np.apply_along_axis(_make_discrete, arr=array, axis=axis, num_bins=num_bins)


def _make_discrete(array, num_bins):
    """
    Takes a 1d array or vector and discretizes it into nonparametric bins
    :param array: np.ndarray
        1d array of continuous data
    :param num_bins: int
        Number of bins for data
    :return array: np.ndarray
        1d array of discrete data
    """

    # Sanity check
    if not isinstance(array, list):
        try:
            if array.shape[1] != 1:
                raise ValueError("make_discrete takes a 1d array")
        except IndexError:
            pass

    # Create a function to convert continuous values to discrete bins
    arr_min = np.min(array)
    arr_max = np.max(array)
    eps_mod = max(np.finfo(float).eps, np.finfo(float).eps * (arr_max - arr_min))
    disc_func = np.vectorize(lambda x: np.floor((x - arr_min) / (arr_max - arr_min + eps_mod) * num_bins))

    # Apply the function to every value in the vector
    return disc_func(array).astype(np.dtype(int))


def _make_table(x, y, num_bins):
    """
    Takes two variable vectors which have been made into discrete integer bins and constructs a contingency table
    :param x: np.ndarray
        1d array of discrete data
    :param y: np.ndarray
        1d array of discrete data
    :param num_bins: int
        Number of bins for data
    :return ctable: np.ndarray (num_bins x num_bins)
        Contingency table of variables X and Y
    """

    reindex = x * num_bins + y
    return np.bincount(reindex, minlength=num_bins ** 2).reshape(num_bins, num_bins).astype(np.dtype(float))


def _calc_mi(table, logtype=DEFAULT_LOG_TYPE):
    """
    Calculate Mutual Information from a contingency table of two variables

    :param table: np.ndarray (num_bins x num_bins)
        Contingency table
    :param logtype: np.log func
        Log function to use
    :return: float
        Mutual information between variable x & y
    """

    # Turn off runtime warnings (there is an explicit check for NaNs and INFs in-function)
    reset = np.seterr(divide='ignore', invalid='ignore')

    m, n = table.shape
    assert n == m

    total = np.sum(table, axis=(0, 1))

    # (PxPy) [n x n]
    mi_val = np.dot((np.sum(table, axis=1) / total).reshape(-1, 1),
                    (np.sum(table, axis=0) / total).reshape(1, -1))

    # (Pxy) [n x n]
    table = np.divide(table, total)

    # (Pxy)/(PxPy) [n x n]
    mi_val = np.divide(table, mi_val)

    # log[(Pxy)/(PxPy)] [n x n]
    mi_val = logtype(mi_val)

    # Pxy(log[(Pxy)/(PxPy)]) [n x n]
    mi_val = np.multiply(table, mi_val)
    mi_val[np.isnan(mi_val)] = 0

    # Summation
    mi_val = np.sum(mi_val, axis=(0, 1))

    np.seterr(**reset)
    return mi_val
