"""
Miscellaneous utility modules.
"""

import pandas as pd

def ownCheck(kvs, rank, chunk=1):
    # initialize a global counter.
    if 0 == rank:
        kvs.put('count', 0)
    checks, lower, upper = 0, -1, -1
    while 1:
        if checks >= upper:
            lower = kvs.get('count')
            upper = lower + chunk
            kvs.put('count', upper)
        yield lower <= checks < upper
        checks += 1


def kvsTearDown(kvs, rank):
    # de-initialize the global counter.
    if 0 == rank:
        # Do a hard reset if rank == 0
        kvs.get('count')


def get_all_except(l, idx):
    return l[:idx] + l[idx + 1:]


def df_from_tsv(file_like, has_index=True):
    "Read a tsv file or buffer with headers and row ids into a pandas dataframe."
    return pd.read_csv(file_like, sep="\t", header=0, index_col=0 if has_index else False)


def metadata_df(file_like):
    "Read a metadata file as a pandas data frame."
    return pd.read_csv(file_like, sep="\t", header=0, index_col="condName")


def read_tf_names(file_like):
    "Read transcription factor names from one-column tsv file.  Return list of names."
    exp = pd.read_csv(file_like, sep="\t", header=None)
    assert exp.shape[1] == 1, "transcription factor file should have one column "
    return list(exp[0])


def df_set_diag(df, val, copy=True):
    """
    Sets the diagonal of a dataframe to a value. Diagonal in this case is anything where row label == column label.

    :param df: pd.DataFrame
        DataFrame to modify
    :param val: numeric
        Value to insert into any cells where row label == column label
    :param copy: bool
        Force-copy the dataframe instead of modifying in place
    :return: pd.DataFrame / int
        Return either the modified dataframe (if copied) or the number of cells modified (if changed in-place)
    """

    # Find all the labels that are shared between rows and columns
    isect = df.index.intersection(df.columns)

    if copy:
        df = df.copy()

    # Set the value where row and column names are the same
    for i in range(len(isect)):
        df.loc[isect[i], isect[i]] = val

    if copy:
        return df
    else:
        return len(isect)


def make_array_2d(arr):
    """
    Changes array shape from 1d to 2d if needed (in-place)
    :param arr:  np.ndarray
    """
    if arr.ndim == 1:
        arr.shape = (arr.shape[0], 1)
