"""
Miscellaneous utility modules.
"""

import pandas as pd
import os

SBATCH_VARS = {'RUNDIR': ('output_dir', str, None),
               'DATADIR': ('input_dir', str, None),
               'SLURM_PROCID': ('rank', int, 0),
               'SLURM_NTASKS_PER_NODE': ('cores', int, 10),
               'SLURM_NTASKS': ('tasks', int, 1)
               }


def slurm_envs():
    envs = {}
    for os_var, (cv, mt, de) in SBATCH_VARS.items():
        try:
            val = mt(os.environ[os_var])
        except (KeyError, TypeError):
            val = de
        envs[cv] = val
    return envs


def ownCheck(kvs, rank, chunk=1, kvs_key='count'):
    """
    Generator

    :param kvs: KVSClient
        KVS object for server access
    :param rank: int
        SLURM proc ID
    :param chunk: int
        The size of the chunk given to each subprocess
    :param kvs_key: str
        The KVS key to increment (default is 'count')

    :yield: bool
        True if this process has dibs on whatever. False if some other process has claimed it first.
    """

    # If we're the main process, set KVS key to 0
    if 0 == rank:
        kvs.put(kvs_key, 0)

    # Start at the baseline
    checks, lower, upper = 0, -1, -1

    while 1:

        # Checks increments every loop
        # If it's greater than the upper bound, get a new lower bound from the KVS count
        # Set the new upper bound by adding chunk to lower
        # And then put the new upper bound back into KVS key

        if checks >= upper:
            lower = kvs.get(kvs_key)
            upper = lower + chunk
            kvs.put(kvs_key, upper)

        # Yield TRUE if this row belongs to this process and FALSE if it doesn't
        yield lower <= checks < upper
        checks += 1


def kvs_sync_processes(kvs, rank, pref=""):
    # Block all processes until they reach this point
    # Then release them
    # It may be wise to use unique prefixes if this is gonna get called rapidly so there's no collision
    # Or not. I'm a comment, not a cop.

    n = slurm_envs()['tasks']
    wkey = pref + '_wait'
    ckey = pref + '_continue'

    kvs.put(wkey, True)
    if rank == 0:
        for _ in range(n):
            kvs.get(wkey)
        for _ in range(n):
            kvs.put(ckey, True)
    kvs.get(ckey)


def kvsTearDown(kvs, rank, kvs_key='count'):
    # de-initialize the global counter.
    if 0 == rank:
        # Do a hard reset if rank == 0
        kvs.get(kvs_key)


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
