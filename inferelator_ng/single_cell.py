import pandas as pd
import numpy as np
import itertools
import shutil

from inferelator_ng.default import DEFAULT_METADATA_FOR_BATCH_CORRECTION
from inferelator_ng.default import DEFAULT_RANDOM_SEED
from inferelator_ng import utils

"""
This file is all preprocessing functions. All functions must take positional arguments expression_matrix and meta_data.
All other arguments must be keyword. All functions must return expression_matrix and meta_data (modified or unmodified).

Normalization functions take batch_factor_column [str] as a kwarg
Imputation functions take random_seed [int] and output_file [str] as a kwarg 

Please note that there are a bunch of packages in here that aren't installed as part of the project dependencies
This is intentional; if you don't have these packages installed, don't try to use them
TODO: Put together a set of tests for this 
"""


def normalize_expression_to_one(expression_matrix, meta_data, **kwargs):
    """

    :param expression_matrix:
    :param meta_data:
    :param batch_factor_column:
    :return expression_matrix, meta_data: pd.DataFrame, pd.DataFrame
    """
    kwargs, batch_factor_column = process_normalize_args(**kwargs)

    utils.Debug.vprint('Normalizing UMI counts per cell ... ')

    # Get UMI counts for each cell
    umi = expression_matrix.sum(axis=1)

    # Divide each cell's raw count data by the total number of UMI counts for that cell
    return expression_matrix.astype(float).divide(umi, axis=0), meta_data


def normalize_medians_for_batch(expression_matrix, meta_data, **kwargs):
    """
    Calculate the median UMI count per cell for each batch. Transform all batches by dividing by a size correction
    factor, so that all batches have the same median UMI count (which is the median batch median UMI count)
    :param expression_matrix: pd.DataFrame
    :param meta_data: pd.DataFrame
    :param batch_factor_column: str
        Which meta data column should be used to determine batches
    :return expression_matrix, meta_data: pd.DataFrame, pd.DataFrame
    """
    kwargs, batch_factor_column = process_normalize_args(**kwargs)

    utils.Debug.vprint('Normalizing median counts between batches ... ')

    # Get UMI counts for each cell
    umi = expression_matrix.sum(axis=1)

    # Create a new dataframe with the UMI counts and the factor to batch correct on
    umi = pd.DataFrame({'umi': umi, batch_factor_column: meta_data[batch_factor_column]})

    # Group and take the median UMI count for each batch
    median_umi = umi.groupby(batch_factor_column).agg('median')

    # Convert to a correction factor based on the median of the medians
    median_umi = median_umi / median_umi['umi'].median()
    umi = umi.join(median_umi, on=batch_factor_column, how="left", rsuffix="_mod")

    # Apply the correction factor to all the data
    return expression_matrix.divide(umi['umi_mod'], axis=0), meta_data


def normalize_sizes_within_batch(expression_matrix, meta_data, **kwargs):
    """
    Calculate the median UMI count within each batch and then resize each sample so that each sample has the same total
    UMI count

    :param expression_matrix: pd.DataFrame
    :param meta_data: pd.DataFrame
    :param batch_factor_column: str
        Which meta data column should be used to determine batches
    :return expression_matrix, meta_data: pd.DataFrame, pd.DataFrame
    """

    kwargs, batch_factor_column = process_normalize_args(**kwargs)

    utils.Debug.vprint('Normalizing to median counts within batches ... ')

    # Get UMI counts for each cell
    umi = expression_matrix.sum(axis=1)

    # Create a new dataframe with the UMI counts and the factor to batch correct on
    umi = pd.DataFrame({'umi': umi, batch_factor_column: meta_data[batch_factor_column]})

    # Group and take the median UMI count for each batch
    median_umi = umi.groupby(batch_factor_column).agg('median')

    # Convert to a correction factor based on the median of the medians
    umi = umi.join(median_umi, on="Condition", how="left", rsuffix="_mod")
    umi['umi_mod'] = umi['umi'] / umi['umi_mod']

    # Apply the correction factor to all the data
    return expression_matrix.divide(umi['umi_mod'], axis=0), meta_data


def normalize_multiBatchNorm(expression_matrix, meta_data, **kwargs):
    """
    Normalize as multiBatchNorm from the R package scran
    :param expression_matrix: pd.DataFrame
    :param meta_data: pd.DataFrame
    :param batch_factor_column: str
        Which meta data column should be used to determine batches
    :param minimum_mean: int
        Minimum mean expression of a gene when considering if it should be included in the correction factor calc
    :return expression_matrix, meta_data: pd.DataFrame, pd.DataFrame
    """

    utils.Debug.vprint('Normalizing by multiBatchNorm ... ')
    kwargs, batch_factor_column = process_normalize_args(**kwargs)
    minimum_mean = kwargs.pop('minimum_mean', 50)

    # Calculate size-corrected average gene expression for each batch
    size_corrected_avg = pd.DataFrame(columns=expression_matrix.columns)
    for batch in meta_data[batch_factor_column].unique().tolist():
        batch_df = expression_matrix.loc[meta_data[batch_factor_column] == batch, :]

        # Get UMI counts for each cell
        umi = batch_df.sum(axis=1)
        size_correction_factor = umi / umi.mean()

        # Get the mean size-corrected count values for this batch
        batch_df = batch_df.divide(size_correction_factor, axis=0).mean(axis=0).to_frame().transpose()
        batch_df.index = pd.Index([batch])

        # Append to the dataframe
        size_corrected_avg = size_corrected_avg.append(batch_df)

    # Calculate median ratios
    inter_batch_coefficients = []
    for b1, b2 in itertools.combinations_with_replacement(size_corrected_avg.index.tolist(), r=2):
        # Get the mean size-corrected count values for this batch pair
        b1_series, b2_series = size_corrected_avg.loc[b1, :], size_corrected_avg.loc[b2, :]
        b1_sum, b2_sum = b1_series.sum(), b2_series.sum()

        # calcAverage
        combined_keep_index = ((b1_series / b1_sum + b2_series / b2_sum) / 2 * (b1_sum + b2_sum) / 2) > minimum_mean
        coeff = (b2_series.loc[combined_keep_index] / b1_series.loc[combined_keep_index]).median()

        # Keep track of the median ratios
        inter_batch_coefficients.append((b1, b2, coeff))
        inter_batch_coefficients.append((b2, b1, 1 / coeff))

    inter_batch_coefficients = pd.DataFrame(inter_batch_coefficients, columns=["batch1", "batch2", "coeff"])
    inter_batch_minimum = inter_batch_coefficients.loc[inter_batch_coefficients["coeff"].idxmin(), :]

    min_batch = inter_batch_minimum["batch2"]

    # Apply the correction factor to all the data batch-wise. Do this with numpy because pandas is a glacier.
    normed_expression = np.ndarray((0, expression_matrix.shape[1]), dtype=np.dtype(float))
    normed_meta = pd.DataFrame(columns=meta_data.columns)

    for i, row in inter_batch_coefficients.loc[inter_batch_coefficients["batch2"] == min_batch, :].iterrows():
        select_rows = meta_data[batch_factor_column] == row["batch1"]
        umi = expression_matrix.loc[select_rows, :].sum(axis=1)
        size_correction_factor = umi / umi.mean() / row["coeff"]
        corrected_df = expression_matrix.loc[select_rows, :].divide(size_correction_factor, axis=0).values
        normed_expression = np.vstack((normed_expression, corrected_df))
        normed_meta = pd.concat([normed_meta, meta_data.loc[select_rows, :]])

    return pd.DataFrame(normed_expression, index=normed_meta.index, columns=expression_matrix.columns), normed_meta


def impute_magic_expression(expression_matrix, meta_data, **kwargs):
    """
    Use MAGIC (van Dijk et al Cell, 2018, 10.1016/j.cell.2018.05.061) to impute data

    :param expression_matrix: pd.DataFrame
    :param meta_data: pd.DataFrame
    :return imputed, meta_data: pd.DataFrame, pd.DataFrame
    """
    kwargs, random_seed, output_file = process_impute_args(**kwargs)

    import magic
    utils.Debug.vprint('Imputing data with MAGIC ... ')
    imputed = pd.DataFrame(magic.MAGIC(random_state=random_seed, **kwargs).fit_transform(expression_matrix.values),
                           index=expression_matrix.index, columns=expression_matrix.columns)

    if output_file is not None:
        imputed.to_csv(output_file, sep="\t")

    return imputed, meta_data


def impute_on_batches(expression_matrix, meta_data, **kwargs):
    """
    Run imputation on separate batches
    :param expression_matrix: pd.DataFrame
    :param meta_data: pd.DataFrame
    :param impute_method: func
        An imputation function from inferelator_ng.single_cell
    :param random_seed: int
        Random seed to put into the imputation method
    :param batch_factor_column: str
        Which meta data column should be used to determine batches
    :return expression_matrix, meta_data: pd.DataFrame, pd.DataFrame
    """

    # Extract random_seed, batch_factor_column, and impute method for use. Extract and eat output_file.
    kwargs, batch_factor_column = process_normalize_args(**kwargs)
    kwargs, random_seed, _ = process_impute_args(**kwargs)
    impute_method = kwargs.pop('impute_method', impute_magic_expression)

    batches = meta_data[batch_factor_column].unique().tolist()
    bc_expression = np.ndarray((0, expression_matrix.shape[1]), dtype=np.dtype(float))
    bc_meta = pd.DataFrame(columns=meta_data.columns)
    for batch in batches:
        rows = meta_data[batch_factor_column] == batch
        batch_corrected, _ = impute_method(expression_matrix.loc[rows, :], None, random_seed=random_seed, **kwargs)
        bc_expression = np.vstack((bc_expression, batch_corrected))
        bc_meta = pd.concat([bc_meta, meta_data.loc[rows, :]])
        random_seed += 1
    return pd.DataFrame(bc_expression, index=bc_meta.index, columns=expression_matrix.columns), bc_meta


def log10_data(expression_matrix, meta_data, **kwargs):
    """
    Transform the expression data by adding one and then taking log10. Ignore any kwargs.
    :param expression_matrix: pd.DataFrame
    :param meta_data: pd.DataFrame
    :return expression_matrix, meta_data: pd.DataFrame, pd.DataFrame
    """
    utils.Debug.vprint('Logging data [log10+1] ... ')
    return np.log10(expression_matrix + 1), meta_data


def log2_data(expression_matrix, meta_data, **kwargs):
    """
    Transform the expression data by adding one and then taking log2. Ignore any kwargs.
    :param expression_matrix: pd.DataFrame
    :param meta_data: pd.DataFrame
    :return expression_matrix, meta_data: pd.DataFrame, pd.DataFrame
    """
    utils.Debug.vprint('Logging data [log2+1]... ')
    return np.log2(expression_matrix + 1), meta_data


def ln_data(expression_matrix, meta_data, **kwargs):
    """
    Transform the expression data by adding one and then taking ln. Ignore any kwargs.
    :param expression_matrix: pd.DataFrame
    :param meta_data: pd.DataFrame
    :return expression_matrix, meta_data: pd.DataFrame, pd.DataFrame
    """
    utils.Debug.vprint('Logging data [ln+1]... ')
    return np.log1p(expression_matrix), meta_data


def filter_genes_for_var(expression_matrix, meta_data, **kwargs):
    no_signal = (expression_matrix.max(axis=0) - expression_matrix.min(axis=0)) == 0
    utils.Debug.vprint("Filtering {gn} genes [Var = 0]".format(gn=no_signal.sum()), level=1)
    return expression_matrix.loc[:, ~no_signal], meta_data


def filter_genes_for_count(expression_matrix, meta_data, count_minimum=None, check_for_scaling=False):
    expression_matrix, meta_data = filter_genes_for_var(expression_matrix, meta_data)
    if count_minimum is None:
        return expression_matrix, meta_data
    else:
        if check_for_scaling and (expression_matrix < 0).sum().sum() > 0:
            raise ValueError("Negative values in the expression matrix. Count thresholding scaled data is unsupported.")

        keep_genes = expression_matrix.sum(axis=0) >= (count_minimum * expression_matrix.shape[0])
        utils.Debug.vprint("Filtering {gn} genes [Count]".format(gn=expression_matrix.shape[1] - keep_genes.sum()),
                           level=1)
        return expression_matrix.loc[:, keep_genes], meta_data


def process_impute_args(**kwargs):
    random_seed = kwargs.pop('random_seed', DEFAULT_RANDOM_SEED)
    output_file = kwargs.pop('output_file', None)
    return kwargs, random_seed, output_file


def process_normalize_args(**kwargs):
    batch_factor_column = kwargs.pop('batch_factor_column', DEFAULT_METADATA_FOR_BATCH_CORRECTION)
    return kwargs, batch_factor_column
