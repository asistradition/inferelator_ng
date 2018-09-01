import numpy as np
import pandas as pd
import itertools
import os
import datetime

from inferelator_ng import bbsr_tfa_workflow
from inferelator_ng import workflow
from inferelator_ng.design_response_translation import PythonDRDriver
from inferelator_ng.prior_gs_split_workflow import PriorGoldStandardSplitWorkflowBase
from inferelator_ng.prior_gs_split_workflow import ResultsProcessorForGoldStandardSplit
from inferelator_ng import utils

# The variable names that get set in the main workflow, but need to get copied to the Regression Worker children
SHARED_CLASS_VARIABLES = ['delTmin', 'delTmax', 'reduce_searchspace', 'rank', 'input_dir', 'output_dir']


class PythonDRDriver_with_tau_vector(PythonDRDriver):
    """ 
        The class exists to modify the design-response calculation to have 
        tau as a vector instead of a scalar      
    """

    # define your own compute_response_variable:
    def compute_response_variable(self, tau, delta_follow, expr_current, expr_follow):
        tvec = tau.iloc[:, 0] / float(delta_follow)
        return tvec * (expr_follow.astype('float64') - expr_current.astype('float64')) + expr_current.astype('float64')


class InfereCLaDR_Regression_Workflow(bbsr_tfa_workflow.BBSR_TFA_Workflow):
    """
    Standard BBSR TFA Workflow except it takes all the data as references to __init__ instead of as filenames on disk or
    as environment variables
    """

    reduce_searchspace = False

    def __init__(self, kvs, rank, expr_data, prior_data, gs_data, tf_names):
        self.kvs = kvs
        self.rank = rank
        self.expression_matrix = expr_data
        self.priors_data = prior_data
        self.gold_standard = gs_data
        self.tf_names = tf_names

    def startup_run(self):
        self.compute_common_data()
        self.compute_activity()

    def startup_finish(self):
        if self.reduce_searchspace:
            gs_genes = self.gold_standard.index.tolist()
            resp_genes = self.response.index.tolist()
            genes_reduced = list(set.intersection(set(gs_genes), set(resp_genes)))
            self.response = self.response.loc[genes_reduced, :]
            self.priors_data = self.priors_data.loc[genes_reduced, :]


class InfereCLaDR_Puppet_Workflow(PriorGoldStandardSplitWorkflowBase, InfereCLaDR_Regression_Workflow):
    """
    Standard BBSR TFA Workflow except it takes all the data as references to __init__ instead of as filenames on disk or
    as environment variables. When run it generates betas and resc_betas, but then returns them instead of processing
    """

    async_start = False

    def run(self):
        self.startup()
        return self.run_regression()

    def emit_results(self, betas, rescaled_betas, gold_standard, priors):
        pass

    def startup_run(self):
        self.set_gold_standard_and_priors()
        self.compute_common_data()
        self.compute_activity()


class TauVector_Workflow:

    def compute_common_data(self):
        self.filter_expression_and_priors()
        drd = PythonDRDriver_with_tau_vector()
        utils.Debug.vprint('Creating design and response matrix ... ')
        drd.delTmin, drd.delTmax, drd.tau = self.delTmin, self.delTmax, self.tau
        self.design, self.response = drd.run(self.expression_matrix, self.meta_data)
        drd.tau = self.tau / 2
        self.design, self.half_tau_response = drd.run(self.expression_matrix, self.meta_data)
        self.expression_matrix = None


class InfereCLaDR_TauVector_Workflow(TauVector_Workflow, InfereCLaDR_Regression_Workflow):
    pass


class InfereCLaDR_Workflow(workflow.WorkflowBase):

    full_gene_clust_mapping_file = 'clusters/gn_clust_full.tsv'
    expr_clust_files = ["clusters/expr_clust1.tsv",
                        "clusters/expr_clust2.tsv",
                        "clusters/expr_clust3.tsv",
                        "clusters/expr_clust4.tsv"]
    gene_clust_files = ["clusters/genes_clust1.tsv",
                        "clusters/genes_clust2.tsv",
                        "clusters/genes_clust3.tsv",
                        "clusters/genes_clust4.tsv",
                        "clusters/genes_clust5.tsv"]
    meta_clust_files = ["clusters/meta_clust1.tsv",
                        "clusters/meta_clust2.tsv",
                        "clusters/meta_clust3.tsv",
                        "clusters/meta_clust4.tsv"]

    seeds = range(42, 62)
    taus = [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 200, 250]

    gene_cluster_data = None
    cluster_idx = None

    def run(self):
        # Load in data
        self.startup()

        # Search for optimal cluster Taus
        data = self.search_clusters()
        if self.is_master():
            data = self.process_aupr_search(data)
            self.write_df(data)
            self.kvs.put('Opt_tau', data)
        else:
            data = self.kvs.view('Opt_tau')
        utils.kvs_sync_then_teardown(self.kvs, self.rank, kvs_key='Opt_tau')

        # Load the main (non-clustered) data
        self.read_expression()
        self.read_metadata()

        # Run the regression using taus generated from each of the separate condition clusters
        for i in range(len(self.expr_clust_files)):
            taus = self.map_tau_to_genes(data, i)['Tau']
            regd = self.create_regression_driver(1, taus, driver=InfereCLaDR_TauVector_Workflow)
            self.assign_class_vars(regd)
            regd.append_to_path('output_dir', str(i))
            regd.run()

    def startup_run(self):
        self.read_tfs()
        self.set_gold_standard_and_priors()
        self.load_gene_clusters()

    def startup_finish(self):
        pass

    def search_clusters(self):
        cluster_data = []
        for i in range(len(self.expr_clust_files)):
            utils.Debug.vprint("Regressing on cluster {i} of {t}".format(i=i + 1, t=len(self.expr_clust_files)),
                               level=0)
            self.load_cluster_data(i)
            self.cluster_idx = i
            cluster_data.extend(self.search_tau_space())
        return cluster_data

    def load_cluster_data(self, idx):
        self.expression_matrix = self.input_dataframe(self.expr_clust_files[idx])
        self.meta_data = self.input_dataframe(self.meta_clust_files[idx], has_index=False)

    def search_tau_space(self):
        data = []
        tots = len(self.seeds) * len(self.taus)
        for i, (s, t) in enumerate(itertools.product(self.seeds, self.taus)):
            utils.Debug.vprint("\tRegressing: tau {ta} (Seed {s}) [{i} / {t}]".format(ta=t, s=s, i=i, t=tots), level=0)
            data.extend(self.run_st_regression(s, t))
            utils.kvs_sync_processes(kvs=self.kvs, rank=self.rank)
        return data

    def run_st_regression(self, s, t):
        regd = self.create_regression_driver(s, t, driver=InfereCLaDR_Puppet_Workflow)
        betas, resc_betas = regd.run()
        data = []
        if self.is_master():
            aupr = self.process_results(betas, resc_betas, regd.gold_standard, regd.priors_data)
            for g, a in enumerate(aupr):
                # (Condition_cluster, Gene_cluster, Tau, Seed, AUPR)
                data.append((self.cluster_idx, g, t, s, a))
        return data

    def create_regression_driver(self, s, t, driver=InfereCLaDR_Regression_Workflow):
        regd = driver(self.kvs, self.rank, self.expression_matrix, self.priors_data, self.gold_standard, self.tf_names)
        self.assign_class_vars(regd)
        regd.random_seed, regd.tau = s, t
        return regd

    def assign_class_vars(self, obj):
        for varname in SHARED_CLASS_VARIABLES:
            setattr(obj, varname, getattr(self, varname))

    def process_results(self, betas, resc_betas, gs_data, prior_data):
        rp = ResultsProcessorForGoldStandardSplit(betas, resc_betas)
        combined_confidences = rp.compute_combined_confidences()
        filter_index, filter_cols = rp.get_nonempty_rows_cols(combined_confidences, gs_data)
        aupr = []
        for c_idx in self.gene_cluster_gen():
            genes_index_filtered = set(c_idx).intersection(filter_index)
            cc_clust, gs_clust = rp.create_filtered_gold_standard_and_confidences(combined_confidences,
                                                                                  gs_data,
                                                                                  prior_data,
                                                                                  genes_index_filtered,
                                                                                  filter_cols)
            precision, recall = rp.calculate_precision_recall(cc_clust, gs_clust)
            aupr.append(rp.calculate_aupr(precision, recall))
        return aupr

    def load_gene_clusters(self):
        data = pd.DataFrame(columns=['Cluster', 'Gene'])
        for i, f in enumerate(self.gene_clust_files):
            genes = pd.read_table(self.input_file(f), sep="\t", header=None)
            assert genes.shape[1] == 1
            genes.columns = ['Gene']
            genes.index = map(str, genes['Gene'].tolist())
            genes['GC'] = i
            data = pd.concat((data, genes), axis=0)
        self.gene_cluster_data = data

    def gene_cluster_gen(self):
        for i in range(len(self.gene_clust_files)):
            data = self.gene_cluster_data[self.gene_cluster_data['GC'] == i]
            yield data['Gene'].tolist()

    def map_tau_to_genes(self, tau_data, cc):
        tau_data = tau_data[tau_data['CC'] == cc]
        return self.gene_cluster_data.join(tau_data[['GC', 'Tau']], on='GC')

    def write_df(self, df):
        if self.output_dir is None:
            self.output_dir = os.path.join(self.input_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        try:
            os.makedirs(self.output_dir)
        except OSError:
            pass
        df.to_csv(os.path.join(self.output_dir, 'predicted_half-lives.tsv'), sep="\t")

    @staticmethod
    def process_aupr_search(data):
        data = pd.DataFrame(data, columns=['CC', 'GC', 'Tau', 'Seed', 'AUPR'])

        # For every CC, GC, and Seed combination, keep the row with the highest AUPR
        group = ['CC', 'GC', 'Seed']
        reidx = data.groupby(group)['AUPR'].transform(max) == data['AUPR']
        data = data[reidx]
        data = data[~data.duplicated(subset=group)]

        # For every CC & GC, keep the row with the median AUPR
        group = ['CC', 'GC']
        reidx = data.groupby(group)['AUPR'].transform('median') == data['AUPR']
        data = data[reidx]
        data = data[~data.duplicated(subset=group)]

        data['HalfLife'] = data['Tau'] * np.log(2)
        return data
