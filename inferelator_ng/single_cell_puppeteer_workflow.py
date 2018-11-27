import os
import datetime

import numpy as np

from inferelator_ng import single_cell_workflow
from inferelator_ng import tfa_workflow
from inferelator_ng import workflow
from inferelator_ng import results_processor


class NoOutputRP(results_processor.ResultsProcessor):

    def summarize_network(self, output_dir, gold_standard, priors):
        combined_confidences = self.compute_combined_confidences()
        (recall, precision) = self.calculate_precision_recall(combined_confidences, gold_standard)
        return self.calculate_aupr(recall, precision)


def make_puppet_workflow(workflow_type):
    class SingleCellPuppetWorkflow(single_cell_workflow.SingleCellWorkflow, workflow_type):
        """
        Standard workflow except it takes all the data as references to __init__ instead of as filenames on disk or
        as environment variables, and saves the model AUPR without outputting anything
        """

        def __init__(self, kvs, rank, expr_data, meta_data, prior_data, gs_data, tf_names, size=None):
            self.kvs = kvs
            self.rank = rank
            self.expression_matrix = expr_data
            self.meta_data = meta_data
            self.priors_data = prior_data
            self.gold_standard = gs_data
            self.tf_names = tf_names
            self.size = size

        def startup_run(self):
            self.compute_activity()

        def emit_results(self, betas, rescaled_betas, gold_standard, priors):
            if self.is_master():
                self.aupr = NoOutputRP(betas, rescaled_betas).summarize_network(None, gold_standard, priors)
            else:
                self.aupr = None

        def get_bootstraps(self):
            if self.size is not None:
                return np.random.choice(self.response.shape[1], size=(self.num_bootstraps, self.size)).tolist()
            else:
                return workflow.WorkflowBase.get_bootstraps(self)

    return SingleCellPuppetWorkflow


class SingleCellPuppeteerWorkflow(single_cell_workflow.SingleCellWorkflow, tfa_workflow.TFAWorkFlow):
    seeds = range(42, 45)
    regression_type = tfa_workflow.BBSR_TFA_Workflow
    header = ["Seed", "AUPR"]

    def compute_activity(self):
        pass

    def single_cell_normalize(self):
        pass

    def emit_results(self, auprs, file_name="aupr.tsv"):

        if self.is_master():
            if self.output_dir is None:
                self.output_dir = os.path.join(self.input_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
            try:
                os.makedirs(self.output_dir)
            except OSError:
                pass
            with open(os.path.join(self.output_dir, file_name), mode="w") as out_fh:
                print("\t".join(self.header), file=out_fh)
                for tup in auprs:
                    print("\t".join(tup), file=out_fh)

    def get_aupr_for_seeds(self, expr_data, meta_data, regression_type, priors_data=None, gold_standard=None):
        if gold_standard is None:
            gold_standard = self.gold_standard
        if priors_data is None:
            priors_data = self.priors_data

        aupr_data = []
        for seed in self.seeds:
            puppet = make_puppet_workflow(regression_type)(self.kvs, self.rank, expr_data, meta_data,
                                                           priors_data, gold_standard, self.tf_names)
            puppet.random_seed = seed
            puppet.run()
            aupr_data.append((seed, puppet.aupr))
        return aupr_data


class SingleCellSizeSampling(SingleCellPuppeteerWorkflow):
    sizes = [1]
    header = ["Size", "Seed", "AUPR"]

    def run(self):
        self.startup()
        aupr_data = self.get_aupr_for_seeds(self.expression_matrix, self.meta_data, self.regression_type)
        self.emit_results(aupr_data)

    def get_aupr_for_resized_data(self, expr_data, meta_data, regression_type):
        aupr_data = []
        for s_ratio in self.sizes:
            new_size = int(s_ratio * self.expression_matrix.shape[1])
            new_idx = np.random.choice(expr_data.shape[1], size=new_size)
            auprs = self.get_aupr_for_seeds(expr_data.loc[:, new_idx],
                                            meta_data.loc[new_idx, :],
                                            regression_type=regression_type)
            aupr_data.extend([(new_size, se, au) for (se, au) in auprs])
        return aupr_data


class SingleCellDropoutConditionSampling(SingleCellPuppeteerWorkflow):
    drop_column = None

    def run(self):
        self.startup()
        aupr_data = self.auprs_for_condition_dropout()
        self.emit_results(aupr_data)

    def auprs_for_condition_dropout(self):
        aupr_data = []
        idx = self.condition_dropouts()
        for r_name, r_idx in idx.items():
            auprs = self.auprs_for_index(r_idx)
            aupr_data.extend([(r_name, se, au) for (se, au) in auprs])
        return aupr_data

    def condition_dropouts(self):
        condition_indexes = dict()
        for cond in self.meta_data[self.drop_column].unique().tolist():
            condition_indexes[cond] = self.meta_data[self.drop_column] == cond
        return condition_indexes

    def auprs_for_index(self, idx):
        local_expr_data = self.expression_matrix.iloc[:, idx]
        local_meta_data = self.meta_data.iloc[idx, :]
        return self.get_aupr_for_seeds(local_expr_data, local_meta_data, self.regression_type)

class SingleCellGSPriorMux(SingleCellPuppeteerWorkflow):

    def run(self):
        pass