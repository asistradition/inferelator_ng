"""
Workflow class that splits the prior into a gold standard and new prior
"""

import pandas as pd
import numpy as np
from . import workflow

class PriorGoldStandardSplitWorkflowBase(workflow.WorkflowBase):

    def set_gold_standard_and_priors(self, gold_standard_split=0.5):
        """
        Ignore the gold standard file. Instead, create a gold standard
        from a 50/50 split of the prior. Half of original prior becomes the new prior, 
        the other half becomes the gold standard
        """

        if self.priors_data is None:
            all_priors = self.input_dataframe(self.priors_file)
        else:
            all_priors = self.priors_data.copy()

        pc = np.sum(all_priors.values != 0)
        gs_count = int(gold_standard_split * pc)

        idx = list(range(pc))
        np.random.shuffle(idx)

        pr_idx = all_priors.values[all_priors.values != 0].copy()
        gs_idx = all_priors.values[all_priors.values != 0].copy()

        pr_idx[idx[0:gs_count]] = 0
        gs_idx[idx[gs_count:]] = 0

        gs = all_priors.values.copy()
        pr = all_priors.values.copy()

        gs[gs != 0] = gs_idx
        pr[pr != 0] = pr_idx

        self.priors_data = pd.DataFrame(pr, index=all_priors.index, columns=all_priors.columns)
        self.gold_standard = pd.DataFrame(gs, index=all_priors.index, columns=all_priors.columns)

class ResultsProcessorForGoldStandardSplit(results_processor.ResultsProcessor):

    def create_filtered_gold_standard_and_confidences(self, combined_confidences, gold_standard, priors, filter_index, filter_cols):
        # this code only runs for a positive gold standard, so explicitly transform it using the absolute value:
        gold_standard = np.abs(gold_standard)
        gold_standard_filtered = gold_standard.loc[filter_index, filter_cols]
        priors_data_filtered = priors.loc[filter_index, filter_cols]
        combined_confidences_filtered = combined_confidences.loc[filter_index, filter_cols]
        # removing correctly predicted interactions that were removed from GS because GS was split:
        combined_confidences_filtered = combined_confidences_filtered*(1-priors_data_filtered.abs())
        # rank from highest to lowest confidence
        return(combined_confidences_filtered, gold_standard_filtered)