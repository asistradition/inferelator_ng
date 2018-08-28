from inferelator_ng.single_cell_bbsr_tfa_workflow import Single_Cell_BBSR_TFA_Workflow
from inferelator_ng import utils

utils.Debug.set_verbose_level(utils.Debug.levels["verbose"])

# Build the workflow
workflow = Single_Cell_BBSR_TFA_Workflow()
# Common configuration parameters
workflow.append_to_path('input_dir', 'yeast')
workflow.expression_matrix_file = 'expr_100k_5000umi.tsv'
workflow.priors_file = "yeast-motif-prior.tsv"
workflow.num_bootstraps = 5
workflow.random_seed = 42

# Run the workflow
workflow.run()
