from inferelator_ng.bbsr_tfa_workflow import BBSR_TFA_Workflow
from inferelator_ng import utils

utils.Debug.set_verbose_level(utils.Debug.levels["verbose"])

#Build the workflow
workflow = BBSR_TFA_Workflow()
# Common configuration parameters
workflow.append_to_path('input_dir', 'yeast')
workflow.priors_file = "yeast-motif-prior.tsv"
workflow.num_bootstraps = 10
workflow.delTmax = 110
workflow.delTmin = 0
workflow.tau = 45
workflow.random_seed = 1

#Run the workflow
workflow.run()

