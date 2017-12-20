# Application construction
BEST_PIPELINE_PYTHON_FILENAME = 'output/tpot_spam_selected_pipeline.py'
MODEL_SELECTOR_TRAIN_DATASET_SIZE = 0.90
MODEL_SELECTOR_TEST_DATASET_SIZE = 0.10
DEFAULT_RANDOM_STATE = 42
MODEL_BUILDER_TRAIN_DATASET_SIZE = 0.90
MODEL_BUILDER_TEST_DATASET_SIZE = 0.10
# To indicate if the dataset should be balanced using Over Sampling technique
PERFORM_OVERSAMPLING = False

# TPOT construction, there are other possibles parameters to configure, please,
# refer to the TPOT documentation in order to modify or add new ones http://rhiever.github.io/tpot/
TPOT_NUMBER_OF_GENERATIONS = 2
TPOP_POPULATION_SIZE = 2
# If -1, then will use all available cpu resources in your machine
TPOP_NUMBER_OF_JOBS = -1
TPOP_LOG_VERBOSITY = 2
