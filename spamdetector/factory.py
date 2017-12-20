from imblearn.over_sampling import RandomOverSampler

from tpot import TPOTClassifier

from sklearn.feature_selection import RFE
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from spamdetector.model_builder import ModelBuilder
from spamdetector.pipeline_selector import PipelineSelector
from spamdetector.dataset import Dataset
from spamdetector.reporter import Reporter
from spamdetector import configuration as config

"""
The main purpose of this factory module is to provide an abstraction layer to the business.
This avoid leaking some detail implementation of the used libraries with the business logic
"""

def create_model_builder(dataset: Dataset):
    over_sampler = get_over_sampler_if_present()
    pipeline = get_pipeline()
    reporter = create_reporter()

    return ModelBuilder(dataset=dataset,
        pipeline=pipeline,
        over_sampler=over_sampler,
        reporter=reporter)

def get_over_sampler_if_present():
    over_sampler = None
    if config.PERFORM_OVERSAMPLING:
        over_sampler = RandomOverSampler(random_state=config.DEFAULT_RANDOM_STATE)

    return over_sampler

def create_reporter():
    return Reporter(accuracy_score, classification_report, confusion_matrix)

def get_pipeline():
    """
    Create a Pipeline. If you want to use another Pipeline (estimator, features selector, ...),
    just change the implementation of this code with ideally the pipeline suggested by TPOT after executing the
    spamdetector.model_selector module.

    After the execution of the spamdetector.model_selector module, a file with the selected Pipeline
    will be written to the configuration.BEST_PIPELINE_PYTHON_FILENAME, please check the configuration module.

    :return Pipeline: the Pipeline selected by TPOT
    """
    extra_trees_classifier = ExtraTreesClassifier(n_estimators=100,
        criterion='entropy', max_features=0.15, n_jobs=-1)
    rfe = RFE(extra_trees_classifier, step=0.35)
    naive_bayes = BernoulliNB(alpha=0.01, fit_prior=True)

    return Pipeline([('feature_selector', rfe), ('estimator', naive_bayes)])


def create_dataset(dataset_path):
    """
    Create a Dataset object using the provided dataset file path

    :param dataset_path: dataset file path
    :return Dataset: the dataset to be used
    """
    vectorizer = DictVectorizer()
    return Dataset(file_path=dataset_path,
        vectorizer=vectorizer, dataset_splitter=train_test_split)


def create_pipeline_selector(dataset:Dataset):
    """
    Create the PipelineSelector object responsible for executing the combinations
    in order to detect the best Pipeline according the provided configuration

    :param dataset: The Dataset object to be used by the PipelineSelector
    :return PipelineSelector: The Pipeline selector
    """
    tpot = TPOTClassifier(generations=config.TPOT_NUMBER_OF_GENERATIONS,
        population_size=config.TPOP_POPULATION_SIZE, n_jobs=config.TPOP_NUMBER_OF_JOBS,
        verbosity=config.TPOP_LOG_VERBOSITY, config_dict='TPOT sparse')
    pipeline_selector = PipelineSelector(dataset=dataset, metaclassifier=tpot)

    return pipeline_selector
