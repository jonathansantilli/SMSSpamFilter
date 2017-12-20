import sys

from spamdetector import factory
from spamdetector.log import logging
from spamdetector.file_helper import FileHelper


def build_model():
    """
    Main entry point to build the classification model using the selected Pipeline
    """
    dataset_path = get_dataset_filepath()
    dataset = factory.create_dataset(dataset_path=dataset_path)
    model_builder = factory.create_model_builder(dataset=dataset)

    dataset_features, labels_by_document_example = dataset.get_features_and_labels()
    X = dataset.vectorize(dataset_features)
    y = labels_by_document_example

    model_builder.build(X, y)


def select_pipeline():
    """
    Main entry point that run the code to perform the Pipeline selection
    """
    dataset_path = get_dataset_filepath()
    dataset = factory.create_dataset(dataset_path=dataset_path)
    pipeline_selector = factory.create_pipeline_selector(dataset=dataset)

    pipeline_selector.select()

def get_dataset_filepath():
    """
    Return the path to the dataset file obtained from the command line as a parameter

    :param str: Dataset file path
    """
    def get_dataset_file_parameter():
        if len(sys.argv) > 1:
            return sys.argv[1]

        return ''

    dataset_path = get_dataset_file_parameter()
    if not dataset_path or not FileHelper().exist_path(dataset_path):
        logging.error('Please, provide a valid dataset file path')
        sys.exit(-1)

    return dataset_path
