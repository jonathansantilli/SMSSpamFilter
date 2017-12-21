from spamdetector import app
from spamdetector.dataset import Dataset
from spamdetector.log import logging
from spamdetector import factory
from spamdetector import configuration as config


class PipelineSelector:
    """
    This class is in charge of generating or detecting the best Pipeline using the provided configuration
    """
    def __init__(self, dataset:Dataset, metaclassifier):
        self.dataset = dataset
        self.metaclassifier = metaclassifier

    def select(self) -> None:
        """
        This method will perform the operations that optimize machine learning pipelines using genetic programming.
        NOTE: Depending on the configuration used to build the metaclassifier,
              this method could take a long time to execute
        """
        dataset_features, labels_by_document_example = self.dataset.get_features_and_labels()

        X = self.dataset.vectorize(dataset_features)
        y = labels_by_document_example

        X_train, X_test, y_train, y_test = self.dataset.split_train_test_dataset(X, y,
            train_size=config.MODEL_SELECTOR_TRAIN_DATASET_SIZE,
            test_size=config.MODEL_SELECTOR_TEST_DATASET_SIZE)

        self._start_best_pipeline_selection(X_train, y_train)

        self._export_best_pipeline(config.BEST_PIPELINE_PYTHON_FILENAME)

    def _start_best_pipeline_selection(self, X_train, y_train) -> None:
        """
        Fit the metaclassifier to find the best Pipeline
        """
        self.metaclassifier.fit(X_train, y_train)

    def _export_best_pipeline(self, pipeline_python_filename) -> None:
        """
        Export the file (python code) to the desired destination
        """
        self.metaclassifier.export(pipeline_python_filename)


if __name__ == '__main__':
    app.select_pipeline()
