import numpy as np

from spamdetector import app
from spamdetector.log import logging
from spamdetector import configuration as config
from spamdetector.dataset import Dataset
from spamdetector.reporter import Reporter

"""
This class is core into the project.
Have the responsibility to create the classification model and show the accuracy report
"""
class ModelBuilder:
    def __init__(self, dataset:Dataset, reporter:Reporter, pipeline, over_sampler):
        self.dataset = dataset
        self.pipeline = pipeline
        self.reporter = reporter
        self.over_sampler = over_sampler

    def build(self, X, y):
        """
        Build the classification model using the provided dataset and pipeline

        :param X: Array, The document features
        :param y: Array, The list of categories that belongs to the documents
        :return pipeline: The fitted pipeline
        """
        X_train, X_test, y_train, y_test = self.dataset.split_train_test_dataset(X, y,
            train_size=config.MODEL_BUILDER_TRAIN_DATASET_SIZE,
            test_size=config.MODEL_BUILDER_TEST_DATASET_SIZE)

        if self.over_sampler:
            X_train, y_train = self._over_sample(X_train, y_train)

        self._fit(X_train, y_train)

        self._report(X_test, y_test)

        return self.pipeline

    def _over_sample(self, X, y) -> list:
        """
        Perform oversampling technique to the provided documents

        :param X: Array, The document features array
        :param y: Array, The list of categories that belongs to the documents
        :return Array: The oversampled features and categories, in that order
        """
        logging.info('Over sampling...')
        X_over_sample, y_over_sample = self.over_sampler.fit_sample(X.toarray(), np.array(y))

        return X_over_sample, y_over_sample

    def _fit(self, X, y):
        """
        Fit the provided Pipeline

        :param X: Array, The document features
        :param y: Array, The list of categories that belongs to the documents
        :return pipeline: the fitted pipeline
        """
        logging.info('Transforming and fitting pipeline...')
        return self.pipeline.fit(X, y)

    def _report(self, X_test, y_test) -> None:
        """
        Show a accuray report in case an Reporter was provided

        :param X: Array, The test document features
        :param y: Array, The list of test categories that belongs to the test documents
        """
        if self.reporter:
            logging.info('Reporting...')
            y_predict = self.pipeline.predict(X_test)

            self.reporter.show_accuracy_score(y_test, y_predict)
            self.reporter.show_precision_recall_and_f1_score(y_test, y_predict)
            self.reporter.show_confusion_matrix(y_test, y_predict)


if __name__ == '__main__':
    app.build_model()
