from spamdetector.log import logging
from spamdetector.file_helper import FileHelper
from spamdetector.document import Document
from spamdetector.document import InvalidDocumentTypeException
from spamdetector import configuration as config


class Dataset:
    """
    This class is responsible for the operations related to the text dataset
    """
    def __init__(self, file_path:str, vectorizer, dataset_splitter):
        self.file_path = file_path
        self.vectorizer = vectorizer
        self.dataset_splitter = dataset_splitter

    def get_features_and_labels(self) -> list:
        """
        Get the dataset features and the category labels by examples

        :return Array: First position a dict with features and the second
                       an array with the categories associated to the examples
        """
        logging.info('Getting features from dataset...')
        document_examples = self._read_documents_examples()
        dataset_features = []
        labels_by_document_example = []
        for document in document_examples:
            features_name_and_score = {}
            tokens = document.get_document_example_tokens_array()

            for token in tokens:
                # This just indicates that the feature is present
                features_name_and_score[token] = 1.0

            dataset_features.append(features_name_and_score)
            labels_by_document_example.append(document.get_document_type())

        return [dataset_features, labels_by_document_example]

    def vectorize(self, dataset_features) -> list:
        """
        Vectorize each key:value dict within the dataset_features

        :param dataset_features: List of the dictionaries that contains the feature:value for each token
        :return Array: A 2 dimentional array
        """
        logging.info('Vectorizing examples...')
        X = self.vectorizer.fit_transform(dataset_features)

        return X

    def split_train_test_dataset(self, X, y, train_size, test_size, random_state=config.DEFAULT_RANDOM_STATE) ->list:
        """
        Split the dataset into training and test data

        :param X: Array of examples
        :param y: Array of the categories for each example
        :param train_size: Training size portion desired to split the X dataset
        :param test_size: Test size portion desired to split the X dataset
        :param random_state: Optional random state to initialize the dataset splitter
        :return Array: The array with: training examples, test examples,
                       traning categories, testing categories, in that order
        """
        logging.info('Splitting dataset...')
        X_train, X_test, y_train, y_test = self.dataset_splitter(X, y,
            train_size=train_size,
            test_size=test_size,
            random_state=random_state)

        return X_train, X_test, y_train, y_test

    def _read_documents_examples(self) -> list:
        """
        Reads the dataset file and return an Array with the Document objects

        :return Array: a Document for each document example
        """
        tab_separated_examples = FileHelper().read_pattern_separated_file(self.file_path, '\t')
        document_examples = []
        for example in tab_separated_examples:
            try:
                document_examples.append(Document(document_type=example[0], document_data=example[1]))
            except InvalidDocumentTypeException as e:
                logging.error(str(e))

        return document_examples
