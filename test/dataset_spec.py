from os import path

from expects import expect, equal
from doublex import Stub

from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split

from spamdetector.dataset import Dataset


with description('Dataset'):
    with before.all:
        self.dataset_path = path.dirname(path.abspath(__file__)) + '/data/spam_ham_examples.txt'
        self.TOTAL_NUMBER_OF_DOCS = 30
        self.NUMBER_OF_FEATURES_TOKENS = 154

    with it('should return features and labels array'):
        NUMBER_OF_HAM_DOCS = 18
        NUMBER_OF_SPAM_DOCS = 12
        dataset = Dataset(file_path=self.dataset_path, vectorizer=Stub(), dataset_splitter=Stub())

        dataset_features, labels_by_document_example = dataset.get_features_and_labels()

        expect(len(dataset_features)).to(equal(self.TOTAL_NUMBER_OF_DOCS))
        expect(labels_by_document_example.count('ham')).to(equal(NUMBER_OF_HAM_DOCS))
        expect(labels_by_document_example.count('spam')).to(equal(NUMBER_OF_SPAM_DOCS))

    with it('should vectorize the key:value (dict) features'):
        NUMBER_OF_FEATURES_TOKENS = 154
        dataset = Dataset(file_path=self.dataset_path,
            vectorizer=DictVectorizer(), dataset_splitter=Stub())
        dataset_features, labels_by_document_example = dataset.get_features_and_labels()

        vectorized_features = dataset.vectorize(dataset_features)

        expect(vectorized_features.shape).to(equal(
            (self.TOTAL_NUMBER_OF_DOCS, self.NUMBER_OF_FEATURES_TOKENS)))

    with it('should split the dataset in training and test sets'):
        NUMBERS_OF_DOCS_FOR_TRAINING = 27
        NUMBERS_OF_DOCS_FOR_TESTING = 3
        dataset = Dataset(file_path=self.dataset_path,
            vectorizer=DictVectorizer(), dataset_splitter=train_test_split)
        dataset_features, labels_by_document_example = dataset.get_features_and_labels()
        X = dataset.vectorize(dataset_features)
        y = labels_by_document_example

        X_train, X_test, y_train, y_test = dataset.split_train_test_dataset(X, y,
            train_size=0.9,
            test_size=0.1)

        expect(X_train.shape).to(equal((NUMBERS_OF_DOCS_FOR_TRAINING, self.NUMBER_OF_FEATURES_TOKENS)))
        expect(X_test.shape).to(equal((NUMBERS_OF_DOCS_FOR_TESTING, self.NUMBER_OF_FEATURES_TOKENS)))
        expect(len(y_train)).to(equal(NUMBERS_OF_DOCS_FOR_TRAINING))
        expect(len(y_test)).to(equal(NUMBERS_OF_DOCS_FOR_TESTING))
