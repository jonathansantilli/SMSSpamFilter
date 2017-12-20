from os import path

from expects import expect, equal
from doublex import Stub

from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split

from spamdetector.dataset import Dataset


with description('Dataset'):
    with before.all:
        self.dataset_path = path.dirname(path.abspath(__file__)) + '/data/spam_ham_examples.txt'

    with it('should return features and labels array'):
        dataset = Dataset(file_path=self.dataset_path,
            vectorizer=Stub(), dataset_splitter=Stub())

        dataset_features, labels_by_document_example = dataset.get_features_and_labels()

        expect(len(dataset_features)).to(equal(30))
        expect(labels_by_document_example.count('ham')).to(equal(18))
        expect(labels_by_document_example.count('spam')).to(equal(12))

    with it('should vectorize the key:value (dict) features'):
        dataset = Dataset(file_path=self.dataset_path,
            vectorizer=DictVectorizer(), dataset_splitter=Stub())
        dataset_features, labels_by_document_example = dataset.get_features_and_labels()

        vectorized_features = dataset.vectorize(dataset_features)

        expect(vectorized_features.shape).to(equal((30, 154)))

    with it('should split the dataset in training and test sets'):
        dataset = Dataset(file_path=self.dataset_path,
            vectorizer=DictVectorizer(), dataset_splitter=train_test_split)
        dataset_features, labels_by_document_example = dataset.get_features_and_labels()
        X = dataset.vectorize(dataset_features)
        y = labels_by_document_example

        X_train, X_test, y_train, y_test = dataset.split_train_test_dataset(X, y,
            train_size=0.9,
            test_size=0.1)

        expect(X_train.shape).to(equal((27, 154)))
        expect(X_test.shape).to(equal((3, 154)))
        expect(len(y_train)).to(equal(27))
        expect(len(y_test)).to(equal(3))
