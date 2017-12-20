from os import path

from expects import expect, equal, be_false, be_true

from spamdetector import factory


with description('ModelBuilder'):
    with before.all:
        self.spam_ham_examples = path.dirname(path.abspath(__file__)) + '/data/spam_ham_examples.txt'

    with it('should build a model'):
        expected_prediction =  ['ham', 'ham', 'ham', 'ham', 'ham', 'ham', 'ham', 'ham', 'ham', 'ham', 'ham',
            'ham', 'ham', 'ham', 'ham', 'ham', 'ham', 'ham', 'spam', 'spam', 'spam', 'spam', 'spam',
            'spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam']
        dataset = factory.create_dataset(dataset_path=self.spam_ham_examples)
        model_builder = factory.create_model_builder(dataset=dataset)
        dataset_features, labels_by_document_example = dataset.get_features_and_labels()
        X = dataset.vectorize(dataset_features)
        y = labels_by_document_example

        model = model_builder.build(X, y)
        predict = model.predict(X)

        expect(sorted(predict)).to(equal(sorted(expected_prediction)))
