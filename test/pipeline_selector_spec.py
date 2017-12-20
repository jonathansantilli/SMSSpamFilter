from os import path

from expects import expect, equal
from doublex import Spy
from doublex_expects import have_been_called

from spamdetector import factory
from spamdetector.pipeline_selector import PipelineSelector


with description('PipelineSelector'):
    with before.all:
        self.spam_ham_examples = path.dirname(path.abspath(__file__)) + '/data/spam_ham_examples.txt'

    with it('should generate a Pipeline'):
        dataset = factory.create_dataset(dataset_path=self.spam_ham_examples)
        metaclassifier = Spy()
        pipeline_selector = PipelineSelector(dataset=dataset, metaclassifier=metaclassifier)
        dataset_features, labels_by_document_example = dataset.get_features_and_labels()
        X = dataset.vectorize(dataset_features)
        y = labels_by_document_example

        pipeline_selector.select()

        expect(metaclassifier.fit).to(have_been_called.once)
        expect(metaclassifier.export).to(have_been_called.once)
