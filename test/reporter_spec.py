from expects import expect, equal
from doublex import Spy
from doublex_expects import have_been_called_with

from spamdetector import factory
from spamdetector.reporter import Reporter


with description('Reporter'):
    with before.all:
        self.y_test = ['category_1', 'category_2', 'category_1']
        self.y_predict = ['category_1', 'category_1', 'category_2']

    with it('should show the accuracy report'):
        with Spy() as accuracy:
            accuracy.score(self.y_test, self.y_predict).returns(0.99)
        reporter = Reporter(accuracy.score, None, None)

        reporter.show_accuracy_score(self.y_test, self.y_predict)

        expect(accuracy.score).to(have_been_called_with(self.y_test, self.y_predict))

    with it('should show the precision, recall and f1_score report'):
        with Spy() as metric:
            metric.classification_report(self.y_test, self.y_predict).returns('')
        reporter = Reporter(None, metric.classification_report, None)

        reporter.show_precision_recall_and_f1_score(self.y_test, self.y_predict)

        expect(metric.classification_report).to(have_been_called_with(self.y_test, self.y_predict))

    with it('should show the confusion matrix report'):
        with Spy() as metric:
            cm = [[1, 0], [0, 1]]
            metric.confusion_matrix(self.y_test,
                self.y_predict, labels=['category_1', 'category_2']).returns(cm)
        reporter = Reporter(None, None, metric.confusion_matrix)

        reporter.show_confusion_matrix(self.y_test, self.y_predict)

        expect(metric.confusion_matrix).to(have_been_called_with(
            self.y_test, self.y_predict, labels=['category_1', 'category_2']))
