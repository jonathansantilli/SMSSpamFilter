from spamdetector.log import logging


"""
The purpose of this class is to show the performance of the generated model
"""
class Reporter:
    def __init__(self, accuracy_score, classification_report, confusion_matrix):
        self.accuracy_score = accuracy_score
        self.classification_report = classification_report
        self.confusion_matrix = confusion_matrix

    def show_accuracy_score(self, y_test, y_predict):
        accuracy = self.accuracy_score(y_test, y_predict)
        logging.info("Accuracy score: {}".format(accuracy))

    def show_precision_recall_and_f1_score(self, y_test, y_predict):
        logging.info(self.classification_report(y_test, y_predict))

    def show_confusion_matrix(self, y_test, y_predict):
        labels = sorted(list(set(y_test)))
        cm = self.confusion_matrix(y_test, y_predict, labels=labels)

        logging.info('  Confusion Matrix')
        logging.info('       ' + str(labels[0]) + ' ' + str(labels[1]))
        logging.info('  ' + str(labels[0]) + '   ' + str(cm[0][0]) + ' ' + str(cm[0][1]))
        logging.info('  ' + str(labels[1]) + '   ' + str(cm[1][0]) + ' ' + str(cm[1][1]))
