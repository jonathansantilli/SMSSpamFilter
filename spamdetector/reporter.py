from spamdetector.log import logging


class Reporter:
    """
    The purpose of this class is to show the performance of the generated model
    """
    def __init__(self, accuracy_score_function,
            classification_report_function, confusion_matrix_function):
        self.accuracy_score = accuracy_score_function
        self.classification_report = classification_report_function
        self.confusion_matrix = confusion_matrix_function

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
