import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, accuracy_score, f1_score, precision_score, recall_score

from aitoolbox.experiment.core_metrics.abstract_metric import AbstractBaseMetric


class AccuracyMetric(AbstractBaseMetric):
    def __init__(self, y_true, y_predicted, positive_class_thresh=0.5):
        """Model prediction accuracy

        Args:
            y_true (numpy.array or list): ground truth targets
            y_predicted (numpy.array or list): predicted targets
            positive_class_thresh (float or None): predicted probability positive class threshold.
                Set it to None when dealing with multi-class labels.
        """
        self.positive_class_thresh = positive_class_thresh
        AbstractBaseMetric.__init__(self, y_true, y_predicted, metric_name='Accuracy')

    def calculate_metric(self):
        if len(self.y_predicted.shape) > 1 and self.y_predicted.shape[1] > 1:
            self.y_predicted = np.argmax(self.y_predicted, axis=1)
        elif self.positive_class_thresh is not None:
            if np.min(self.y_predicted) >= 0. and np.max(self.y_predicted) <= 1.:
                self.y_predicted = self.y_predicted >= self.positive_class_thresh
            else:
                print('Thresholding the predicted probabilities as if they are binary. However, found'
                      'predicted value above 1.0. Threshold has not been applied.')

        if len(self.y_true.shape) > 1 and self.y_true.shape[1] > 1:
            self.y_true = np.argmax(self.y_true, axis=1)

        return accuracy_score(self.y_true, self.y_predicted)


class ROCAUCMetric(AbstractBaseMetric):
    def __init__(self, y_true, y_predicted):
        """Model prediction ROC-AUC

        Args:
            y_true (numpy.array or list): ground truth targets
            y_predicted (numpy.array or list): predicted targets
        """
        AbstractBaseMetric.__init__(self, y_true, y_predicted, metric_name='ROC_AUC')

    def calculate_metric(self):
        return roc_auc_score(self.y_true, self.y_predicted)


class PrecisionRecallCurveAUCMetric(AbstractBaseMetric):
    def __init__(self, y_true, y_predicted):
        """Model prediction PR-AUC

        Args:
            y_true (numpy.array or list): ground truth targets
            y_predicted (numpy.array or list): predicted targets
        """
        AbstractBaseMetric.__init__(self, y_true, y_predicted, metric_name='PrecisionRecall_AUC')

    def calculate_metric(self):
        precision, recall, thresholds = precision_recall_curve(self.y_true, self.y_predicted)
        return auc(recall, precision)


class F1ScoreMetric(AbstractBaseMetric):
    def __init__(self, y_true, y_predicted, positive_class_thresh=0.5):
        """Model prediction F1 score

        Args:
            y_true (numpy.array or list): ground truth targets
            y_predicted (numpy.array or list): predicted targets
            positive_class_thresh (float): predicted probability positive class threshold
        """
        self.positive_class_thresh = positive_class_thresh
        AbstractBaseMetric.__init__(self, y_true, y_predicted, metric_name='F1_score')

    def calculate_metric(self):
        y_label_predicted = np.where(self.y_predicted > self.positive_class_thresh, 1, 0)
        return f1_score(self.y_true, y_label_predicted)


class PrecisionMetric(AbstractBaseMetric):
    def __init__(self, y_true, y_predicted, positive_class_thresh=0.5):
        """Model prediction precision

        Args:
            y_true (numpy.array or list): ground truth targets
            y_predicted (numpy.array or list): predicted targets
            positive_class_thresh (float): predicted probability positive class threshold
        """
        self.positive_class_thresh = positive_class_thresh
        AbstractBaseMetric.__init__(self, y_true, y_predicted, metric_name='Precision')

    def calculate_metric(self):
        y_label_predicted = np.where(self.y_predicted > self.positive_class_thresh, 1, 0)
        return precision_score(self.y_true, y_label_predicted)


class RecallMetric(AbstractBaseMetric):
    def __init__(self, y_true, y_predicted, positive_class_thresh=0.5):
        """Model prediction recall score

        Args:
            y_true (numpy.array or list): ground truth targets
            y_predicted (numpy.array or list): predicted targets
            positive_class_thresh (float): predicted probability positive class threshold
        """
        self.positive_class_thresh = positive_class_thresh
        AbstractBaseMetric.__init__(self, y_true, y_predicted, metric_name='Recall')

    def calculate_metric(self):
        y_label_predicted = np.where(self.y_predicted > self.positive_class_thresh, 1, 0)
        return recall_score(self.y_true, y_label_predicted)
