from AIToolbox.ExperimentSave.MetricsGeneral.BaseMetric import AbstractBaseMetric

import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, accuracy_score, f1_score, precision_score, recall_score


class AccuracyMetric(AbstractBaseMetric):
    def __init__(self, y_true, y_predicted, y_predicted_label_thresh=0.5):
        """

        Args:
            y_true (numpy.array or list):
            y_predicted (numpy.array or list):
            y_predicted_label_thresh (float):
        """
        self.y_predicted_label_thresh = y_predicted_label_thresh
        AbstractBaseMetric.__init__(self, y_true, y_predicted)
        self.metric_name = 'Accuracy'

    def calculate_metric(self):
        y_label_predicted = np.where(self.y_predicted > self.y_predicted_label_thresh, 1, 0)
        self.metric_result = accuracy_score(self.y_true, y_label_predicted)


class ROCAUCMetric(AbstractBaseMetric):
    def __init__(self, y_true, y_predicted):
        """

        Args:
            y_true (numpy.array or list):
            y_predicted (numpy.array or list):
        """
        AbstractBaseMetric.__init__(self, y_true, y_predicted)
        self.metric_name = 'ROC_AUC'

    def calculate_metric(self):
        self.metric_result = roc_auc_score(self.y_true, self.y_predicted)


class PrecisionRecallCurveAUCMetric(AbstractBaseMetric):
    def __init__(self, y_true, y_predicted):
        """

        Args:
            y_true (numpy.array or list):
            y_predicted (numpy.array or list):
        """
        AbstractBaseMetric.__init__(self, y_true, y_predicted)
        self.metric_name = 'PrecisionRecall_AUC'

    def calculate_metric(self):
        precision, recall, thresholds = precision_recall_curve(self.y_true, self.y_predicted)
        self.metric_result = auc(recall, precision)


class F1ScoreMetric(AbstractBaseMetric):
    def __init__(self, y_true, y_predicted, y_predicted_label_thresh=0.5):
        """

        Args:
            y_true (numpy.array or list):
            y_predicted (numpy.array or list):
        """
        self.y_predicted_label_thresh = y_predicted_label_thresh
        AbstractBaseMetric.__init__(self, y_true, y_predicted)
        self.metric_name = 'F1_score'

    def calculate_metric(self):
        y_label_predicted = np.where(self.y_predicted > self.y_predicted_label_thresh, 1, 0)
        self.metric_result = f1_score(self.y_true, y_label_predicted)


class PrecisionMetric(AbstractBaseMetric):
    def __init__(self, y_true, y_predicted, y_predicted_label_thresh=0.5):
        """

        Args:
            y_true (numpy.array or list):
            y_predicted (numpy.array or list):
        """
        self.y_predicted_label_thresh = y_predicted_label_thresh
        AbstractBaseMetric.__init__(self, y_true, y_predicted)
        self.metric_name = 'Precision'

    def calculate_metric(self):
        y_label_predicted = np.where(self.y_predicted > self.y_predicted_label_thresh, 1, 0)
        self.metric_result = precision_score(self.y_true, y_label_predicted)


class RecallMetric(AbstractBaseMetric):
    def __init__(self, y_true, y_predicted, y_predicted_label_thresh=0.5):
        """

        Args:
            y_true (numpy.array or list):
            y_predicted (numpy.array or list):
        """
        self.y_predicted_label_thresh = y_predicted_label_thresh
        AbstractBaseMetric.__init__(self, y_true, y_predicted)
        self.metric_name = 'Recall'

    def calculate_metric(self):
        y_label_predicted = np.where(self.y_predicted > self.y_predicted_label_thresh, 1, 0)
        self.metric_result = recall_score(self.y_true, y_label_predicted)
