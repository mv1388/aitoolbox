from abc import ABC, abstractmethod

import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score


class AbstractMetric(ABC):
    def __init__(self, y_true, y_predicted):
        """

        Args:
            y_true (numpy.core.multiarray or list):
            y_predicted (numpy.core.multiarray or list):
        """
        self.metric_name = None
        self.y_true = np.array(y_true)
        self.y_predicted = np.array(y_predicted)
        self.metric_result = None

        self.calculate_metric()

    def get_metric(self):
        """

        Returns:
            float:
        """
        return self.metric_result

    def get_metric_dict(self):
        """

        Returns:
            dict:
        """
        return {self.metric_name: self.metric_result}

    @abstractmethod
    def calculate_metric(self):
        pass


class AccuracyMetric(AbstractMetric):
    def __init__(self, y_true, y_predicted, y_predicted_label_thresh=0.5):
        """

        Args:
            y_true:
            y_predicted:
            y_predicted_label_thresh:
        """
        self.y_predicted_label_thresh = y_predicted_label_thresh
        AbstractMetric.__init__(self, y_true, y_predicted)
        self.metric_name = 'accuracy'

    def calculate_metric(self):
        y_label_predicted = np.where(self.y_predicted > self.y_predicted_label_thresh, 1, 0)
        self.metric_result = accuracy_score(self.y_true, y_label_predicted)


class ROCAUCMetric(AbstractMetric):
    def __init__(self, y_true, y_predicted):
        """

        Args:
            y_true (numpy.core.multiarray or list):
            y_predicted (numpy.core.multiarray or list):
        """
        AbstractMetric.__init__(self, y_true, y_predicted)
        self.metric_name = 'ROC_AUC'

    def calculate_metric(self):
        self.metric_result = roc_auc_score(self.y_true, self.y_predicted)


class PrecisionRecallCurveAUCMetric(AbstractMetric):
    def __init__(self, y_true, y_predicted):
        """

        Args:
            y_true (numpy.core.multiarray or list):
            y_predicted (numpy.core.multiarray or list):
        """
        AbstractMetric.__init__(self, y_true, y_predicted)
        self.metric_name = 'PrecisionRecall_curve_AUC'

    def calculate_metric(self):
        raise NotImplementedError
