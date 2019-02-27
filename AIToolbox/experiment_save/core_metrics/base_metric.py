from abc import ABC, abstractmethod

import numpy as np


class AbstractBaseMetric(ABC):
    def __init__(self, y_true, y_predicted, np_array=True):
        """

        Args:
            y_true (numpy.array or list or str):
            y_predicted (numpy.array or list or str):
            np_array (bool):
        """
        self.y_true = np.array(y_true) if np_array else y_true
        self.y_predicted = np.array(y_predicted) if np_array else y_predicted
        self.metric_name = None
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
