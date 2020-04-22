from abc import ABC, abstractmethod

import numpy as np


class AbstractBaseMetric(ABC):
    def __init__(self, y_true, y_predicted, metric_name, np_array=True):
        """Base metric with core metric functionality needed by all the derived actual performance metrics

        Args:
            y_true (numpy.array or list or str): ground truth targets
            y_predicted (numpy.array or list or str): predicted targets
            metric_name (str): name of the calculated metric
            np_array (bool): should the provided targets be converted to numpy array or left as they are
        """
        self.y_true = np.array(y_true) if np_array else y_true
        self.y_predicted = np.array(y_predicted) if np_array else y_predicted
        self.metric_name = metric_name
        self.metric_result = None

        self.metric_result = self.calculate_metric()

    @abstractmethod
    def calculate_metric(self):
        """Perform metric calculation and return it from this function
        
        Returns:
            float or dict: return metric_result
        """
        pass

    def get_metric(self):
        """Returns metric result

        Returns:
            float or dict: return metric_result
        """
        return self.metric_result

    def get_metric_dict(self):
        """Creates and return metric result key-value dict

        Returns:
            dict: metric dict
        """
        return {self.metric_name: self.metric_result}

    def __str__(self):
        return f'{self.metric_name}: {self.metric_result}'

    def __len__(self):
        if type(self.metric_result) is list or type(self.metric_result) is dict:
            return len(self.metric_result)
        elif type(self.metric_result) is float or type(self.metric_result) is int:
            return 1
        elif self.metric_result is None:
            return 0
        else:
            raise TypeError(f'Current metric_result does not support len evaluation. '
                            f'The type is: {type(self.metric_result)}. '
                            f'Optimally it should be either a list, dict, float or int.')

    def __lt__(self, other):
        self_val, other_val = self._get_metric_self_other_val(other)
        return self_val < other_val

    def __le__(self, other):
        self_val, other_val = self._get_metric_self_other_val(other)
        return self_val <= other_val

    def __gt__(self, other):
        self_val, other_val = self._get_metric_self_other_val(other)
        return self_val > other_val

    def __ge__(self, other):
        self_val, other_val = self._get_metric_self_other_val(other)
        return self_val >= other_val

    def _get_metric_self_other_val(self, other):
        """Metric comparison prep util

        Args:
            other (aitoolbox.experiment.core_metrics.abstract_metric.AbstractBaseMetric or float or int): other compared
                metric

        Returns:
            float or int: metric value
        """
        if type(self.metric_result) is float or type(self.metric_result) is int:
            if isinstance(other, AbstractBaseMetric) and \
                    (type(other.metric_result) is float or type(other.metric_result) is int):
                return self.metric_result, other.metric_result
            elif type(other) is float or type(other) is int:
                return self.metric_result, other

        raise ValueError('Can do comparison only on metrics where metric_result is int of float')

    def __contains__(self, item):
        if item == self.metric_name:
            return True
        if type(self.metric_result) is dict:
            if item in self.metric_result:
                return True
        return False

    def __getitem__(self, item):
        if item == self.metric_name:
            return self.metric_result
        if type(self.metric_result) is dict:
            if item in self.metric_result:
                return self.metric_result[item]
        raise KeyError(f'Key {item} not found')

    def __add__(self, other):
        """Concatenate two metrics

        Args:
            other (AbstractBaseMetric or dict): new metric to be added

        Returns:
            dict: combined metric dict
        """
        return self.concat_metric(other)

    def __radd__(self, other):
        """Append another metric

        Args:
            other (AbstractBaseMetric or dict): new metric to be added

        Returns:
            dict: combined metric dict
        """
        return self.concat_metric(other)

    def concat_metric(self, other):
        """Concatenate another metric to this one

        Args:
            other (AbstractBaseMetric or dict): new metric to be added

        Returns:
            dict: combined metric dict
        """
        if isinstance(other, AbstractBaseMetric):
            other_metric_dict = other.get_metric_dict()
        elif type(other) == dict:
            other_metric_dict = other
        else:
            raise TypeError(f'Provided other not the accepted type. Provided type is: {type(other)}')

        return {**self.get_metric_dict(), **other_metric_dict}
