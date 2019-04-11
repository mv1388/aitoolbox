from AIToolbox.experiment_save.core_metrics.abstract_metric import AbstractBaseMetric

from sklearn.metrics import mean_squared_error, mean_absolute_error


class MeanSquaredErrorMetric(AbstractBaseMetric):
    def __init__(self, y_true, y_predicted):
        """

        Args:
            y_true (numpy.array or list):
            y_predicted (numpy.array or list):

        """
        AbstractBaseMetric.__init__(self, y_true, y_predicted, metric_name='Mean_squared_error')

    def calculate_metric(self):
        self.metric_result = mean_squared_error(self.y_true, self.y_predicted)


class MeanAbsoluteErrorMetric(AbstractBaseMetric):
    def __init__(self, y_true, y_predicted):
        """

        Args:
            y_true (numpy.array or list):
            y_predicted (numpy.array or list):

        """
        AbstractBaseMetric.__init__(self, y_true, y_predicted, metric_name='Mean_absolute_error')

    def calculate_metric(self):
        self.metric_result = mean_absolute_error(self.y_true, self.y_predicted)
