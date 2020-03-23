from aitoolbox.experiment.core_metrics.abstract_metric import AbstractBaseMetric

from sklearn.metrics import mean_squared_error, mean_absolute_error


class MeanSquaredErrorMetric(AbstractBaseMetric):
    def __init__(self, y_true, y_predicted):
        """Model prediction MSE

        Args:
            y_true (numpy.array or list): ground truth targets
            y_predicted (numpy.array or list): predicted targets
        """
        AbstractBaseMetric.__init__(self, y_true, y_predicted, metric_name='Mean_squared_error')

    def calculate_metric(self):
        return mean_squared_error(self.y_true, self.y_predicted)


class MeanAbsoluteErrorMetric(AbstractBaseMetric):
    def __init__(self, y_true, y_predicted):
        """Model prediction MAE

        Args:
            y_true (numpy.array or list): ground truth targets
            y_predicted (numpy.array or list): predicted targets
        """
        AbstractBaseMetric.__init__(self, y_true, y_predicted, metric_name='Mean_absolute_error')

    def calculate_metric(self):
        return mean_absolute_error(self.y_true, self.y_predicted)
