from AIToolbox.ExperimentSave.MetricsGeneral.BaseMetric import AbstractBaseMetric


class ROGUEMetric(AbstractBaseMetric):
    def __init__(self, y_true, y_predicted):
        """

        Args:
            y_true (numpy.array or list):
            y_predicted (numpy.array or list):
        """
        AbstractBaseMetric.__init__(self, y_true, y_predicted)
        self.metric_name = 'ROGUE'

    def calculate_metric(self):
        raise NotImplementedError


class BLEUScoreMetric(AbstractBaseMetric):
    def __init__(self, y_true, y_predicted):
        """

        Args:
            y_true (numpy.array or list):
            y_predicted (numpy.array or list):
        """
        AbstractBaseMetric.__init__(self, y_true, y_predicted)
        self.metric_name = 'BLEU_score'

    def calculate_metric(self):
        raise NotImplementedError


class PerplexityMetric(AbstractBaseMetric):
    def __init__(self, y_true, y_predicted):
        """

        Args:
            y_true (numpy.array or list):
            y_predicted (numpy.array or list):
        """
        AbstractBaseMetric.__init__(self, y_true, y_predicted)
        self.metric_name = 'Perplexity'

    def calculate_metric(self):
        raise NotImplementedError
