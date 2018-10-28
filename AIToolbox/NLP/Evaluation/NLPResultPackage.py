from AIToolbox.ExperimentSave.ResultPackage import AbstractResultPackage
from AIToolbox.NLP.Evaluation.NLPMetrics import BLEUScoreMetric, PerplexityMetric, ROGUEMetric


class QuestionAnswerResultPackage(AbstractResultPackage):
    def __init__(self, y_true, y_predicted, hyperparameters=None, strict_content_check=False):
        """

        Args:
            y_true (numpy.array or list):
            y_predicted (numpy.array or list):
            hyperparameters (dict):
            strict_content_check (bool):
        """
        AbstractResultPackage.__init__(self, y_true, y_predicted, hyperparameters, strict_content_check)

    def prepare_results_dict(self):
        """

        Returns:

        """
        raise NotImplementedError


class TextSummarizationResultPackage(AbstractResultPackage):
    def __init__(self, y_true, y_predicted, hyperparameters=None, strict_content_check=False):
        """

        Args:
            y_true (numpy.array or list):
            y_predicted (numpy.array or list):
            hyperparameters (dict):
            strict_content_check (bool):
        """
        AbstractResultPackage.__init__(self, y_true, y_predicted, hyperparameters, strict_content_check)

    def prepare_results_dict(self):
        """

        Returns:

        """
        # rogue_result = ROGUEMetric(self.y_true, self.y_predicted).get_metric_dict()

        raise NotImplementedError


class MachineTranslationResultPackage(AbstractResultPackage):
    def __init__(self, y_true, y_predicted, hyperparameters=None, strict_content_check=False):
        """

        Args:
            y_true (numpy.array or list):
            y_predicted (numpy.array or list):
            hyperparameters (dict):
            strict_content_check (bool):
        """
        AbstractResultPackage.__init__(self, y_true, y_predicted, hyperparameters, strict_content_check)

    def prepare_results_dict(self):
        """

        Returns:

        """
        # bleu_result = BLEUScoreMetric(self.y_true, self.y_predicted).get_metric_dict()
        # perplexity_result = PerplexityMetric(self.y_true, self.y_predicted).get_metric_dict()

        raise NotImplementedError
