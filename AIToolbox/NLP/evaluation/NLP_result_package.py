from AIToolbox.experiment_save.result_package import AbstractResultPackage
from AIToolbox.experiment_save.core_metrics.classification import AccuracyMetric
from AIToolbox.NLP.evaluation.NLP_metrics import ROGUEMetric, ROGUENonOfficialMetric, \
    BLEUCorpusScoreMetric, PerplexityMetric


class QuestionAnswerResultPackage(AbstractResultPackage):
    def __init__(self, paragraph_text_tokens, strict_content_check=False, **kwargs):
        """

        Args:
            paragraph_text_tokens (list):
            strict_content_check (bool):
            **kwargs (dict):
        """
        self.paragraph_text_tokens = paragraph_text_tokens
        AbstractResultPackage.__init__(self, strict_content_check, **kwargs)

    def prepare_results_dict(self):
        """

        Returns:

        """
        # raise NotImplementedError

        y_span_start_true = self.y_true[:, 0]
        y_span_start_predicted = [el[0] for el in self.y_predicted]
        y_span_end_true = self.y_true[:, 1]
        y_span_end_predicted = [el[1] for el in self.y_predicted]

        true_text = [paragraph_tex[start_span:end_span + 1]
                     for start_span, end_span, paragraph_tex in
                     zip(y_span_start_true, y_span_end_true, self.paragraph_text_tokens)]

        pred_text = [paragraph_tex[start_span:end_span + 1]
                     for start_span, end_span, paragraph_tex in
                     zip(y_span_start_predicted, y_span_end_predicted, self.paragraph_text_tokens)]

        rogue_metric = ROGUENonOfficialMetric(true_text, pred_text)

        # TODO: implement ROGUEMetric where the official ROGUE calculation script is used

        self.results_dict = rogue_metric


class QuestionAnswerSpanClassificationResultPackage(AbstractResultPackage):
    def __init__(self, strict_content_check=False, **kwargs):
        """

        Args:
            strict_content_check (bool):
            **kwargs (dict):
        """
        # self.y_span_start_true = y_span_start_true
        # self.y_span_end_true = y_span_end_true
        # self.y_span_start_predicted = y_span_start_predicted
        # self.y_span_end_predicted = y_span_end_predicted
        AbstractResultPackage.__init__(self, strict_content_check, **kwargs)

    def prepare_results_dict(self):
        """

        Available general data:

        y_span_start_true (numpy.array or list):
        y_span_start_predicted (numpy.array or list):
        y_span_end_true (numpy.array or list):
        y_span_end_predicted (numpy.array or list):
        strict_content_check (bool):
        **kwargs (dict):

        Returns:

        """

        raise NotImplementedError


        y_span_start_true = self.y_true[:, 0]
        y_span_start_predicted = [el[0] for el in self.y_predicted]
        y_span_end_true = self.y_true[:, 1]
        y_span_end_predicted = [el[1] for el in self.y_predicted]

        # span_start_accuracy = AccuracyMetric(self.y_span_start_true, self.y_span_start_predicted)
        # span_start_accuracy.metric_name += '_span_start'
        # span_end_accuracy = AccuracyMetric(self.y_span_end_true, self.y_span_end_predicted)
        # span_end_accuracy.metric_name += '_span_end'

        span_start_accuracy = AccuracyMetric(y_span_start_true, y_span_start_predicted)
        span_start_accuracy.metric_name += '_span_start'
        span_end_accuracy = AccuracyMetric(y_span_end_true, y_span_end_predicted)
        span_end_accuracy.metric_name += '_span_end'

        span_start_accuracy_result = span_start_accuracy.get_metric_dict()
        span_end_accuracy_result = span_end_accuracy.get_metric_dict()

        self.results_dict = {**span_start_accuracy_result, **span_end_accuracy_result}


class TextSummarizationResultPackage(AbstractResultPackage):
    def __init__(self, strict_content_check=False, **kwargs):
        """

        Args:
            strict_content_check (bool):
            **kwargs (dict):
        """
        AbstractResultPackage.__init__(self, strict_content_check, **kwargs)

    def prepare_results_dict(self):
        """

        Returns:

        """
        # rogue_result = ROGUEMetric(self.y_true, self.y_predicted).get_metric_dict()

        raise NotImplementedError


class MachineTranslationResultPackage(AbstractResultPackage):
    def __init__(self, strict_content_check=False, **kwargs):
        """

        Args:
            strict_content_check (bool):
            **kwargs (dict):
        """
        AbstractResultPackage.__init__(self, strict_content_check, **kwargs)

    def prepare_results_dict(self):
        """

        Returns:

        """
        # bleu_result = BLEUCorpusScoreMetric(self.y_true, self.y_predicted).get_metric_dict()
        # perplexity_result = PerplexityMetric(self.y_true, self.y_predicted).get_metric_dict()

        raise NotImplementedError
