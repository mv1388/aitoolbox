from AIToolbox.experiment_save.result_package import AbstractResultPackage
from AIToolbox.experiment_save.core_metrics.classification import AccuracyMetric
from AIToolbox.NLP.evaluation.NLP_metrics import ROGUEMetric, ROGUENonOfficialMetric, \
    BLEUCorpusScoreMetric, PerplexityMetric


class QuestionAnswerResultPackage(AbstractResultPackage):
    def __init__(self, paragraph_text_tokens, output_text_dir, strict_content_check=False, **kwargs):
        """

        Args:
            paragraph_text_tokens (list):
            output_text_dir (str):
            strict_content_check (bool):
            **kwargs (dict):
        """
        # self.paragraph_text_tokens = paragraph_text_tokens
        self.paragraph_text_tokens = [[str(w) for w in paragraph] for paragraph in paragraph_text_tokens]
        self.output_text_dir = output_text_dir
        AbstractResultPackage.__init__(self, strict_content_check, **kwargs)

    def prepare_results_dict(self):
        """

        Returns:

        """
        y_span_start_true = self.y_true[:, 0]
        y_span_start_predicted = self.y_predicted[:, 0]
        y_span_end_true = self.y_true[:, 1]
        y_span_end_predicted = self.y_predicted[:, 1]

        true_text = [paragraph_text[start_span:end_span + 1]
                     for start_span, end_span, paragraph_text in
                     zip(y_span_start_true.astype('int'), y_span_end_true.astype('int'), self.paragraph_text_tokens)]

        pred_text = [paragraph_text[start_span:end_span + 1]
                     for start_span, end_span, paragraph_text in
                     zip(y_span_start_predicted.astype('int'), y_span_end_predicted.astype('int'), self.paragraph_text_tokens)]

        # Just for testing
        # pred_text = [paragraph_text[start_span:end_span + 2]
        #              for start_span, end_span, paragraph_text in
        #              zip(y_span_start_true.astype('int'), y_span_end_true.astype('int'), self.paragraph_text_tokens)]

        rogue_metric = ROGUEMetric(true_text, pred_text, self.output_text_dir)
        # rogue_metric_non_official = ROGUENonOfficialMetric(true_text, pred_text)
        # self.results_dict = {**rogue_metric, **rogue_metric_non_official}

        self.results_dict = rogue_metric.get_metric_dict()


class QuestionAnswerSpanClassificationResultPackage(AbstractResultPackage):
    def __init__(self, strict_content_check=False, **kwargs):
        """

        Args:
            strict_content_check (bool):
            **kwargs (dict):
        """
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
        y_span_start_true = self.y_true[:, 0]
        y_span_start_predicted = self.y_predicted[:, 0]
        y_span_end_true = self.y_true[:, 1]
        y_span_end_predicted = self.y_predicted[:, 1]

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
