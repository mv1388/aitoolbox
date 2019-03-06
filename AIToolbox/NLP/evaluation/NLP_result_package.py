import os

from AIToolbox.experiment_save.result_package import AbstractResultPackage
from AIToolbox.experiment_save.core_metrics.classification import AccuracyMetric
from AIToolbox.NLP.evaluation.NLP_metrics import ROUGEMetric, ROUGENonPerlMetric, \
    BLEUCorpusScoreMetric, PerplexityMetric


class QuestionAnswerResultPackage(AbstractResultPackage):
    def __init__(self, paragraph_text_tokens, output_text_dir=None, target_actual_text=None,
                 use_perl_rouge=True,
                 strict_content_check=False, **kwargs):
        """

        Args:
            paragraph_text_tokens (list):
            output_text_dir (str):
            target_actual_text (list or None):
            use_perl_rouge (bool):
            strict_content_check (bool):
            **kwargs (dict):
        """
        if use_perl_rouge is True and output_text_dir is None:
            raise ValueError('When using the perl based ROUGE definition the output_text_dir path must be given.')
        if target_actual_text is not None:
            if len(paragraph_text_tokens) != len(target_actual_text):
                raise ValueError('paragraph_text_tokens size not the same as target_actual_text.')

        # todo: check if this is efficient
        self.paragraph_text_tokens = [[str(w) for w in paragraph] for paragraph in paragraph_text_tokens]
        self.target_actual_text = target_actual_text
        self.use_target_actual_text = target_actual_text is not None

        self.output_text_dir = os.path.expanduser(output_text_dir) if output_text_dir else None
        self.use_perl_rouge = use_perl_rouge
        AbstractResultPackage.__init__(self, strict_content_check, **kwargs)

    def prepare_results_dict(self):
        """

        Returns:

        """
        y_span_start_true = self.y_true[:, 0]
        y_span_start_predicted = self.y_predicted[:, 0]
        y_span_end_true = self.y_true[:, 1]
        y_span_end_predicted = self.y_predicted[:, 1]
        
        if self.use_target_actual_text:
            true_text = self.target_actual_text
        else:
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

        if self.use_perl_rouge:
            rogue_metric = ROUGEMetric(true_text, pred_text, self.output_text_dir,
                                       target_actual_text=self.use_target_actual_text)
        else:
            rogue_metric = ROUGENonPerlMetric(true_text, pred_text, target_actual_text=self.use_target_actual_text)

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
        # rogue_result = ROUGEMetric(self.y_true, self.y_predicted).get_metric_dict()

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
