import os

from AIToolbox.experiment_save.result_package.abstract_result_packages import AbstractResultPackage
from AIToolbox.experiment_save.core_metrics.classification import AccuracyMetric
from AIToolbox.NLP.evaluation.NLP_metrics import ROUGEMetric, ROUGEPerlMetric, BLEUCorpusScoreMetric, PerplexityMetric
from AIToolbox.NLP.evaluation.attention_heatmap import AttentionHeatMap


class QuestionAnswerResultPackage(AbstractResultPackage):
    def __init__(self, paragraph_text_tokens, target_actual_text=None, output_text_dir=None,
                 use_perl_rouge=False,
                 strict_content_check=False, **kwargs):
        """

        Args:
            paragraph_text_tokens (list):
            target_actual_text (list or None):
            output_text_dir (str):
            use_perl_rouge (bool):
            strict_content_check (bool):
            **kwargs (dict):
        """
        if use_perl_rouge is True and output_text_dir is None:
            raise ValueError('When using the perl based ROUGE definition the output_text_dir path must be given.')
        if target_actual_text is not None:
            if len(paragraph_text_tokens) != len(target_actual_text):
                raise ValueError('paragraph_text_tokens size not the same as target_actual_text.')

        AbstractResultPackage.__init__(self, pkg_name='QuestionAnswerResult',
                                       strict_content_check=strict_content_check, **kwargs)
        # todo: check if this is efficient
        self.paragraph_text_tokens = [[str(w) for w in paragraph] for paragraph in paragraph_text_tokens]
        self.target_actual_text = target_actual_text
        self.use_target_actual_text = target_actual_text is not None

        self.output_text_dir = os.path.expanduser(output_text_dir) if output_text_dir else None
        self.use_perl_rouge = use_perl_rouge

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

        if not self.use_perl_rouge:
            rogue_metric = ROUGEMetric(true_text, pred_text, target_actual_text=self.use_target_actual_text,
                                       output_text_dir=self.output_text_dir)
        else:
            rogue_metric = ROUGEPerlMetric(true_text, pred_text, self.output_text_dir,
                                           target_actual_text=self.use_target_actual_text)

        # self.results_dict = {**rogue_metric, **rogue_metric_non_official}
        self.results_dict = rogue_metric.get_metric_dict()


class QuestionAnswerSpanClassificationResultPackage(AbstractResultPackage):
    def __init__(self, strict_content_check=False, **kwargs):
        """

        Args:
            strict_content_check (bool):
            **kwargs (dict):
        """
        AbstractResultPackage.__init__(self, pkg_name='QuestionAnswerSpanClassificationResult',
                                       strict_content_check=strict_content_check, **kwargs)

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
        AbstractResultPackage.__init__(self, pkg_name='TextSummarizationResult',
                                       strict_content_check=strict_content_check, **kwargs)

    def prepare_results_dict(self):
        """

        Returns:

        """
        # rogue_result = ROUGEMetric(self.y_true, self.y_predicted).get_metric_dict()

        raise NotImplementedError


class MachineTranslationResultPackage(AbstractResultPackage):
    def __init__(self, output_attn_heatmap_dir=None, strict_content_check=False, **kwargs):
        """

        Args:
            output_attn_heatmap_dir (str or None):
            strict_content_check (bool):
            **kwargs (dict):
        """
        AbstractResultPackage.__init__(self, pkg_name='MachineTranslationResult',
                                       strict_content_check=strict_content_check, **kwargs)
        self.output_attn_heatmap_dir = output_attn_heatmap_dir
        self.attention_matrices = None

    def prepare_results_dict(self):
        """

        Returns:

        """
        # bleu_result = BLEUCorpusScoreMetric(self.y_true, self.y_predicted).get_metric_dict()
        # perplexity_result = PerplexityMetric(self.y_true, self.y_predicted).get_metric_dict()
        #
        # self.results_dict = {**bleu_result, **perplexity_result}
        #
        # # Don't include TrainLoop objects inside the package - it makes it useful only for PyTorch, not other frameworks
        # if self.output_attn_heatmap_dir is not None:
        #     # Get this from **kwargs or find another way of getting attention matrices
        #     self.attention_matrices = self.additional_results['additional_results']['attention_matrices']
        #
        #     attn_heatmap_metric = AttentionHeatMap(self.attention_matrices, self.y_true, self.y_predicted,
        #                                            self.output_attn_heatmap_dir)
        #
        #     attn_heatmap_plot_paths = attn_heatmap_metric.get_metric_dict()
        #     self.results_dict = {**self.results_dict, **attn_heatmap_plot_paths}

        raise NotImplementedError

    # def list_additional_results_dump_paths(self):
    #     file_name = 'attention_heatmaps.zip'
    #     results_file_local_path = os.path.join(self.output_attn_heatmap_dir, file_name)
    #
    #     self.zip_additional_results_dump(os.path.join(self.output_attn_heatmap_dir, 'attention_heatmaps'),
    #                                      results_file_local_path)
    #
    #     return [[file_name, results_file_local_path]]
