import os

from aitoolbox.utils import dict_util
from aitoolbox.experiment.local_save.local_results_save import BaseLocalResultsSaver
from aitoolbox.experiment.result_package.abstract_result_packages import AbstractResultPackage
from aitoolbox.experiment.core_metrics.classification import AccuracyMetric
from aitoolbox.nlp.experiment_evaluation.NLP_metrics import ROUGEMetric, ROUGEPerlMetric, \
    ExactMatchTextMetric, F1TextMetric, \
    BLEUSentenceScoreMetric, BLEUCorpusScoreMetric, BLEUScoreStrTorchNLPMetric, PerplexityMetric, \
    GLUEMetric, XNLIMetric
from aitoolbox.nlp.experiment_evaluation.attention_heatmap import AttentionHeatMap


class QuestionAnswerResultPackage(AbstractResultPackage):
    def __init__(self, paragraph_text_tokens, target_actual_text=None, output_text_dir=None,
                 use_perl_rouge=False, flatten_result_dict=False,
                 strict_content_check=False, **kwargs):
        """Question Answering task performance evaluation result package

        Args:
            paragraph_text_tokens (list):
            target_actual_text (list or None):
            output_text_dir (str or None):
            use_perl_rouge (bool):
            flatten_result_dict (bool):
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
        self.flatten_result_dict = flatten_result_dict

    def prepare_results_dict(self):
        """

        Returns:
            dict:

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

        if not self.use_perl_rouge:
            rogue_metric = ROUGEMetric(true_text, pred_text, target_actual_text=self.use_target_actual_text,
                                       output_text_dir=self.output_text_dir)
        else:
            rogue_metric = ROUGEPerlMetric(true_text, pred_text, self.output_text_dir,
                                           target_actual_text=self.use_target_actual_text)

        em_metric = ExactMatchTextMetric(true_text, pred_text, target_actual_text=self.use_target_actual_text)
        f1_metric = F1TextMetric(true_text, pred_text, target_actual_text=self.use_target_actual_text)

        results_dict = rogue_metric + em_metric + f1_metric

        if self.flatten_result_dict:
            results_dict = dict_util.flatten_dict(results_dict)

        return results_dict

    def set_experiment_dir_path_for_additional_results(self, project_name, experiment_name, experiment_timestamp,
                                                       local_model_result_folder_path):
        if self.output_text_dir is not None:
            _, experiment_dir_path, _ = \
                BaseLocalResultsSaver.get_experiment_local_results_folder_paths(project_name, experiment_name,
                                                                                experiment_timestamp, local_model_result_folder_path)
            self.output_text_dir = os.path.join(experiment_dir_path, self.output_text_dir)

    def list_additional_results_dump_paths(self):
        if self.output_text_dir is not None:
            zip_path = self.zip_additional_results_dump(self.output_text_dir, self.output_text_dir)
            zip_file_name = os.path.basename(zip_path)
            return [[zip_file_name, zip_path]]


class QuestionAnswerSpanClassificationResultPackage(AbstractResultPackage):
    def __init__(self, strict_content_check=False, **kwargs):
        """Extractive Question Answering task performance evaluation result package

        Evaluates the classification of the correct answer start and end points.

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
            dict:

        """
        y_span_start_true = self.y_true[:, 0]
        y_span_start_predicted = self.y_predicted[:, 0]
        y_span_end_true = self.y_true[:, 1]
        y_span_end_predicted = self.y_predicted[:, 1]

        span_start_accuracy = AccuracyMetric(y_span_start_true, y_span_start_predicted, positive_class_thresh=None)
        span_start_accuracy.metric_name += '_span_start'
        span_end_accuracy = AccuracyMetric(y_span_end_true, y_span_end_predicted, positive_class_thresh=None)
        span_end_accuracy.metric_name += '_span_end'

        return span_start_accuracy + span_end_accuracy


class TextSummarizationResultPackage(AbstractResultPackage):
    def __init__(self, strict_content_check=False, **kwargs):
        """Text summarization task performance evaluation package

        Args:
            strict_content_check (bool):
            **kwargs (dict):

        """
        AbstractResultPackage.__init__(self, pkg_name='TextSummarizationResult',
                                       strict_content_check=strict_content_check, **kwargs)

    def prepare_results_dict(self):
        """

        Returns:
            dict:

        """
        # rogue_result = ROUGEMetric(self.y_true, self.y_predicted).get_metric_dict()

        raise NotImplementedError


class MachineTranslationResultPackage(AbstractResultPackage):
    def __init__(self, target_vocab, source_vocab=None, source_sents=None, output_text_dir=None, output_attn_heatmap_dir=None,
                 strict_content_check=False, **kwargs):
        """Machine Translation task performance evaluation package

        Args:
            target_vocab (aitoolbox.nlp.core.vocabulary.Vocabulary):
            source_vocab (aitoolbox.nlp.core.vocabulary.Vocabulary or None):
            source_sents (list or None):
            output_text_dir (str or None):
            output_attn_heatmap_dir (str or None):
            strict_content_check (bool):
            **kwargs (dict):

        """
        if output_text_dir is not None and (source_vocab is None or source_sents is None):
            raise ValueError(f'output_text_dir is not none which initiates the text results dump on disk. '
                             f'However, the the source_vocab or source_sents are not provided. '
                             f'To save text on disk these have to be supplied.\nCurrently:\n'
                             f'output_text_dir: {output_text_dir}\n'
                             f'source_vocab: {source_vocab}\n'
                             f'source_sents: {source_sents}\n')

        AbstractResultPackage.__init__(self, pkg_name='MachineTranslationResult',
                                       strict_content_check=strict_content_check, np_array=False, **kwargs)
        self.requires_loss = True
        self.target_vocab = target_vocab
        self.source_vocab = source_vocab
        self.source_sents = source_sents
        self.output_text_dir = output_text_dir
        self.output_attn_heatmap_dir = output_attn_heatmap_dir
        self.attention_matrices = None

        self.y_true_text = None
        self.y_predicted_text = None

    def prepare_results_dict(self):
        """

        Returns:
            dict: result dict which is combination of different BLEU metric calculations and possibly
                saved attention heatmap plot files and perplexity

        """
        self.y_true_text = [self.target_vocab.convert_idx_sent2sent(sent, rm_default_tokens=True) for sent in self.y_true]
        self.y_predicted_text = [self.target_vocab.convert_idx_sent2sent(sent, rm_default_tokens=True) for sent in self.y_predicted]

        bleu_avg_sent = BLEUSentenceScoreMetric(self.y_true_text, self.y_predicted_text,
                                                self.source_sents, self.output_text_dir)
        bleu_corpus_result = BLEUCorpusScoreMetric(self.y_true_text, self.y_predicted_text)
        # bleu_perl_result = BLEUScoreStrTorchNLPMetric(self.y_true_text, self.y_predicted_text)

        perplexity_result = PerplexityMetric(self.additional_results['additional_results']['loss'])

        results_dict = bleu_corpus_result + bleu_avg_sent + perplexity_result

        # Don't include TrainLoop objects inside the package - it makes it useful only for PyTorch, not other frameworks
        if self.output_attn_heatmap_dir is not None:
            # Get this from **kwargs or find another way of getting attention matrices
            self.attention_matrices = self.additional_results['additional_results']['attention_matrices']

            source_sent_idx_tokens = self.additional_results['additional_results']['source_sent_text']
            source_sent_text = [self.source_vocab.convert_idx_sent2sent(sent, rm_default_tokens=False)
                                for sent in source_sent_idx_tokens]

            attn_heatmap_metric = AttentionHeatMap(self.attention_matrices, source_sent_text, self.y_predicted_text,
                                                   self.output_attn_heatmap_dir)

            results_dict = results_dict + attn_heatmap_metric

        return results_dict

    def set_experiment_dir_path_for_additional_results(self, project_name, experiment_name, experiment_timestamp,
                                                       local_model_result_folder_path):
        if self.output_text_dir is not None:
            _, experiment_dir_path, _ = \
                BaseLocalResultsSaver.get_experiment_local_results_folder_paths(project_name, experiment_name,
                                                                                experiment_timestamp, local_model_result_folder_path)
            self.output_text_dir = os.path.join(experiment_dir_path, self.output_text_dir)

        if self.output_attn_heatmap_dir is not None:
            _, experiment_dir_path, _ = \
                BaseLocalResultsSaver.get_experiment_local_results_folder_paths(project_name, experiment_name,
                                                                                experiment_timestamp, local_model_result_folder_path)
            self.output_attn_heatmap_dir = os.path.join(experiment_dir_path, self.output_attn_heatmap_dir)

    def list_additional_results_dump_paths(self):
        additional_results_paths = []

        if self.output_text_dir is not None:
            zip_path = self.zip_additional_results_dump(self.output_text_dir, self.output_text_dir)
            zip_file_name = os.path.basename(zip_path)
            additional_results_paths.append([zip_file_name, zip_path])

        if self.output_attn_heatmap_dir is not None:
            zip_path = self.zip_additional_results_dump(self.output_attn_heatmap_dir, self.output_attn_heatmap_dir)
            zip_file_name = os.path.basename(zip_path)
            additional_results_paths.append([zip_file_name, zip_path])

        if len(additional_results_paths) > 0:
            return additional_results_paths


class GLUEResultPackage(AbstractResultPackage):
    def __init__(self, task_name):
        """GLUE task result package

        Wrapper around HF Transformers ``glue_compute_metrics()``

        Args:
            task_name (str): name of the GLUE task
        """
        super().__init__('GLUE benchmark')
        self.task_name = task_name

    def prepare_results_dict(self):
        glue_result = GLUEMetric(self.y_true, self.y_predicted, self.task_name).get_metric_dict()
        glue_result = dict_util.flatten_dict(glue_result)
        return glue_result


class XNLIResultPackage(AbstractResultPackage):
    def __init__(self):
        """XNLI task result package

        Wrapper around HF Transformers ``xnli_compute_metrics()``
        """
        super().__init__('XNLI benchmark')

    def prepare_results_dict(self):
        xnli_result = XNLIMetric(self.y_true, self.y_predicted).get_metric_dict()
        xnli_result = dict_util.flatten_dict(xnli_result)
        return xnli_result
