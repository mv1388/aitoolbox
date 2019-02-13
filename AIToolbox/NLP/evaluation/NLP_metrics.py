import os
import shutil
from pyrouge import Rouge155
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from torchnlp.metrics import bleu

from AIToolbox.experiment_save.core_metrics.base_metric import AbstractBaseMetric


class ROGUEMetric(AbstractBaseMetric):
    def __init__(self, y_true, y_predicted, output_text_dir):
        """

        Use this package:
            https://pypi.org/project/pyrouge/
            https://github.com/bheinzerling/pyrouge


        Args:
            y_true (numpy.array or list): gold standard summaries are ‘model’ summaries
            y_predicted (numpy.array or list): your summaries are ‘system’ summaries
            output_text_dir (str):
        """
        self.output_text_dir = output_text_dir
        AbstractBaseMetric.__init__(self, y_true, y_predicted)
        self.metric_name = 'ROGUE'

    def calculate_metric(self):
        self.dump_answer_text_to_disk(self.y_true, self.y_predicted, self.output_text_dir)

        rouge = Rouge155()
        # In ROUGE, your summaries are ‘system’ summaries and the gold standard summaries are ‘model’ summaries.
        rouge.system_dir = os.path.join(self.output_text_dir, 'pred_answer')
        rouge.model_dir = os.path.join(self.output_text_dir, 'true_answer')
        rouge.system_filename_pattern = 'pred_answer_text.(\d+).txt'
        rouge.model_filename_pattern = 'true_answer_text.#ID#.txt'

        rouge_output = rouge.convert_and_evaluate()
        output_dict = rouge.output_to_dict(rouge_output)
        
        self.metric_result = output_dict

    @staticmethod
    def dump_answer_text_to_disk(true_text, pred_text, output_text_dir):
        """

        Args:
            true_text (list):
            pred_text (list):
            output_text_dir (str):

        Returns:

        """
        if os.path.exists(output_text_dir):
            shutil.rmtree(output_text_dir)

        os.mkdir(output_text_dir)
        os.mkdir(os.path.join(output_text_dir, 'true_answer'))
        os.mkdir(os.path.join(output_text_dir, 'pred_answer'))

        for i, text in enumerate(true_text):
            with open(os.path.join(output_text_dir, f'true_answer/true_answer_text.{i}.txt'), 'w') as f:
                f.write(' '.join(text))

        for i, text in enumerate(pred_text):
            with open(os.path.join(output_text_dir, f'pred_answer/pred_answer_text.{i}.txt'), 'w') as f:
                f.write(' '.join(text) if len(text) > 0 else ' ')


class ROGUENonOfficialMetric(AbstractBaseMetric):
    def __init__(self, y_true, y_predicted):
        """

        From this package:
            https://github.com/pltrdy/rouge


        Args:
            y_true (list):
            y_predicted (list):
        """
        AbstractBaseMetric.__init__(self, y_true, y_predicted)
        self.metric_name = 'ROGUE_nonOfficial'

    def calculate_metric(self):
        rouge_calc = Rouge()
        hypothesis = self.y_predicted
        reference = self.y_true
        self.metric_result = rouge_calc.get_scores(hypothesis, reference, avg=True)


class BLEUSentenceScoreMetric(AbstractBaseMetric):
    def __init__(self, y_true, y_predicted):
        """

        NLTK provides the sentence_bleu() function for evaluating a candidate sentence
        against one or more reference sentences.

        https://machinelearningmastery.com/calculate-bleu-score-for-text-python/

        The reference sentences must be provided as a list of sentences where each reference is a list of tokens.
        The candidate sentence is provided as a list of tokens. For example:

            reference = [['this', 'is', 'a', 'test'], ['this', 'is' 'test']]
            candidate = ['this', 'is', 'a', 'test']
            score = sentence_bleu(reference, candidate)

        Args:
            y_true (list):
            y_predicted (list):
        """
        AbstractBaseMetric.__init__(self, y_true, y_predicted)
        self.metric_name = 'BLEU_sentence_score'

    def calculate_metric(self):
        self.metric_result = sentence_bleu(self.y_true, self.y_predicted)


class BLEUCorpusScoreMetric(AbstractBaseMetric):
    def __init__(self, y_true, y_predicted):
        """

        Function called corpus_bleu() for calculating the BLEU score for multiple sentences such as a paragraph or
        a document.

        https://machinelearningmastery.com/calculate-bleu-score-for-text-python/

        The references must be specified as a list of documents where each document is a list of references and each
        alternative reference is a list of tokens, e.g. a list of lists of lists of tokens. The candidate documents must
        be specified as a list where each document is a list of tokens, e.g. a list of lists of tokens.

            references = [[['this', 'is', 'a', 'test'], ['this', 'is' 'test']]]
            candidates = [['this', 'is', 'a', 'test']]
            score = corpus_bleu(references, candidates)

        Args:
            y_true (list):
            y_predicted (list):
        """
        AbstractBaseMetric.__init__(self, y_true, y_predicted)
        self.metric_name = 'BLEU_corpus_score'

    def calculate_metric(self):
        self.metric_result = corpus_bleu(self.y_true, self.y_predicted)


class BLEUScoreStrTorchNLPMetric(AbstractBaseMetric):
    def __init__(self, y_true, y_predicted, lowercase=False):
        """

        Args:
            y_true (list):
            y_predicted (list):
        """
        AbstractBaseMetric.__init__(self, y_true, y_predicted)
        self.metric_name = 'BLEU_str_torchNLP_score'
        self.lowercase = lowercase

    def calculate_metric(self):
        self.metric_result = bleu.get_moses_multi_bleu(self.y_predicted, self.y_true, lowercase=self.lowercase)


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
