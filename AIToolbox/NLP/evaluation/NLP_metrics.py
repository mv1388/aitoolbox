from pyrouge import Rouge155
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from torchnlp.metrics import bleu


from AIToolbox.experiment_save.core_metrics.base_metric import AbstractBaseMetric


class ROGUEMetric(AbstractBaseMetric):
    def __init__(self, y_true, y_predicted):
        """

        Use this package:

            https://pypi.org/project/pyrouge/
            https://github.com/bheinzerling/pyrouge


        Args:
            y_true (numpy.array or list):
            y_predicted (numpy.array or list):
        """
        AbstractBaseMetric.__init__(self, y_true, y_predicted)
        self.metric_name = 'ROGUE'

    def calculate_metric(self):
        raise NotImplementedError


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
