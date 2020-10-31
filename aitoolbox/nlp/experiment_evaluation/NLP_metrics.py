import os
import re
import shutil
import string
from collections import Counter
import numpy as np
from pyrouge import Rouge155
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from torchnlp.metrics import bleu
from transformers import glue_compute_metrics, xnli_compute_metrics

from aitoolbox.experiment.core_metrics.abstract_metric import AbstractBaseMetric


class ROUGEMetric(AbstractBaseMetric):
    def __init__(self, y_true, y_predicted,
                 target_actual_text=False, output_text_dir=None,
                 output_text_cleaning_regex=(r'<.*?>', r'[^a-zA-Z0-9.?! ]+')):
        """ROGUE score calculation

        From this package:
            https://github.com/pltrdy/rouge


        Args:
            y_true (numpy.array or list):
            y_predicted (numpy.array or list):
            target_actual_text (bool):
            output_text_dir (str):
            output_text_cleaning_regex (list):

        """
        self.output_text_cleaning_regex = output_text_cleaning_regex
        self.target_actual_text = target_actual_text
        self.output_text_dir = output_text_dir
        AbstractBaseMetric.__init__(self, y_true, y_predicted, metric_name='ROGUE', np_array=False)

    def calculate_metric(self):
        if self.output_text_dir is not None:
            # Not affecting the metric calculation. Just for record keeping it drops the texts to disk so they can be
            # reviewed
            self.dump_answer_text_to_disk(self.y_true, self.y_predicted,
                                          self.output_text_dir, self.output_text_cleaning_regex,
                                          self.target_actual_text)

        self.prepare_text()

        rouge_calc = Rouge()
        hypothesis = self.y_predicted
        reference = self.y_true

        # TODO: remove try-except... just for testing
        try:
            return rouge_calc.get_scores(hypothesis, reference, avg=True)
        except:
            print('hypothesis')
            print(hypothesis)
            print('reference')
            print(reference)
            exit()

    def prepare_text(self):
        if not self.target_actual_text:
            self.y_true = [' '.join(sent) for sent in self.y_true]
        self.y_predicted = [' '.join(sent) if len(sent) > 0 else ' ' for sent in self.y_predicted]

    @staticmethod
    def dump_answer_text_to_disk(true_text, pred_text, output_text_dir, output_text_cleaning_regex, target_actual_text):
        """

        Problems:
            Defined regex text cleaning to deal with Illegal division by zero
            https://ireneli.eu/2018/01/11/working-with-rouge-1-5-5-evaluation-metric-in-python/

        Args:
            true_text (list):
            pred_text (list):
            output_text_dir (str):
            output_text_cleaning_regex (list):
            target_actual_text (bool):

        Returns:

        """
        if os.path.exists(output_text_dir):
            shutil.rmtree(output_text_dir)

        os.mkdir(output_text_dir)

        for i, (pred_answ, true_answ) in enumerate(zip(pred_text, true_text)):
            with open(os.path.join(output_text_dir, f'answer_pred_true_{i}.txt'), 'w', encoding='utf-8') as f:
                # Default regex cleaners: (r'<.*?>', r'[^a-zA-Z0-9.?! ]+')
                pred_answ_clean = ROUGEPerlMetric.regex_clean_text(pred_answ, output_text_cleaning_regex)
                pred_answ_clean = ' '.join(pred_answ_clean) if len(pred_answ_clean) > 0 else ' '

                if target_actual_text:
                    true_answ_clean = [true_answ]
                else:
                    true_answ_clean = ROUGEPerlMetric.regex_clean_text(true_answ, output_text_cleaning_regex)
                true_answ_clean = ' '.join(true_answ_clean)

                f.write(f'Answer to question {i}:\n')
                f.write(f'Predicted:\t{pred_answ_clean}\n')
                f.write(f'True:\t{true_answ_clean}\n')


class ROUGEPerlMetric(AbstractBaseMetric):
    def __init__(self, y_true, y_predicted,
                 output_text_dir, output_text_cleaning_regex=(r'<.*?>', r'[^a-zA-Z0-9.?! ]+'),
                 target_actual_text=False):
        """ROGUE score calculation using the Perl implementation

        Use this package:
            https://pypi.org/project/pyrouge/
            https://github.com/bheinzerling/pyrouge


        Problems:
            Defined regex text cleaning to deal with Illegal division by zero
            https://ireneli.eu/2018/01/11/working-with-rouge-1-5-5-evaluation-metric-in-python/


        Args:
            y_true (numpy.array or list): gold standard summaries are ‘model’ summaries
            y_predicted (numpy.array or list): your summaries are ‘system’ summaries
            output_text_dir (str):
            output_text_cleaning_regex (list):
            target_actual_text (bool):

        """
        self.output_text_dir = output_text_dir
        self.output_text_cleaning_regex = output_text_cleaning_regex
        self.target_actual_text = target_actual_text
        AbstractBaseMetric.__init__(self, y_true, y_predicted, metric_name='ROGUE_Perl', np_array=False)

    def calculate_metric(self):
        self.dump_answer_text_to_disk(self.y_true, self.y_predicted,
                                      self.output_text_dir, self.output_text_cleaning_regex,
                                      self.target_actual_text)

        rouge = Rouge155()
        # In ROUGE, your summaries are ‘system’ summaries and the gold standard summaries are ‘model’ summaries.
        rouge.system_dir = os.path.join(self.output_text_dir, 'pred_answer')
        rouge.model_dir = os.path.join(self.output_text_dir, 'true_answer')
        rouge.system_filename_pattern = 'pred_answer_text.(\d+).txt'
        rouge.model_filename_pattern = 'true_answer_text.#ID#.txt'

        rouge_output = rouge.convert_and_evaluate()
        output_dict = rouge.output_to_dict(rouge_output)
        
        return output_dict

    @staticmethod
    def dump_answer_text_to_disk(true_text, pred_text, output_text_dir, output_text_cleaning_regex, target_actual_text):
        """

        Problems:
            Defined regex text cleaning to deal with Illegal division by zero
            https://ireneli.eu/2018/01/11/working-with-rouge-1-5-5-evaluation-metric-in-python/

        Args:
            true_text (list):
            pred_text (list):
            output_text_dir (str):
            output_text_cleaning_regex (list):
            target_actual_text (bool):

        Returns:

        """
        if os.path.exists(output_text_dir):
            shutil.rmtree(output_text_dir)

        os.mkdir(output_text_dir)
        os.mkdir(os.path.join(output_text_dir, 'true_answer'))
        os.mkdir(os.path.join(output_text_dir, 'pred_answer'))

        for i, text in enumerate(true_text):
            # TODO: Encoding setting not tested yet
            with open(os.path.join(output_text_dir, f'true_answer/true_answer_text.{i}.txt'), 'w', encoding='utf-8') as f:
                # Default regex cleaners: (r'<.*?>', r'[^a-zA-Z0-9.?! ]+')
                if target_actual_text:
                    text_clean = [text]
                else:
                    text_clean = ROUGEPerlMetric.regex_clean_text(text, output_text_cleaning_regex)
                f.write(' '.join(text_clean))

        for i, text in enumerate(pred_text):
            # TODO: Encoding setting not tested yet
            with open(os.path.join(output_text_dir, f'pred_answer/pred_answer_text.{i}.txt'), 'w', encoding='utf-8') as f:
                # Default regex cleaners: (r'<.*?>', r'[^a-zA-Z0-9.?! ]+')
                text_clean = ROUGEPerlMetric.regex_clean_text(text, output_text_cleaning_regex)
                f.write(' '.join(text_clean) if len(text_clean) > 0 else ' ')

    @staticmethod
    def regex_clean_text(text, cleaning_regex_list):
        """

        Args:
            text (list):
            cleaning_regex_list (list):

        Returns:
            list:

        """
        # The default is: (r'<.*?>', r'[^a-zA-Z0-9.?! ]+')
        for cleaning_regex in cleaning_regex_list:
            re_pattern = re.compile(cleaning_regex)
            text = [re_pattern.sub('', t) for t in text if len(re_pattern.sub('', t)) > 0]
        return text


class ExactMatchTextMetric(AbstractBaseMetric):
    def __init__(self, y_true, y_predicted,
                 target_actual_text=False, output_text_dir=None):
        """Calculate exact match of answered strings

        Args:
            y_true (numpy.array or list):
            y_predicted (numpy.array or list):
            target_actual_text (bool):
            output_text_dir (str):
        """
        if len(y_true) != len(y_predicted):
            raise ValueError(f'len(y_true) != len(y_predicted). Got {len(y_true)} != {len(y_predicted)}')

        self.target_actual_text = target_actual_text
        self.output_text_dir = output_text_dir
        AbstractBaseMetric.__init__(self, y_true, y_predicted, metric_name='EM', np_array=False)

    def calculate_metric(self):
        if self.output_text_dir is not None:
            # Not affecting the metric calculation. Just for record keeping it drops the texts to disk so they can be
            # reviewed
            ROUGEMetric.dump_answer_text_to_disk(self.y_true, self.y_predicted,
                                                 self.output_text_dir, [], self.target_actual_text)

        if not self.target_actual_text:
            self.y_true = [' '.join(sent) for sent in self.y_true]
        self.y_predicted = [' '.join(sent) for sent in self.y_predicted]

        em = 0
        for pred_answ, true_answ in zip(self.y_predicted, self.y_true):
            em += int(self.normalize_answer(pred_answ) == self.normalize_answer(true_answ))

        return 100. * em / len(self.y_true)

    @staticmethod
    def normalize_answer(text_str):
        """Convert to lowercase and remove punctuation, articles and extra whitespace.

        All methods below this line are from the official SQuAD 2.0 eval script
        https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/

        Args:
            text_str (str):

        Returns:
            str
        """
        def remove_articles(text):
            regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
            return re.sub(regex, ' ', text)

        def white_space_fix(text):
            return ' '.join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(text_str))))


class F1TextMetric(AbstractBaseMetric):
    def __init__(self, y_true, y_predicted,
                 target_actual_text=False, output_text_dir=None):
        """Calculate F1 score of answered strings

        Args:
            y_true (numpy.array or list):
            y_predicted (numpy.array or list):
            target_actual_text (bool):
            output_text_dir (str):
        """
        if len(y_true) != len(y_predicted):
            raise ValueError(f'len(y_true) != len(y_predicted). Got {len(y_true)} != {len(y_predicted)}')

        self.target_actual_text = target_actual_text
        self.output_text_dir = output_text_dir
        AbstractBaseMetric.__init__(self, y_true, y_predicted, metric_name='F1', np_array=False)

    def calculate_metric(self):
        if self.output_text_dir is not None:
            # Not affecting the metric calculation. Just for record keeping it drops the texts to disk so they can be
            # reviewed
            ROUGEMetric.dump_answer_text_to_disk(self.y_true, self.y_predicted,
                                                 self.output_text_dir, [], self.target_actual_text)

        if not self.target_actual_text:
            self.y_true = [' '.join(sent) for sent in self.y_true]
        self.y_predicted = [' '.join(sent) for sent in self.y_predicted]

        f1 = 0
        for pred_answ, true_answ in zip(self.y_predicted, self.y_true):
            f1 += self.compute_f1(true_answ, pred_answ)

        return 100. * f1 / len(self.y_true)

    @staticmethod
    def compute_f1(a_gold, a_pred):
        gold_toks = F1TextMetric.get_tokens(a_gold)
        pred_toks = F1TextMetric.get_tokens(a_pred)
        common = Counter(gold_toks) & Counter(pred_toks)
        num_same = sum(common.values())
        if len(gold_toks) == 0 or len(pred_toks) == 0:
            # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
            return int(gold_toks == pred_toks)
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(pred_toks)
        recall = 1.0 * num_same / len(gold_toks)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    @staticmethod
    def get_tokens(s):
        if not s:
            return []
        return ExactMatchTextMetric.normalize_answer(s).split()


class BLEUSentenceScoreMetric(AbstractBaseMetric):
    def __init__(self, y_true, y_predicted, source_sents=None, output_text_dir=None):
        """BLEU score calculation

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
            source_sents (list or None):
            output_text_dir (str or None):

        """
        if output_text_dir is not None and source_sents is None:
            raise ValueError('output_text_dir is not None and source_sents is None; '
                             'if output_text_dir you must give the source_sents')

        self.output_text_dir = output_text_dir
        self.source_sents = source_sents
        AbstractBaseMetric.__init__(self, y_true, y_predicted, metric_name='BLEU_sentence_score', np_array=False)

    def calculate_metric(self):
        self.check_transl_sent_num_match([self.y_true, self.y_predicted])

        sentence_bleu_results = [sentence_bleu([true_t], pred_t) for true_t, pred_t in zip(self.y_true, self.y_predicted)]

        if self.output_text_dir is not None:
            self.dump_translation_text_to_disk(self.source_sents,
                                               [' '.join(sent) for sent in self.y_predicted],
                                               [' '.join(sent) for sent in self.y_true],
                                               sentence_bleu_results, self.output_text_dir)

        return np.mean(sentence_bleu_results)

    @staticmethod
    def dump_translation_text_to_disk(source_sents, pred_translations, true_translations, sentence_bleu_results,
                                      output_text_dir):
        """

        Args:
            source_sents (list):
            pred_translations (list):
            true_translations (list):
            sentence_bleu_results (list):
            output_text_dir (str):

        Returns:

        """
        BLEUSentenceScoreMetric.check_transl_sent_num_match([pred_translations, true_translations,
                                                             source_sents, sentence_bleu_results])

        if os.path.exists(output_text_dir):
            shutil.rmtree(output_text_dir)

        os.mkdir(output_text_dir)

        for i, (source, pred_transl, true_transl, bleu_result) in enumerate(zip(source_sents, pred_translations,
                                                                                true_translations, sentence_bleu_results)):
            with open(os.path.join(output_text_dir, f'transl_{i}.txt'), 'w', encoding='utf-8') as f:
                f.write(f'Source:\t{source}\n')
                f.write(f'Predicted:\t{pred_transl}\n')
                f.write(f'True:\t{true_transl}\n')
                f.write(f'BLEU: {bleu_result}\n')

    @staticmethod
    def check_transl_sent_num_match(sent_types):
        """

        Args:
            sent_types (list): list of lists
            
        Raises:
            ValueError

        """
        num_sents = len(sent_types[0])
        for sent_t in sent_types:
            if len(sent_t) != num_sents:
                raise ValueError(f"The length of list elements across different text types does not match "
                                 f"The featured lengths are: {', '.join([str(len(el)) for el in sent_types])}")


class BLEUCorpusScoreMetric(AbstractBaseMetric):
    def __init__(self, y_true, y_predicted, source_sents=None, output_text_dir=None):
        """BLEU corpus score calculation

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
            source_sents (list or None):
            output_text_dir (str or None):

        """
        self.output_text_dir = output_text_dir
        self.source_sents = source_sents
        AbstractBaseMetric.__init__(self, y_true, y_predicted, metric_name='BLEU_corpus_score', np_array=False)

    def calculate_metric(self):
        BLEUSentenceScoreMetric.check_transl_sent_num_match([self.y_true, self.y_predicted])

        if self.output_text_dir is not None:
            BLEUSentenceScoreMetric.dump_translation_text_to_disk(self.source_sents,
                                                                  [' '.join(sent) for sent in self.y_predicted],
                                                                  [' '.join(sent) for sent in self.y_true],
                                                                  ['na'] * len(self.y_predicted), self.output_text_dir)

        return corpus_bleu([[sent] for sent in self.y_true], self.y_predicted)


class BLEUScoreStrTorchNLPMetric(AbstractBaseMetric):
    def __init__(self, y_true, y_predicted, lowercase=False, source_sents=None, output_text_dir=None):
        """BLEU score calculation using the TorchNLP implementation

        Example:
            hypotheses = [
              "The brown fox jumps over the dog 笑",
              "The brown fox jumps over the dog 2 笑"
              ]
            references = [
              "The quick brown fox jumps over the lazy dog 笑",
              "The quick brown fox jumps over the lazy dog 笑"
              ]

            get_moses_multi_bleu(hypotheses, references, lowercase=True)
            46.51

        Args:
            y_true (list):
            y_predicted (list):
            lowercase (bool):
            source_sents (list or None):
            output_text_dir (str or None):

        """
        self.output_text_dir = output_text_dir
        self.source_sents = source_sents
        self.lowercase = lowercase
        AbstractBaseMetric.__init__(self, y_true, y_predicted, metric_name='BLEU_str_torchNLP_score', np_array=False)

    def calculate_metric(self):
        BLEUSentenceScoreMetric.check_transl_sent_num_match([self.y_true, self.y_predicted])

        sentence_bleu_results = [
            bleu.get_moses_multi_bleu([' '.join(true_t)], [' '.join(pred_t)], lowercase=self.lowercase)
            for true_t, pred_t in zip(self.y_true, self.y_predicted)
        ]
        
        if self.output_text_dir is not None:
            BLEUSentenceScoreMetric.dump_translation_text_to_disk(self.source_sents,
                                                                  [' '.join(sent) for sent in self.y_predicted],
                                                                  [' '.join(sent) for sent in self.y_true],
                                                                  sentence_bleu_results, self.output_text_dir)

        return float(np.mean(sentence_bleu_results))


class PerplexityMetric(AbstractBaseMetric):
    def __init__(self, batch_losses):
        """Perplexity metric used in MT

        Args:
            batch_losses (numpy.array or list):
        """
        AbstractBaseMetric.__init__(self, None, batch_losses, metric_name='Perplexity', np_array=False)

    def calculate_metric(self):
        return np.exp(np.mean(self.y_predicted))


class GLUEMetric(AbstractBaseMetric):
    def __init__(self, y_true, y_predicted, task_name):
        """GLUE evaluation metrics

        Wrapper around HF Transformers ``glue_compute_metrics()``

        Args:
            y_true:
            y_predicted:
            task_name (str): name of the GLUE task
        """
        self.task_name = task_name
        super().__init__(y_true, y_predicted, metric_name=f'GLUE_{task_name}')

    def calculate_metric(self):
        metric_dict = glue_compute_metrics(task_name=self.task_name, preds=self.y_predicted, labels=self.y_true)
        metric_dict = {k.replace('/', '_'): v for k, v in metric_dict.items()}
        return metric_dict


class XNLIMetric(AbstractBaseMetric):
    def __init__(self, y_true, y_predicted):
        """XNLI evaluation metrics

        Wrapper around HF Transformers ``xnli_compute_metrics()``

        Args:
            y_true:
            y_predicted:
        """
        super().__init__(y_true, y_predicted, metric_name='xnli_accuracy')

    def calculate_metric(self):
        metric_dict = xnli_compute_metrics(task_name='xnli', preds=self.y_predicted, labels=self.y_true)
        metric_dict = {k.replace('/', '_'): v for k, v in metric_dict.items()}
        return metric_dict
