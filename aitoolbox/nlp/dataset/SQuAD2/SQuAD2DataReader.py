import os
import json
from collections import Counter
from tqdm import tqdm

from allennlp.data.tokenizers import WordTokenizer
from allennlp.data.dataset_readers.reading_comprehension import util

from aitoolbox.cloud.AWS.data_access import SQuAD2DatasetFetcher
from aitoolbox.nlp.core.vocabulary import Vocabulary

"""

    Implementation relying on the allennlp library for the data processing instead of aitoolbox's NLP core package functions

    Faster processing, however, now non alphanum remove, etc.

"""


def get_dataset_local_copy(local_dataset_folder_path, protect_local_folder=True):
    """Interface method for getting a local copy of SQuAD2 dataset

    If a local copy is not found, dataset is automatically downloaded from S3.

    Args:
        local_dataset_folder_path (str):
        protect_local_folder (bool):

    Returns:
        None
    """
    dataset_fetcher = SQuAD2DatasetFetcher(bucket_name='dataset-store',
                                           local_dataset_folder_path=local_dataset_folder_path)
    dataset_fetcher.fetch_dataset(protect_local_folder=protect_local_folder)


class SQuAD2ConcatContextDatasetReader:
    def __init__(self, file_path, tokenizer=None, is_train=True, dev_mode_size=None):
        """

        Args:
            file_path (str):
            tokenizer:
            is_train (bool):
            dev_mode_size:
        """
        self.file_path = os.path.expanduser(file_path)
        self.is_train = is_train
        self.dataset = None

        self._tokenizer = tokenizer or WordTokenizer()

        with open(self.file_path) as dataset_file:
            dataset_json = json.load(dataset_file)
            self.dataset = dataset_json['data']

        if dev_mode_size is not None and dev_mode_size > 0:
            self.dataset = self.dataset[:dev_mode_size]

        self.vocab = Vocabulary('SQuAD2', document_level=False)

    def read(self):
        """Read SQuAD data. Tested and it works

        Returns:
            list, aitoolbox.nlp.core.vocabulary.Vocabulary:
        """
        data = []

        for article in tqdm(self.dataset, total=len(self.dataset)):
            for paragraph_json in article['paragraphs']:

                paragraph_text = paragraph_json["context"]
                tokenized_paragraph = self.tokenize_process_paragraph(paragraph_text)

                for question_answer in paragraph_json['qas']:
                    if question_answer['is_impossible']:
                        continue

                    question_text = question_answer["question"].strip().replace("\n", "")
                    tokenized_question = self.tokenize_process_question(question_text)

                    answer_texts = [answer['text'] for answer in question_answer['answers']]

                    span_starts = [answer['answer_start'] for answer in question_answer['answers']]
                    span_ends = [start + len(answer) for start, answer in zip(span_starts, answer_texts)]

                    example = self.process_example(tokenized_paragraph, tokenized_question,
                                                   zip(span_starts, span_ends), answer_texts)
                    data.append(example)

        return data, self.vocab

    def process_example(self, paragraph_tokens, question_tokens, char_spans, answer_texts):
        """

        Args:
            paragraph_tokens:
            question_tokens:
            char_spans:
            answer_texts:

        Returns:

        """
        token_spans = []
        passage_offsets = [(token.idx, token.idx + len(token.text)) for token in paragraph_tokens]

        for char_span_start, char_span_end in char_spans:
            (span_start, span_end), error = util.char_span_to_token_span(passage_offsets,
                                                                         (char_span_start, char_span_end))
            # if error:
            #     print('Error')

            token_spans.append((span_start, span_end))

        candidate_answers = Counter()
        for span_start, span_end in token_spans:
            candidate_answers[(span_start, span_end)] += 1
        span_start, span_end = candidate_answers.most_common(1)[0][0]

        # TODO: check if I really always only want 1 answer and not multiple options.
        #  Maybe have a parameter deciding which option
        selected_answer_text = answer_texts[token_spans.index(candidate_answers.most_common(1)[0][0])]

        return paragraph_tokens, question_tokens, (span_start, span_end), selected_answer_text

    def tokenize_process_paragraph(self, paragraph_text):
        """

        Args:
            paragraph_text:

        Returns:

        """
        tokenized_paragraph = self._tokenizer.tokenize(paragraph_text)

        if self.is_train:
            self.vocab.add_sentence(tokenized_paragraph)

        return tokenized_paragraph

    def tokenize_process_question(self, question_text):
        """

        Args:
            question_text:

        Returns:

        """
        tokenized_question = self._tokenizer.tokenize(question_text)

        if self.is_train:
            self.vocab.add_sentence(tokenized_question)

        return tokenized_question


class GeneratorSQuAD2ConcatContextDatasetReader(SQuAD2ConcatContextDatasetReader):
    """

        This implementation with the generator has not been tested yet.

        Check especially in the read() if calling list(self.read_generator()) also in turn fills the self.vocab,
        thus you get returned a complete vocabulary


    """

    def __init__(self, file_path, tokenizer=None, is_train=True, dev_mode_size=None):
        """

        Args:
            file_path:
            tokenizer:
            is_train:
            dev_mode_size:
        """
        SQuAD2ConcatContextDatasetReader.__init__(self, file_path, tokenizer, is_train, dev_mode_size)

    def read(self):
        """

        Returns:
            list:
        """
        return list(self.read_generator()), self.vocab

    def read_generator(self):
        """

        Yields:
            list:
        """
        for article in tqdm(self.dataset, total=len(self.dataset)):
            for paragraph_json in article['paragraphs']:

                paragraph_text = paragraph_json["context"]
                tokenized_paragraph = self.tokenize_process_paragraph(paragraph_text)

                for question_answer in paragraph_json['qas']:
                    if question_answer['is_impossible']:
                        continue

                    question_text = question_answer["question"].strip().replace("\n", "")
                    tokenized_question = self.tokenize_process_question(question_text)

                    answer_texts = [answer['text'] for answer in question_answer['answers']]

                    span_starts = [answer['answer_start'] for answer in question_answer['answers']]
                    span_ends = [start + len(answer) for start, answer in zip(span_starts, answer_texts)]

                    example = self.process_example(tokenized_paragraph, tokenized_question,
                                                   zip(span_starts, span_ends), answer_texts)

                    yield example
