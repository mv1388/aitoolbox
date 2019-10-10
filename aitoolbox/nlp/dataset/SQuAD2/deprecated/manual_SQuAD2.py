import json
import pickle
import os
from tqdm import tqdm

from aitoolbox.cloud.AWS.data_access import SQuAD2DatasetFetcher
from aitoolbox.nlp.core.vocabulary import Vocabulary
from aitoolbox.nlp.core.core import *


def get_dataset_local_copy(local_dataset_folder_path, protect_local_folder=True):
    """Interface method for getting a local copy of SQuAD2 dataset

    If a local copy is not found, dataset is automatically downloaded from S3.

    Args:
        local_dataset_folder_path (str):
        protect_local_folder (bool):

    Returns:
        None

    """
    dataset_fetcher = SQuAD2DatasetFetcher(bucket_name='dataset-store', local_dataset_folder_path=local_dataset_folder_path)
    dataset_fetcher.fetch_dataset(protect_local_folder)


class SQuAD2DatasetPrepareResult:
    def __init__(self, dataset_name, dataset_type='train', save_vocab=True):
        """

        Args:
            dataset_name:
            dataset_type:
            vocab_memory_safeguard:
        """
        self.dataset_name = dataset_name
        self.dataset_type = dataset_type
        self.save_vocab = save_vocab

        self.context_text_list = None
        self.question_text_list = None
        self.answer_text_list = None
        self.orig_answer_start_end_tuple_list = None
        self.answer_start_end_tuple_list = None

        self.vocab = None
        self.max_ctx_qs_len = None

    def store_data(self, context_text_list, question_text_list, answer_text_list,
                   orig_answer_start_end_tuple_list, answer_start_end_tuple_list):
        """

        Args:
            context_text_list:
            question_text_list:
            answer_text_list:
            orig_answer_start_end_tuple_list:
            answer_start_end_tuple_list:

        Returns:

        """
        self.context_text_list = context_text_list
        self.question_text_list = question_text_list
        self.answer_text_list = answer_text_list
        self.orig_answer_start_end_tuple_list = orig_answer_start_end_tuple_list
        self.answer_start_end_tuple_list = answer_start_end_tuple_list

    def store_vocab(self, vocab):
        """

        Args:
            vocab:

        Returns:

        """
        if self.save_vocab:
            self.vocab = vocab
        else:
            print('save_vocab is on False... not saving the vocabulary')

    def store_max_context_questions_max_len(self, max_ctx_qs_len):
        """

        Args:
            max_ctx_qs_len:

        Returns:

        """
        self.max_ctx_qs_len = max_ctx_qs_len


class SQuAD2DataPreparation:
    def __init__(self, train_path, dev_path, skip_is_impossible=True, skip_examples_w_span=True):
        """

        Args:
            train_path:
            dev_path:
            skip_is_impossible:
            skip_examples_w_span:
        """
        self.train_path = os.path.expanduser(train_path)
        self.dev_path = os.path.expanduser(dev_path)
        self.train_data = self.load_json_file(self.train_path)
        self.dev_data = self.load_json_file(self.dev_path)

        self.skip_is_impossible = skip_is_impossible
        self.is_impossible_ctr = 0

        self.skip_examples_w_span = skip_examples_w_span
        self.span_not_found_ctr = 0

        self.vocab = Vocabulary('SQuAD2', document_level=False)

    def process_data(self, dump_folder_path=None):
        """

        Args:
            dump_folder_path:

        Returns:

        """
        train_data = self.build_dataset(self.train_data, 'train')
        dev_data = self.build_dataset(self.dev_data, 'dev')

        if dump_folder_path is not None:
            with open(os.path.join(dump_folder_path, 'train_data_SQuAD2.p'), 'wb') as f:
                pickle.dump(train_data, f)
            with open(os.path.join(dump_folder_path, 'dev_data_SQuAD2.p'), 'wb') as f:
                pickle.dump(dev_data, f)

        return train_data, dev_data, self.vocab

    def vectorize_data(self, train_data=None, dev_data=None, vocab=None, dump_folder_path=None):
        """

        Args:
            train_data:
            dev_data:
            vocab:
            dump_folder_path:

        Returns:

        """
        if train_data is None and dev_data is None and vocab is None:
            train_data, dev_data, vocab = self.process_data(dump_folder_path)

        vect_train_data = self.get_vectorized_data_prep_result(train_data, vocab)
        vect_dev_data = self.get_vectorized_data_prep_result(dev_data, vocab)

        print(vect_train_data.vocab)
        # print(vect_dev_data.vocab)

        if dump_folder_path is not None:
            with open(os.path.join(dump_folder_path, 'vect_train_data_SQuAD2.p'), 'wb') as f:
                pickle.dump(vect_train_data, f)
            with open(os.path.join(dump_folder_path, 'vect_dev_data_SQuAD2.p'), 'wb') as f:
                pickle.dump(vect_dev_data, f)

        return vect_train_data, vect_dev_data, vocab

    def get_vectorized_data_prep_result(self, data_prep_result, vocab):
        """

        Args:
            data_prep_result:
            vocab:

        Returns:

        """
        dataset_type = data_prep_result.dataset_type

        vect_data_result = SQuAD2DatasetPrepareResult('SQuAD2', dataset_type,
                                                      save_vocab=dataset_type == 'train')

        vect_data_result.store_data(
            [vocab.convert_sent2idx_sent(ctx) for ctx in data_prep_result.context_text_list],
            [vocab.convert_sent2idx_sent(qus) for qus in data_prep_result.question_text_list],
            [vocab.convert_sent2idx_sent(answ) for answ in data_prep_result.answer_text_list],
            data_prep_result.orig_answer_start_end_tuple_list,
            data_prep_result.answer_start_end_tuple_list
        )
        vect_data_result.store_vocab(vocab)
        vect_data_result.store_max_context_questions_max_len(data_prep_result.max_ctx_qs_len)

        return vect_data_result

    def build_dataset(self, data_json, dataset_name):
        """

        Args:
            data_json:
            dataset_name:

        Returns:

        """
        context_text_list = []
        question_text_list = []
        answer_text_list = []
        orig_answer_start_end_tuple_list = []
        answer_start_end_tuple_list = []

        max_context_text_len = 0
        max_question_text_len = 0

        prep_dataset_result = SQuAD2DatasetPrepareResult('SQuAD2', dataset_name,
                                                         save_vocab=dataset_name == 'train')

        for wiki_page in tqdm(data_json, total=len(data_json), desc=dataset_name):
            title = wiki_page['title']
            paragraphs_list = wiki_page['paragraphs']
            for paragraph in paragraphs_list:
                context_paragraph = paragraph['context']
                context_paragraph_tokens = self.process_context_text(context_paragraph, dataset_name == 'train')

                question_answer_list = paragraph['qas']

                for question_answer_dict in question_answer_list:
                    is_impossible = question_answer_dict['is_impossible']
                    answer_list = question_answer_dict['answers']

                    if self.skip_is_impossible and is_impossible:
                        self.is_impossible_ctr += 1
                        continue

                    question_text = question_answer_dict['question']
                    question_text_tokens = self.process_question_text(question_text, dataset_name == 'train')

                    for answer_dict in answer_list:
                        answer_text = answer_dict['text']
                        answer_text_tokens = self.process_answer_text(answer_text, dataset_name == 'train')

                        orig_answer_start_idx = answer_dict['answer_start']
                        orig_answer_end_idx = answer_dict['answer_start'] + len(answer_dict['text'])
                        orig_answer_start_end_tuple = (orig_answer_start_idx, orig_answer_end_idx)

                        first_answ_span_tokens = find_sub_list(answer_text_tokens, context_paragraph_tokens)
                        if self.skip_examples_w_span and first_answ_span_tokens is None:
                            self.span_not_found_ctr += 1
                            continue

                        context_text_list.append(context_paragraph_tokens)
                        question_text_list.append(question_text_tokens)
                        answer_text_list.append(answer_text_tokens)
                        orig_answer_start_end_tuple_list.append(orig_answer_start_end_tuple)
                        answer_start_end_tuple_list.append(first_answ_span_tokens)

                        max_context_text_len = max(max_context_text_len, len(context_paragraph_tokens))
                        max_question_text_len = max(max_question_text_len, len(question_text_tokens))

        prep_dataset_result.store_data(context_text_list, question_text_list, answer_text_list,
                                       orig_answer_start_end_tuple_list, answer_start_end_tuple_list)

        prep_dataset_result.store_vocab(self.vocab)
        max_ctx_qs_len = {'context': max_context_text_len, 'questions': max_question_text_len}
        prep_dataset_result.store_max_context_questions_max_len(max_ctx_qs_len)

        return prep_dataset_result

    def process_context_text(self, context_text, is_train):
        """

        Args:
            context_text:
            is_train:

        Returns:

        """
        norm_context_text = normalize_string(context_text)
        token_norm_context_text = norm_context_text.split(' ')
        if is_train:
            self.vocab.add_sentence(token_norm_context_text)
        return token_norm_context_text

    def process_question_text(self, question_text, is_train):
        """

        Args:
            question_text:
            is_train:

        Returns:

        """
        norm_context_text = normalize_string(question_text)
        token_norm_context_text = norm_context_text.split(' ')
        if is_train:
            self.vocab.add_sentence(token_norm_context_text)
        return token_norm_context_text

    def process_answer_text(self, answer_text, is_train):
        """

        Args:
            answer_text:
            is_train:

        Returns:

        """
        norm_context_text = normalize_string(answer_text)
        token_norm_context_text = norm_context_text.split(' ')
        if is_train:
            self.vocab.add_sentence(token_norm_context_text)
        return token_norm_context_text

    def load_json_file(self, file_path):
        """

        Args:
            file_path:

        Returns:

        """
        file_path = os.path.expanduser(file_path)
        with open(file_path) as f:
            data = json.load(f)['data']
        return data

    def load_prep_dumps(self, dump_folder_path):
        """

        Args:
            dump_folder_path:

        Returns:

        """
        dump_folder_path = os.path.expanduser(dump_folder_path)
        with open(os.path.join(dump_folder_path, 'train_data_SQuAD2.p'), 'rb') as f:
            train_data = pickle.load(f)
        with open(os.path.join(dump_folder_path, 'dev_data_SQuAD2.p'), 'rb') as f:
            dev_data = pickle.load(f)

        return train_data, dev_data, train_data.vocab

    def load_vect_prep_dumps(self, dump_folder_path):
        """

        Args:
            dump_folder_path:

        Returns:

        """
        dump_folder_path = os.path.expanduser(dump_folder_path)
        with open(os.path.join(dump_folder_path, 'vect_train_data_SQuAD2.p'), 'rb') as f:
            train_data = pickle.load(f)
        with open(os.path.join(dump_folder_path, 'vect_dev_data_SQuAD2.p'), 'rb') as f:
            dev_data = pickle.load(f)

        return train_data, dev_data, train_data.vocab
