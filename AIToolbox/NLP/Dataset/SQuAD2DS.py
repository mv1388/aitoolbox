from AIToolbox.NLP.DataPrep.core import *


def process_context_text(context_text, use_word_tokenize=True, rm_non_alphanum=True):
    """

    Args:
        context_text:
        use_word_tokenize:
        rm_non_alphanum:

    Returns:

    """
    return basic_tokenize(context_text, use_word_tokenize, rm_non_alphanum,
                          start_label='START_DOC', end_label='END_DOC')


def process_question_text(question_text, use_word_tokenize=True, rm_non_alphanum=True):
    """

    Args:
        question_text:
        use_word_tokenize:
        rm_non_alphanum:

    Returns:

    """
    return basic_tokenize(question_text, use_word_tokenize, rm_non_alphanum,
                          start_label='START_Q', end_label='END_Q')


def process_answer_text(answer_text, use_word_tokenize=True, rm_non_alphanum=True):
    """

    Args:
        answer_text:
        use_word_tokenize:
        rm_non_alphanum:

    Returns:

    """
    return basic_tokenize(answer_text, use_word_tokenize, rm_non_alphanum,
                          start_label='START_ANSW', end_label='END_ANSW')


def build_dataset(data_json, use_word_tokenize=True, rm_non_alphanum=True):
    """

    Args:
        data_json:
        use_word_tokenize:
        rm_non_alphanum:

    Returns:

    """
    print('Building datasets')

    context_text_list = []
    question_text_list = []
    answer_text_list = []
    answer_start_idx_list = []

    vocab_context_text = set()
    vocab_question_text = set()
    vocab_answer_text = set()

    max_context_text_len = 0
    max_question_text_len = 0

    for wiki_page in data_json:
        title = wiki_page['title']
        paragraphs_list = wiki_page['paragraphs']

        for paragraph in paragraphs_list:
            context_paragraph = paragraph['context']
            context_paragraph = process_context_text(context_paragraph, use_word_tokenize, rm_non_alphanum)

            question_answer_list = paragraph['qas']

            for question_answer_dict in question_answer_list:
                is_impossible = question_answer_dict['is_impossible']
                if is_impossible:
                    continue

                question_text = question_answer_dict['question']
                question_text = process_question_text(question_text, use_word_tokenize, rm_non_alphanum)

                answer_list = question_answer_dict['answers']

                for answer_dict in answer_list:
                    answer_text = answer_dict['text']
                    answer_text = process_answer_text(answer_text, use_word_tokenize, rm_non_alphanum)

                    answer_start_idx = answer_dict['answer_start']
                    first_answ_span = find_sub_list(answer_text[1:-1], context_paragraph)

                    if first_answ_span is None:
                        # print('aaaaaa')
                        continue

                    context_text_list.append(context_paragraph)
                    question_text_list.append(question_text)
                    answer_text_list.append(answer_text)
                    answer_start_idx_list.append(answer_start_idx)

                    vocab_context_text |= set(context_paragraph)
                    vocab_question_text |= set(question_text)
                    vocab_answer_text |= set(answer_text)

                    max_context_text_len = max(max_context_text_len, len(context_paragraph))
                    max_question_text_len = max(max_question_text_len, len(question_text))

    return context_text_list, question_text_list, answer_text_list, answer_start_idx_list, \
           vocab_context_text, vocab_question_text, vocab_answer_text, \
           max_context_text_len, max_question_text_len

