import re
from nltk.tokenize import word_tokenize


"""
            Not my code, so be careful
            
            Got this from github project
"""


def find_sub_list(sl, l):
    sll = len(sl)
    for ind in (i for i, e in enumerate(l) if e == sl[0]):
        if l[ind:ind+sll] == sl:
            return ind, ind+sll-1


def process_context_text(context_text, use_word_tokenize=True, rm_non_alphanum=True):
    if rm_non_alphanum:
        context_text = re.sub(r'\([^)]*\)', '', context_text)
        context_text = context_text.replace('-', ' ')
        context_text = re.sub(r'[^0-9a-zA-Z. ]', '', context_text)

    if use_word_tokenize:
        main_text_token = ['START_DOC'] + [el.lower() for el in word_tokenize(context_text)] + ['END_DOC']
    else:
        main_text_token = ['START_DOC'] + context_text.lower().split() + ['END_DOC']

    return main_text_token


def process_question_text(question_text, use_word_tokenize=True, rm_non_alphanum=True):
    if use_word_tokenize:
        question_text = ['START_Q'] + [el.lower() for el in word_tokenize(question_text)] + ['END_Q']
    else:
        question_text = ['START_Q'] + question_text.lower().split() + ['END_Q']

    return question_text


def process_answer_text(answer_text, use_word_tokenize=True, rm_non_alphanum=True):
    if use_word_tokenize:
        answer_text = ['START_ANSW'] + [el.lower() for el in word_tokenize(answer_text)] + ['END_ANSW']
    else:
        answer_text = ['START_ANSW'] + answer_text.lower().split() + ['END_ANSW']

    return answer_text


def build_dataset(data_json, use_word_tokenize=True, rm_non_alphanum=True):
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


def prepare_vocab_mapping(vocab):
    vocab = set(vocab)
    vocab = sorted(vocab)

    word2idx = dict((c, i + 2) for i, c in enumerate(vocab))
    word2idx['<OOV>'] = 1

    vocab_size = len(word2idx) + 1

    return word2idx, vocab_size


def vectorize_one_text(text_list, word_idx):
    return [word_idx[w] if w in word_idx else word_idx['<OOV>'] for w in text_list]


def vectorize_text(texts_list, word_idx, text_maxlen=None):
    text_idx_list = [vectorize_one_text(text_list, word_idx) for text_list in texts_list]

    if text_maxlen is not None and text_maxlen > 0:
        raise ValueError( )

    return text_idx_list
