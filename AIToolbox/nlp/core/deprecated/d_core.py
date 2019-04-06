import re
from nltk.tokenize import word_tokenize


def basic_tokenize(text, use_word_tokenize=True, rm_non_alphanum=True, start_label='START', end_label='END'):
    """

    Args:
        text (str):
        use_word_tokenize (bool):
        rm_non_alphanum (bool):
        start_label (str):
        end_label (str):

    Returns:
        list:

    """
    if rm_non_alphanum:
        text = re.sub(r'\([^)]*\)', '', text)
        text = text.replace('-', ' ')
        text = re.sub(r'[^0-9a-zA-Z. ]', '', text)

    if use_word_tokenize:
        text_token = [el.lower() for el in word_tokenize(text)]
    else:
        text_token = text.lower().split()

    if start_label is not None:
        text_token = [start_label] + text_token
    if end_label is not None:
        text_token = text_token + [end_label]

    return text_token


def prepare_vocab_mapping(vocab, padding=True, special_labels=('<OOV>',)):
    """

    Args:
        vocab (list or set):
        padding (bool):
        special_labels (list or tuple):

    Returns:
        (dict, int):

    """
    vocab = set(vocab)
    vocab = sorted(vocab)

    idx_start = padding + len(special_labels)

    word2idx = dict((c, i + idx_start) for i, c in enumerate(vocab))
    for i, l in enumerate(special_labels, start=padding):
        word2idx[l] = i

    vocab_size = len(word2idx) + padding

    return word2idx, vocab_size


def vectorize_one_text(text_tokens, word_idx):
    """

    Args:
        text_tokens (list):
        word_idx (dict):

    Returns:
        list: text tokens converted into idx numbers

    """
    return [word_idx[w] if w in word_idx else word_idx['<OOV>'] for w in text_tokens]


def vectorize_text(text_tokens_lists, word_idx, text_maxlen=None, shorten_text_mode='end'):
    """

    Args:
        text_tokens_lists (list):
        word_idx (dict):
        text_maxlen (int or None):
        shorten_text_mode (str):

    Returns:
         list:

    """
    if text_maxlen is not None and text_maxlen > 0:
        if shorten_text_mode == 'end':
            text_idx_list = [vectorize_one_text(text_list[:text_maxlen], word_idx) for text_list in text_tokens_lists]
        elif shorten_text_mode == 'start':
            text_idx_list = [vectorize_one_text(text_list[len(text_list)-text_maxlen:], word_idx)
                             if len(text_list) > text_maxlen else
                             vectorize_one_text(text_list, word_idx)
                             for text_list in text_tokens_lists]
        else:
            raise ValueError('shorten_text_mode not end or start')
    else:
        text_idx_list = [vectorize_one_text(text_list, word_idx) for text_list in text_tokens_lists]

    return text_idx_list
