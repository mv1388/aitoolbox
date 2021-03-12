import re
import unicodedata


def unicode_to_ascii(text_string):
    """Turn a Unicode string to plain ASCII

    Taken from: http://stackoverflow.com/a/518232/2809427

    Args:
        text_string (str):

    Returns:
        str:
    """
    return ''.join(
        c for c in unicodedata.normalize('NFD', text_string)
        if unicodedata.category(c) != 'Mn'
    )


def normalize_string(text_string, unicode_to_ascii_convert=True):
    """Lowercase, trim, and remove non-letter characters

    Args:
        text_string (str):
        unicode_to_ascii_convert (bool):

    Returns:
        str:
    """
    text_string = text_string.lower().strip()
    if unicode_to_ascii_convert:
        text_string = unicode_to_ascii(text_string)
    text_string = re.sub(r"([.!?])", r" \1", text_string)
    text_string = re.sub(r"[^0-9a-zA-Z.!?]+", r" ", text_string)
    text_string = re.sub(r"\s+", r" ", text_string).strip()
    return text_string


def str2bool(w):
    """

    Args:
        w:

    Returns:
        bool:
    """
    if w.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif w.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError('Boolean value expected.')


def find_sub_list(sub_list, main_list):
    """Find starting and ending position of a sublist in a longer list.

    Args:
        sub_list (list): sublist
        main_list (list): main longer list

    Returns:
        (int, int): start and end index in the list l. Returns None if sublist is not found in the main list.
    """
    if len(sub_list) > len(main_list):
        raise ValueError('len(sub_list) > len(main_list); should be len(sub_list) <= len(main_list)')

    sll = len(sub_list)
    for ind in (i for i, e in enumerate(main_list) if e == sub_list[0]):
        if main_list[ind:ind+sll] == sub_list:
            return ind, ind+sll-1
