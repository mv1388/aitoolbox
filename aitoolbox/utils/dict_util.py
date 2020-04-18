import collections
import copy


def combine_prediction_metadata_batches(metadata_list):
    """Combines a list of dicts with the same keys and lists as values into a single dict with concatenated lists
        for each corresponding key

    Args:
        metadata_list (list): list of dicts with matching keys and lists for values

    Returns:
        dict: combined single dict
    """
    combined_metadata = {}

    for metadata_batch in metadata_list:
        for meta_el in metadata_batch:
            if meta_el not in combined_metadata:
                combined_metadata[meta_el] = []
            combined_metadata[meta_el] += metadata_batch[meta_el]

    return combined_metadata


def flatten_dict(nested_dict, parent_key='', sep='_'):
    """Flatten the nested dict of dicts of ...

    Args:
        nested_dict (dict): input dict
        parent_key (str):
        sep (str): separator when flattening the category

    Returns:
        dict: flattened dict
    """
    items = []
    for k, v in nested_dict.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def combine_dict_elements(list_of_dicts):
    """Combine into single list the elements with the same key across several dicts

    Args:
        list_of_dicts (list): list of dicts with matching keys

    Returns:
        dict: combined single dict
    """
    combined_dict = {}

    for d in list_of_dicts:
        for k, v in d.items():
            if k not in combined_dict:
                combined_dict[k] = []
            combined_dict[k].append(v)

    return combined_dict


def flatten_combine_dict(train_history):
    """Flatten all dict of dicts and combine elements with the same key into a single list in the dict

    Args:
        train_history (dict):

    Returns:
        dict:
    """
    train_history_cp = copy.deepcopy(train_history)
    train_history_flat_comb = {}

    for k in train_history_cp:
        if all(type(el) == dict for el in train_history_cp[k]):
            flat_dict_list = [flatten_dict(d) for d in train_history_cp[k]]
            combined_dict = combine_dict_elements(flat_dict_list)

            for k_comb, v_comb in combined_dict.items():
                train_history_flat_comb[f'{k}_{k_comb}'] = v_comb
        else:
            train_history_flat_comb[k] = train_history_cp[k]

    return train_history_flat_comb
