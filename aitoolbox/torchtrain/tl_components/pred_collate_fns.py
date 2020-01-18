import torch


def append_predictions(y_batch, predictions):
    """

    Args:
        y_batch (torch.Tensor): predictions for the new batch
        predictions (list): accumulation list where all the batched predictions are appended

    Returns:
        list: predictions list with the new tensor appended
    """
    predictions.append(y_batch)
    return predictions


def append_concat_predictions(y_batch, predictions):
    """

    Args:
        y_batch (torch.Tensor or list):
        predictions (list): accumulation list where all the batched predictions are added

    Returns:
        list: predictions list with the new tensor appended
    """
    if isinstance(y_batch, list):
        predictions += y_batch
    else:
        predictions.append(y_batch)

    return predictions


def torch_cat_transf(predictions):
    """PyTorch concatenation of the given list of tensors

    Args:
        predictions (list): expects a list of torch.Tensor

    Returns:
        torch.Tensor: concatenated tensor made up of provided smaller tensors
    """
    return torch.cat(predictions)


def keep_list_transf(predictions):
    """Identity transformation of the predictions keeping them as they were

    Args:
        predictions (list): list of predictions

    Returns:
        list: returns unaltered list of predictions
    """
    return predictions
