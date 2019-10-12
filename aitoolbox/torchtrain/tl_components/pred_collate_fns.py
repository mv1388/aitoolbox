import torch


def append_predictions(y_batch, predictions):
    predictions.append(y_batch)
    return predictions


def append_concat_predictions(y_batch, predictions):
    if type(y_batch) is list:
        predictions += y_batch
    else:
        predictions.append(y_batch)

    return predictions


def torch_cat_transf(predictions):
    return torch.cat(predictions)


def not_list_torch_cat_transf(predictions):
    return torch.cat(predictions) if type(predictions) is not list else predictions
