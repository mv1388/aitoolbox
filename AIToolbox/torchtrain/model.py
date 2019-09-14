from abc import ABC, abstractmethod
import torch.nn as nn
from torch.nn.modules import Module

from AIToolbox.torchtrain.data.batch_model_feed_defs import AbstractModelFeedDefinition


class TTModel(nn.Module, ABC):
    """
    TT in TTModel --> TorchTrain Model

    User also needs to implement forward() method according to the nn.Module inheritance
    """
    @abstractmethod
    def get_loss(self, batch_data, criterion, device):
        """Get loss during training stage

        Called from fit() in TrainLoop

        Executed during training stage where model weights are updated based on the loss returned from this function.

        Args:
            batch_data:
            criterion:
            device:

        Returns:
            PyTorch loss
        """
        pass

    def get_loss_eval(self, batch_data, criterion, device):
        """Get loss during evaluation stage

        Called from evaluate_model_loss() in TrainLoop.

        The difference compared with get_loss() is that here the backprop weight update is not done.
        This function is executed in the evaluation stage not training.

        For simple examples this function can just call the get_loss() and return its result.

        Args:
            batch_data:
            criterion:
            device:

        Returns:

        """
        return self.get_loss(batch_data, criterion, device)

    @abstractmethod
    def get_predictions(self, batch_data, device):
        """Get predictions during evaluation stage

        Args:
            batch_data:
            device:

        Returns:
            np.array, np.array, dict: y_test.cpu(), y_pred.cpu(), metadata
        """
        pass


# For back-compatibility: TTFullModel was the previous name for TTModel
TTFullModel = TTModel


class ModelWrap:
    def __init__(self, model, batch_model_feed_def):
        """TrainLoop model wrapper combining PyTorch model and model feed definition

        NOTE: especially useful in the case when you want to train on multi-GPU where TTModel abstract functions
            can't be used.

        ModelWrap can be used as a replacement of TTModel when using the TrainLoop.

        Args:
            model (Module): neural network model
            batch_model_feed_def (AbstractModelFeedDefinition or None): data
                prep definition for batched data. This definition prepares the data for each batch that gets than fed
                into the neural network.
        """
        if not isinstance(model, Module):
            raise TypeError('Provided model is not inherited base PyTorch Module')
        if not isinstance(batch_model_feed_def, AbstractModelFeedDefinition):
            raise TypeError('Provided the base PyTorch model but did not give '
                            'the batch_model_feed_def inherited from AbstractModelFeedDefinition')

        self.model = model
        self.batch_model_feed_def = batch_model_feed_def
