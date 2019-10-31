from abc import ABC, abstractmethod
import functools
import torch.nn as nn
from torch.nn.modules import Module

from aitoolbox.utils.util import copy_function
from aitoolbox.torchtrain.data.batch_model_feed_defs import AbstractModelFeedDefinition


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
            PyTorch loss
        """
        return self.get_loss(batch_data, criterion, device)

    @abstractmethod
    def get_predictions(self, batch_data, device):
        """Get predictions during evaluation stage

        Args:
            batch_data:
            device:

        Returns:
            np.array, np.array, dict: y_pred.cpu(), y_test.cpu(), metadata
        """
        pass


# For back-compatibility: TTFullModel was the previous name for TTModel
TTFullModel = TTModel


class TTModelBasic(TTModel):
    """
    Extension of the TTModel abstract class with already implemented simple loss and prediction calculation functions

    This is mainly meant to be used for simple models where during the development of multiple model architectures,
    the get_loss() and get_predictions() would always stay the same. Using TTModelBasic thus removes the need to
    constantly duplicate code in these two functions.
    """
    def get_loss(self, batch_data, criterion, device):
        *batch_input_data, targets = [data.to(device) for data in batch_data]

        predictions = self(*batch_input_data)
        loss = criterion(predictions, targets)

        return loss

    def get_predictions(self, batch_data, device):
        *batch_input_data, targets = batch_data
        batch_input_data = [data.to(device) for data in batch_input_data]

        predictions = self(batch_input_data)

        return predictions.cpu(), targets, {}


class TTDataParallel(nn.DataParallel):
    def __init__(self, module, add_model_attributes=None,
                 default_model_methods=('get_loss', 'get_loss_eval', 'get_predictions'), **kwargs):
        """torchtrain enabled DataParallel

        This DataParallel wrapper works in the same way as the original PyTorch nn.DataParallel. Furthermore it exposes
        TTModel batch data feeding definitions (additional abstract methods) to the TrainLoop while still enabling
        multi GPU training.

        Args:
            module (TTModel): neural network model
            add_model_attributes (list or tuple or None): additional TTModel attributes which need to be transferred to
                the TTDataParallel level to enable their use in the transferred/exposed class methods
            default_model_methods (list or tuple): list of core methods which are present also in TTModel abstract class
            **kwargs: additional parameters for underlying nn.DataParallel
        """
        super().__init__(module, **kwargs)

        # Core TTModel methods which every model has
        self.get_loss_fn = copy_function(module.get_loss)
        self.get_loss_eval_fn = copy_function(module.get_loss_eval)
        self.get_predictions_fn = copy_function(module.get_predictions)

        # Transfer any additional sub-methods from TTModel to TTDataParallel
        methods = list(set(dir(module))
                       - set(dir(nn.Module)) - set(vars(module)['_modules'].keys()) - set(default_model_methods))
        additional_methods = [method_name for method_name in methods if callable(getattr(module, method_name, None))]

        for method_name in additional_methods:
            setattr(self, method_name,
                    functools.partial(copy_function(getattr(module, method_name)), self))

        # Optionally transfer additional TTModel attributes to the TTDataParallel level
        if add_model_attributes is not None and isinstance(add_model_attributes, (list, tuple)):
            for attr_name in add_model_attributes:
                setattr(self, attr_name, getattr(module, attr_name))

    def get_loss(self, batch_data, criterion, device):
        return self.get_loss_fn(self, batch_data, criterion, device)

    def get_loss_eval(self, batch_data, criterion, device):
        return self.get_loss_eval_fn(self, batch_data, criterion, device)

    def get_predictions(self, batch_data, device):
        return self.get_predictions_fn(self, batch_data, device)


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
