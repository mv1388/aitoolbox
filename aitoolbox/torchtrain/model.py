from abc import ABC, abstractmethod
import torch.nn as nn

from aitoolbox.torchtrain.data.batch_model_feed_defs import AbstractModelFeedDefinition


class TTModel(nn.Module, ABC):
    """
    TTModel is an extension of core PyTorch nn.Module

    TT in TTModel --> TorchTrain Model

    In addition to the common ``forward()`` method required by the base nn.Module, the user also needs to implement
    the additional AIToolbox specific ``get_loss()`` and ``get_predictions()`` methods.

    ``transfer_model_attributes`` (list or tuple): additional TTModel attributes which need to be transferred to
    the TTDataParallel level to enable their use in the transferred/exposed class methods. When coding
    the model's __init__() method user should also fill in the string names of attributes that should be
    transferred in case the model is wrapped for DP/DDP.
    """
    def __init__(self):
        super().__init__()
        self.transfer_model_attributes = []

    @abstractmethod
    def get_loss(self, batch_data, criterion, device):
        """Get loss during training stage

        Called from fit() in TrainLoop

        Executed during training stage where model weights are updated based on the loss returned from this function.

        Args:
            batch_data: model input data batch
            criterion: loss criterion
            device: device on which the model is being trained

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
            batch_data: model input data batch
            criterion: loss criterion
            device: device on which the model is being trained

        Returns:
            PyTorch loss
        """
        return self.get_loss(batch_data, criterion, device)

    @abstractmethod
    def get_predictions(self, batch_data, device):
        """Get predictions during evaluation stage

        Args:
            batch_data: model input data batch
            device: device on which the model is making the prediction

        Returns:
            torch.Tensor, torch.Tensor, dict: y_pred.cpu(), y_test.cpu(), metadata
                in the form of dict of lists/torch.Tensors/np.arrays
        """
        pass


class TTBasicModel(TTModel):
    """Extension of the TTModel abstract class with already implemented simple loss and prediction calculation functions

    The pre-implemented get_loss() and get_predictions() will take all the provided data sources from the data loader
    except the last one as an input to the model. The last data source from the data loader will be treated as
    the target variable. (*batch_input_data, targets = batch_data)

    This base class is mainly meant to be used for simple models. TTBasicModel removes the need to constantly
    duplicate code in get_loss and get_predictions.
    """
    def get_loss(self, batch_data, criterion, device):
        *batch_input_data, targets = [data.to(device) for data in batch_data]

        predictions = self(*batch_input_data)
        loss = criterion(predictions, targets)

        return loss

    def get_predictions(self, batch_data, device):
        *batch_input_data, targets = batch_data
        batch_input_data = [data.to(device) for data in batch_input_data]

        predictions = self(*batch_input_data)

        return predictions.cpu(), targets, {}


class TTBasicMultiGPUModel(TTBasicModel):
    """Extension of the TTModel abstract class with already implemented simple loss and prediction calculation functions
        which support leveled utilization when training on multi-GPU.

    The pre-implemented get_loss() and get_predictions() will take all the provided data sources from the data loader
    except the last one as an input to the model. The last data source from the data loader will be treated as
    the target variable. (*batch_input_data, targets = batch_data)

    In the case of the get_loss() the inout into the model's forward() function will also provide `targets` and
    `criterion` arguments in order to enable calculation of the loss inside forward() function.

    The forward() function should have the following parameter signature and should finish with:

        def forward(*batch_input_data, targets=None, criterion=None):
            ... predictions calculation via the computational graph ...

            if criterion is not None:
                return criterion(predictions, targets)
            else:
                return predictions

    This base class is mainly meant to be used for simple models. TTBasicModel removes the need to constantly
    duplicate code in get_loss and get_predictions.
    """
    def get_loss(self, batch_data, criterion, device):
        *batch_input_data, targets = [data.to(device) for data in batch_data]

        loss = self(*batch_input_data, targets=targets, criterion=criterion)
        return loss


class MultiGPUModelWrap(TTBasicMultiGPUModel):
    def __init__(self, model):
        """Model wrapper optimizing the model for multi-GPU training by moving the loss calculation to the GPUs

        Args:
            model (nn.Module or TTModel): neural network model. The model should follow the basic PyTorch model
                definition where the forward() function returns predictions
        """
        TTBasicMultiGPUModel.__init__(self)
        if not isinstance(model, nn.Module):
            raise TypeError(f'Provided model not inherited from nn.Module')

        self.model = model

    def forward(self, *input_data, targets=None, criterion=None):
        """DP friendly forward abstraction on top of the wrapped model's usual forward() function

        Args:
            *input_data: whatever input data should be passed into the wrapped model's forward() function
            targets: target variables which the model is training to fit
            criterion: loss function

        Returns:
            PyTorch loss or model output predictions. If loss function criterion is provided this function returns the
                calculated loss, otherwise the model output predictions are returned
        """
        predictions = self.model(*input_data)

        if criterion is not None:
            return criterion(predictions, targets)

        return predictions


class ModelWrap:
    def __init__(self, model, batch_model_feed_def):
        """TrainLoop model wrapper combining PyTorch model and model feed definition

        NOTE: especially useful in the case when you want to train on multi-GPU where TTModel abstract functions
            can't be used.

        ModelWrap can be used as a replacement of TTModel when using the TrainLoop.

        Args:
            model (nn.Module): neural network model
            batch_model_feed_def (AbstractModelFeedDefinition or None): data
                prep definition for batched data. This definition prepares the data for each batch that gets than fed
                into the neural network.
        """
        if not isinstance(model, nn.Module):
            raise TypeError('Provided model is not inherited base PyTorch Module')
        if not isinstance(batch_model_feed_def, AbstractModelFeedDefinition):
            raise TypeError('Provided the base PyTorch model but did not give '
                            'the batch_model_feed_def inherited from AbstractModelFeedDefinition')

        self.model = model
        self.batch_model_feed_def = batch_model_feed_def
