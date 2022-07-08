from abc import ABC, abstractmethod


# Class / Functions defining the handling of a single batch and feeding it into the PyTorch model
#
# Such a function is supplied as an argument to the main train loop code


class AbstractModelFeedDefinition(ABC):
    """
    Model Feed Definition

    The primary way of defining the model for TrainLoop training is to utilize:
        aitoolbox.torchtrain.model.TTModel

    Use of the ModelFeedDefinition is the legacy way of defining the model. However, in certain scenarios where the
    TTModel might prove to increase complexity, ModelFeedDefinition still is useful for augmenting the nn.Module with
    the logic to calculate loss and predictions.
    """

    @abstractmethod
    def get_loss(self, model, batch_data, criterion, device):
        """Get loss during training stage

        Called from fit() in TrainLoop

        Executed during training stage where model weights are updated based on the loss returned from this function.

        Args:
            model (nn.Module): neural network model
            batch_data: model input data batch
            criterion: loss criterion
            device: device on which the model is being trained

        Returns:
            PyTorch loss
        """
        pass

    def get_loss_eval(self, model, batch_data, criterion, device):
        """Get loss during evaluation stage

        Called from evaluate_model_loss() in TrainLoop.

        The difference compared with get_loss() is that here the backprop weight update is not done.
        This function is executed in the evaluation stage not training.

        For simple examples this function can just call the get_loss() and return its result.

        Args:
            model (nn.Module): neural network model
            batch_data: model input data batch
            criterion: loss criterion
            device: device on which the model is being trained

        Returns:
            PyTorch loss
        """
        return self.get_loss(model, batch_data, criterion, device)

    @abstractmethod
    def get_predictions(self, model, batch_data, device):
        """Get predictions during evaluation stage

        Args:
            model (nn.Module): neural network model
            batch_data: model input data batch
            device: device on which the model is being trained

        Returns:
            np.array, np.array, dict: y_pred.cpu(), y_test.cpu(), metadata
        """
        pass
