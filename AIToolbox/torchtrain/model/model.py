from abc import ABC, abstractmethod
import torch.nn as nn


class TTFullModel(nn.Module, ABC):
    @abstractmethod
    def get_loss(self, batch_data, criterion, device):
        """Get loss during training stage

        Called from do_train() in TrainLoop

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


class TTForwardModel(TTFullModel):
    def get_loss(self, batch_data, criterion, device):
        pass
    
    def get_loss_eval(self, batch_data, criterion, device):
        return self.get_loss(batch_data, criterion, device)
    
    def get_predictions(self, batch_data, device):
        pass


# class TTModelWrap(TTFullModel):
#     def __init__(self, pytorch_model, batch_model_feed_def):
#         """
#
#         Args:
#             pytorch_model (torch.nn.modules.Module): neural network model
#             batch_model_feed_def (AIToolbox.torchtrain.batch_model_feed_defs.AbstractModelFeedDefinition): data prep
#                 definition for batched data. This definition prepares the data for each batch that gets than fed into
#                 the neural network.
#         """
#         super().__init__()

