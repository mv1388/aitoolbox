from abc import ABC, abstractmethod
import torch


"""
    Class / Functions defining the handling of a single batch and feeding it into the PyTorch model

    Such a function is supplied as an argument to the main train loop code
"""


class AbstractModelFeedDefinition(ABC):
    @abstractmethod
    def get_loss(self, model, batch_data, criterion, device):
        """Get loss during training stage

        Called from do_train() in TrainLoop

        Executed during training stage where model weights are updated based on the loss returned from this function.

        Args:
            model:
            batch_data:
            criterion:
            device:

        Returns:
            PyTorch loss
        """
        pass

    @abstractmethod
    def get_loss_eval(self, model, batch_data, criterion, device):
        """Get loss during evaluation stage

        Called from evaluate_model_loss() in TrainLoop.

        The difference compared with get_loss() is that here the backprop weight update is not done.
        This function is executed in the evaluation stage not training.

        For simple examples this function can just call the get_loss() and return its result.

        Args:
            model:
            batch_data:
            criterion:
            device:

        Returns:

        """
        pass

    @abstractmethod
    def get_predictions(self, model, batch_data, device):
        """Get predictions during evaluation stage

        Args:
            model:
            batch_data:
            device:

        Returns:
            np.array, np.array, dict: y_test.cpu(), y_pred.cpu(), metadata
        """
        pass


class QASpanSQuADModelFeedDefinition(AbstractModelFeedDefinition):
    # def __init__(self):
    #     self.feed_type = ''

    def get_loss(self, model, batch_data, criterion, device):
        paragraph_batch, paragraph_lengths, question_batch, question_lengths, span = batch_data

        paragraph_batch = paragraph_batch.to(device)
        paragraph_lengths = paragraph_lengths.to(device)
        question_batch = question_batch.to(device)
        question_lengths = question_lengths.to(device)
        span = span.to(device)

        output_start_span, output_end_span = model(paragraph_batch, question_batch, paragraph_lengths, question_lengths)

        loss1 = criterion(output_start_span, span[:, 0].long())
        loss2 = criterion(output_end_span, span[:, 1].long())
        loss = loss1 + loss2

        return loss

    def get_loss_eval(self, model, batch_data, criterion, device):
        return self.get_loss(model, batch_data, criterion, device)

    def get_predictions(self, model, batch_data, device):
        paragraph_batch, paragraph_lengths, question_batch, question_lengths, span = batch_data

        paragraph_batch = paragraph_batch.to(device)
        paragraph_lengths = paragraph_lengths.to(device)
        question_batch = question_batch.to(device)
        question_lengths = question_lengths.to(device)

        output_start_span, output_end_span = model(paragraph_batch, question_batch, paragraph_lengths, question_lengths)

        _, output_start_span_idx = output_start_span.max(1)
        _, output_end_span_idx = output_end_span.max(1)

        y_test = span
        y_pred = torch.stack((output_start_span_idx, output_end_span_idx), 1)

        metadata = None

        return y_test.cpu(), y_pred.cpu(), metadata


class MachineTranslationFeedDefinition(AbstractModelFeedDefinition):
    def get_loss(self, model, batch_data, criterion, device):
        raise NotImplementedError

    def get_loss_eval(self, model, batch_data, criterion, device):
        raise NotImplementedError

    def get_predictions(self, model, batch_data, device):
        raise NotImplementedError


class TextClassificationFeedDefinition(AbstractModelFeedDefinition):
    def get_loss(self, model, batch_data, criterion, device):
        raise NotImplementedError

    def get_loss_eval(self, model, batch_data, criterion, device):
        raise NotImplementedError

    def get_predictions(self, model, batch_data, device):
        raise NotImplementedError


class ImageClassificationFeedDefinition(AbstractModelFeedDefinition):
    def get_loss(self, model, batch_data, criterion, device):
        raise NotImplementedError

    def get_loss_eval(self, model, batch_data, criterion, device):
        raise NotImplementedError

    def get_predictions(self, model, batch_data, device):
        raise NotImplementedError
