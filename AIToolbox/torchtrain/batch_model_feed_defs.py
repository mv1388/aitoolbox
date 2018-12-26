from abc import ABC, abstractmethod


"""
    Class / Functions defining the handling of a single batch and feeding it into the PyTorch model

    Such a function is supplied as an argument to the main train loop code
"""


class AbstractModelFeedDefinition(ABC):
    @abstractmethod
    def get_loss(self, model, batch_data, criterion, device):
        pass

    @abstractmethod
    def get_predictions(self, model, batch_data, device):
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

    def get_predictions(self, model, batch_data, device):
        raise NotImplementedError

        return y_test, y_pred
