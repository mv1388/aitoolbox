from aitoolbox.torchtrain.model_predict import AbstractModelPredictor, PyTorchModelPredictor
from aitoolbox.kerastrain.train_loop import TrainLoop


class KerasModelPredictor(PyTorchModelPredictor):
    def __init__(self, model, data_loader, optimizer, criterion, metrics):
        """

        Args:
            model:
            data_loader:
        """
        AbstractModelPredictor.__init__(self, model, data_loader)

        self.train_loop = TrainLoop(self.model, None, None, self.data_loader, optimizer, criterion, metrics)

    def model_get_loss(self, loss_criterion=None):
        if loss_criterion is not None and loss_criterion != self.train_loop.criterion:
            self.train_loop.model.compile(self.train_loop.optimizer, loss_criterion, self.train_loop.metrics)

        return self.train_loop.evaluate_loss_on_test_set()