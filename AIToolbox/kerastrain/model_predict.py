from AIToolbox.torchtrain.model_predict import AbstractModelPredictor


class KerasModelPredictor(AbstractModelPredictor):
    def __init__(self, model, data_loader):
        """

        Args:
            model:
            data_loader:
        """
        AbstractModelPredictor.__init__(self, model, data_loader)

    def model_predict(self):
        raise NotImplementedError

    def model_get_loss(self):
        raise NotImplementedError

    def evaluate_result_package(self, result_package, return_result_package=True):
        raise NotImplementedError

    def execute_batch_end_callbacks(self):
        raise NotImplementedError

    def execute_epoch_end_callbacks(self):
        raise NotImplementedError
