from aitoolbox.torchtrain.model_predict import AbstractModelPredictor
from aitoolbox.kerastrain.model_predict import KerasModelPredictor


class TensorFlowModelPredictor(KerasModelPredictor):
    def __init__(self, model, data_loader):
        """

        Args:
            model:
            data_loader:
        """
        AbstractModelPredictor.__init__(self, model, data_loader)

        raise NotImplementedError
