from AIToolbox.torchtrain.train_loop import TrainLoop
from AIToolbox.experiment_save.training_history import PyTorchTrainingHistory


class PyTorchModelReRunner:
    def __init__(self, model, data_loader, batch_model_feed_def):
        """

        Args:
            model (torch.nn.modules.Module):
            data_loader (torch.utils.data.DataLoader):
            batch_model_feed_def (AIToolbox.torchtrain.batch_model_feed_defs.AbstractModelFeedDefinition):
        """
        self.model = model
        self.data_loader = data_loader
        self.batch_model_feed_def = batch_model_feed_def

        self.train_loop = TrainLoop(self.model, None, self.data_loader, batch_model_feed_def, None, None)

    def model_predict(self):
        """

        Returns:
            (torch.Tensor, torch.Tensor):
        """
        return self.train_loop.predict_on_validation_set()

    def model_get_loss(self):
        """

        Returns:
            float:
        """
        return self.train_loop.evaluate_loss_on_validation_set()

    def evaluate_result_package(self, result_package, return_result_package=True):
        """

        Args:
            result_package (AIToolbox.experiment_save.result_package.AbstractResultPackage):
            return_result_package (bool):

        Returns:

        """
        train_history = self.train_loop.train_history
        epoch_list = list(range(len(self.train_loop.train_history[list(self.train_loop.train_history.keys())[0]])))
        train_hist_pkg = PyTorchTrainingHistory(train_history, epoch_list)

        y_test, y_pred = self.train_loop.predict_on_validation_set()

        result_package.prepare_result_package(y_test, y_pred,
                                              hyperparameters={}, training_history=train_hist_pkg)

        if return_result_package:
            return result_package
        else:
            return result_package.get_results()

    def evaluate_metric(self, metric):
        """

        Args:
            metric:

        Returns:

        """
        raise NotImplementedError

    def evaluate_metric_list(self, metrics_list):
        """

        Args:
            metrics_list (list):

        Returns:

        """
        raise NotImplementedError
