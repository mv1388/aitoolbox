from abc import ABC, abstractmethod

from AIToolbox.torchtrain.train_loop import TrainLoop


class AbstractModelReRunner(ABC):
    def __init__(self, model, data_loader):
        self.model = model
        self.data_loader = data_loader

    @abstractmethod
    def model_predict(self):
        pass

    @abstractmethod
    def model_get_loss(self):
        pass

    @abstractmethod
    def evaluate_result_package(self, result_package, return_result_package=True):
        pass


class PyTorchModelReRunner(AbstractModelReRunner):
    def __init__(self, model, data_loader, batch_model_feed_def):
        """Use trained PyTorch model to predict on the (new) dataset

        Args:
            model (torch.nn.modules.Module): model used for prediction
            data_loader (torch.utils.data.DataLoader): specify the dataset for which the predictions are wanted
            batch_model_feed_def (AIToolbox.torchtrain.batch_model_feed_defs.AbstractModelFeedDefinition): batch data
                feed definition
        """
        AbstractModelReRunner.__init__(self, model, data_loader)
        self.batch_model_feed_def = batch_model_feed_def

        self.train_loop = TrainLoop(self.model, None, None, self.data_loader, batch_model_feed_def, None, None)

    def model_predict(self):
        """Run the dataset through the network and return true target values, target predictions and metadata

        Returns:
            (torch.Tensor, torch.Tensor, dict): y_true, y_pred, metadata
        """
        return self.train_loop.predict_on_test_set()

    def model_get_loss(self):
        """Run the dataset through the network without updating the weights and return the loss

        Returns:
            float: loss
        """
        return self.train_loop.evaluate_loss_on_test_set()

    def evaluate_result_package(self, result_package, return_result_package=True):
        """Evaluate the model performance based on the provided result package

        Args:
            result_package (AIToolbox.experiment_save.result_package.abstract_result_packages.AbstractResultPackage):
                result package defining the performance evaluation logic
            return_result_package (bool): should return a full result package or extract just the result dict

        Returns:
            AIToolbox.experiment_save.result_package.abstract_result_packages.AbstractResultPackage | dict:
        """
        y_test, y_pred, additional_results = self.train_loop.predict_on_test_set()

        result_package.prepare_result_package(y_test, y_pred,
                                              hyperparameters={}, training_history=self.train_loop.train_history,
                                              additional_results=additional_results)

        if return_result_package:
            return result_package
        else:
            return result_package.get_results()

    def evaluate_metric(self, metric):
        """

        Args:
            metric (AIToolbox.experiment_save.core_metrics.abstract_metric.AbstractBaseMetric):

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


class KerasModelReRunner(AbstractModelReRunner):
    def __init__(self, model, data_loader):
        """

        Args:
            model:
            data_loader:
        """
        AbstractModelReRunner.__init__(self, model, data_loader)

    def model_predict(self):
        raise NotImplementedError

    def model_get_loss(self):
        raise NotImplementedError

    def evaluate_result_package(self, result_package, return_result_package=True):
        raise NotImplementedError


class TensorFlowModelReRunner(AbstractModelReRunner):
    def __init__(self, model, data_loader):
        """

        Args:
            model:
            data_loader:
        """
        AbstractModelReRunner.__init__(self, model, data_loader)

    def model_predict(self):
        raise NotImplementedError

    def model_get_loss(self):
        raise NotImplementedError

    def evaluate_result_package(self, result_package, return_result_package=True):
        raise NotImplementedError
