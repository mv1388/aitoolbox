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

    @abstractmethod
    def execute_batch_end_callbacks(self):
        pass

    @abstractmethod
    def execute_epoch_end_callbacks(self):
        pass


class PyTorchModelReRunner(AbstractModelReRunner):
    def __init__(self, model, data_loader, callbacks=None):
        """

        Args:
            model (AIToolbox.torchtrain.model.TTFullModel or AIToolbox.torchtrain.model.ModelWrap): neural
                network model
            data_loader (torch.utils.data.DataLoader):
        """
        AbstractModelReRunner.__init__(self, model, data_loader)

        self.train_loop = TrainLoop(self.model, None, None, self.data_loader, None, None)
        self.train_loop.callbacks_handler.register_callbacks(callbacks)

    def model_predict(self):
        """

        Returns:
            (torch.Tensor, torch.Tensor, dict):
        """
        return self.train_loop.predict_on_test_set()

    def model_get_loss(self):
        """

        Returns:
            float:
        """
        return self.train_loop.evaluate_loss_on_test_set()

    def evaluate_result_package(self, result_package, return_result_package=True):
        """

        Args:
            result_package (AIToolbox.experiment_save.result_package.abstract_result_packages.AbstractResultPackage):
            return_result_package (bool):

        Returns:

        """
        y_test, y_pred, additional_results = self.train_loop.predict_on_test_set()

        result_package.prepare_result_package(y_test, y_pred,
                                              hyperparameters={}, training_history=self.train_loop.train_history,
                                              additional_results=additional_results)

        if return_result_package:
            return result_package
        else:
            return result_package.get_results()

    def execute_batch_end_callbacks(self):
        """

        Returns:
            None
        """
        if len(self.train_loop.callbacks) == 0:
            print('execute_batch_end_callbacks has no effect as there are no registered callbacks')
        self.train_loop.callbacks_handler.execute_batch_end()

    def execute_epoch_end_callbacks(self):
        """

        Returns:
            None
        """
        if len(self.train_loop.callbacks) == 0:
            print('execute_epoch_end_callbacks has no effect as there are no registered callbacks')
        self.train_loop.callbacks_handler.execute_epoch_end()

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

    def execute_batch_end_callbacks(self):
        raise NotImplementedError

    def execute_epoch_end_callbacks(self):
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

    def execute_batch_end_callbacks(self):
        raise NotImplementedError

    def execute_epoch_end_callbacks(self):
        raise NotImplementedError
