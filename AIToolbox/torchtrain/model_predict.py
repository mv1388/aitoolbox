from abc import ABC, abstractmethod

from AIToolbox.torchtrain.train_loop import TrainLoop
from AIToolbox.cloud.AWS.results_save import S3ResultsSaver
from AIToolbox.cloud.GoogleCloud.results_save import GoogleStorageResultsSaver
from AIToolbox.experiment.local_save.local_results_save import LocalResultsSaver


class AbstractModelPredictor(ABC):
    def __init__(self, model, data_loader):
        self.model = model
        self.data_loader = data_loader

    @abstractmethod
    def model_predict(self):
        pass

    @abstractmethod
    def model_get_loss(self, loss_criterion):
        pass

    @abstractmethod
    def evaluate_model(self, result_package,
                       project_name, experiment_name, local_model_result_folder_path,
                       **kwargs):
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


class PyTorchModelPredictor(AbstractModelPredictor):
    def __init__(self, model, data_loader, callbacks=None):
        """

        Args:
            model (AIToolbox.torchtrain.model.TTModel or AIToolbox.torchtrain.model.ModelWrap): neural
                network model
            data_loader (torch.utils.data.DataLoader):
        """
        AbstractModelPredictor.__init__(self, model, data_loader)

        self.train_loop = TrainLoop(self.model, None, None, self.data_loader, None, None)
        self.train_loop.callbacks_handler.register_callbacks(callbacks)

        self.train_loop.model.to(self.train_loop.device)

    def model_predict(self):
        """

        Returns:
            (torch.Tensor, torch.Tensor, dict):
        """
        return self.train_loop.predict_on_test_set()

    def model_get_loss(self, loss_criterion):
        """

        Args:
            loss_criterion (torch.nn.modules.loss._Loss): criterion criterion during the training procedure.

        Returns:
            float: loss
        """
        self.train_loop.criterion = loss_criterion
        return self.train_loop.evaluate_loss_on_test_set()

    def evaluate_model(self, result_package,
                       project_name, experiment_name, local_model_result_folder_path,
                       cloud_save_mode='s3', bucket_name='model-result', save_true_pred_labels=False):
        """

        Args:
            result_package (AIToolbox.experiment.result_package.abstract_result_packages.AbstractResultPackage):
            project_name (str):
            experiment_name (str):
            local_model_result_folder_path (str):
            cloud_save_mode (str):
            bucket_name (str):
            save_true_pred_labels (bool):

        Returns:

        """
        LocalResultsSaver.create_experiment_local_results_folder(project_name, experiment_name,
                                                                 self.train_loop.experiment_timestamp,
                                                                 local_model_result_folder_path)
        result_package.set_experiment_dir_path_for_additional_results(project_name=project_name,
                                                                      experiment_name=experiment_name,
                                                                      experiment_timestamp=self.train_loop.experiment_timestamp,
                                                                      local_model_result_folder_path=local_model_result_folder_path)

        evaluated_result_package = self.evaluate_result_package(result_package, return_result_package=True)

        if cloud_save_mode == 's3' or cloud_save_mode == 'aws_s3' or cloud_save_mode == 'aws':
            results_saver = S3ResultsSaver(bucket_name, local_model_result_folder_path)
        elif cloud_save_mode == 'gcs' or cloud_save_mode == 'google_storage' or cloud_save_mode == 'google storage':
            results_saver = GoogleStorageResultsSaver(bucket_name, local_model_result_folder_path)
        else:
            results_saver = LocalResultsSaver(local_model_result_folder_path)

        results_saver.save_experiment_results(evaluated_result_package,
                                              project_name=project_name, experiment_name=experiment_name,
                                              experiment_timestamp=self.train_loop.experiment_timestamp,
                                              save_true_pred_labels=save_true_pred_labels)

    def evaluate_result_package(self, result_package, return_result_package=True):
        """

        Args:
            result_package (AIToolbox.experiment.result_package.abstract_result_packages.AbstractResultPackage):
            return_result_package (bool):

        Returns:
            AIToolbox.experiment.result_package.abstract_result_packages.AbstractResultPackage or dict:
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

    def evaluate_metric(self, metric_class, return_metric=True):
        """

        Only for really simple cases where the output from the network can be directly used for metric calculation.
        For more advanced cases where the network output needs to be preprocessed before the metric evaluation,
        the use of the result package is preferred.

        Args:
            metric_class (AIToolbox.experiment.core_metrics.abstract_metric.AbstractBaseMetric): metric class not the object
            return_metric (bool):

        Returns:
            AIToolbox.experiment.core_metrics.abstract_metric.AbstractBaseMetric or dict:
        """
        y_test, y_pred, additional_results = self.train_loop.predict_on_test_set()

        metric_result = metric_class(y_test, y_pred)

        if return_metric:
            return metric_result
        else:
            return metric_result.get_metric_dict()

    def evaluate_metric_list(self, metrics_class_list, return_metric_list=True):
        """

        Args:
            metrics_class_list (list): list of metric classes not the objects
            return_metric_list (bool):

        Returns:
            list or dict
        """
        y_test, y_pred, additional_results = self.train_loop.predict_on_test_set()

        metric_final_results = [] if return_metric_list else {}

        for metric_class in metrics_class_list:
            metric_result = metric_class(y_test, y_pred)

            if return_metric_list:
                metric_final_results.append(metric_result)
            else:
                metric_final_results = metric_final_results + metric_result

        return metric_final_results
