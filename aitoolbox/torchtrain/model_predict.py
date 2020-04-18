from aitoolbox.torchtrain.train_loop import TrainLoop
from aitoolbox.cloud.AWS.results_save import S3ResultsSaver
from aitoolbox.cloud.GoogleCloud.results_save import GoogleStorageResultsSaver
from aitoolbox.experiment.local_save.local_results_save import LocalResultsSaver
from aitoolbox.experiment.training_history import TrainingHistory

# Removed abstract class definition: https://github.com/mv1388/aitoolbox/pull/306


class PyTorchModelPredictor:
    def __init__(self, model, data_loader, callbacks=None):
        """PyTorch model predictions based on provided dataloader

        Args:
            model (aitoolbox.torchtrain.model.TTModel or aitoolbox.torchtrain.model.ModelWrap): neural
                network model
            data_loader (torch.utils.data.DataLoader): dataloader based on which the model output predictions are made
        """
        self.model = model
        self.data_loader = data_loader

        self.train_loop = TrainLoop(self.model, None, None, self.data_loader, None, None)
        self.train_loop.callbacks_handler.register_callbacks(callbacks)

        self.train_loop.model.to(self.train_loop.device)

    def model_predict(self):
        """Calculate model output predictons

        Returns:
            (torch.Tensor, torch.Tensor, dict): y_pred, y_true, metadata
        """
        return self.train_loop.predict_on_test_set()

    def model_get_loss(self, loss_criterion):
        """Calculate model's loss on the given dataloader and based on provided loss function

        Args:
            loss_criterion (torch.nn.modules.loss._Loss): criterion criterion during the training procedure

        Returns:
            float: loss
        """
        self.train_loop.criterion = loss_criterion
        return self.train_loop.evaluate_loss_on_test_set()

    def evaluate_model(self, result_package,
                       project_name, experiment_name, local_model_result_folder_path,
                       cloud_save_mode='s3', bucket_name='model-result', cloud_dir_prefix='',
                       save_true_pred_labels=False):
        """Evaluate model's performance with full experiment tracking

        Args:
            result_package (aitoolbox.experiment.result_package.abstract_result_packages.AbstractResultPackage): result
                package defining the evaluation metrics on which the model is evaluated when predicting the values
                from the provided dataloader
            project_name (str): root name of the project
            experiment_name (str): name of the particular experiment
            local_model_result_folder_path (str): root local path where project folder will be created
            cloud_save_mode (str or None): Storage destination selector.
                For AWS S3: 's3' / 'aws_s3' / 'aws'
                For Google Cloud Storage: 'gcs' / 'google_storage' / 'google storage'
                Everything else results just in local storage to disk
            bucket_name (str): name of the bucket in the cloud storage
            cloud_dir_prefix (str): path to the folder inside the bucket where the experiments are going to be saved
            save_true_pred_labels (bool): should ground truth labels also be saved

        Returns:
            None
        """
        LocalResultsSaver.create_experiment_local_results_folder(project_name, experiment_name,
                                                                 self.train_loop.experiment_timestamp,
                                                                 local_model_result_folder_path)
        result_package.set_experiment_dir_path_for_additional_results(project_name=project_name,
                                                                      experiment_name=experiment_name,
                                                                      experiment_timestamp=self.train_loop.experiment_timestamp,
                                                                      local_model_result_folder_path=local_model_result_folder_path)

        evaluated_result_package = self.evaluate_result_package(result_package, return_result_package=True)

        training_history = TrainingHistory()
        for metric_name, metric_val in evaluated_result_package.get_results().items():
            training_history.insert_single_result_into_history(metric_name, metric_val)

        if cloud_save_mode == 's3' or cloud_save_mode == 'aws_s3' or cloud_save_mode == 'aws':
            results_saver = S3ResultsSaver(bucket_name, cloud_dir_prefix, local_model_result_folder_path)
        elif cloud_save_mode == 'gcs' or cloud_save_mode == 'google_storage' or cloud_save_mode == 'google storage':
            results_saver = GoogleStorageResultsSaver(bucket_name, cloud_dir_prefix, local_model_result_folder_path)
        else:
            results_saver = LocalResultsSaver(local_model_result_folder_path)

        results_saver.save_experiment_results(evaluated_result_package,
                                              training_history=training_history,
                                              project_name=project_name, experiment_name=experiment_name,
                                              experiment_timestamp=self.train_loop.experiment_timestamp,
                                              save_true_pred_labels=save_true_pred_labels)

    def evaluate_result_package(self, result_package, return_result_package=True):
        """Evaluate model's performance based on provided Result Package

        Args:
            result_package (aitoolbox.experiment.result_package.abstract_result_packages.AbstractResultPackage):
            return_result_package (bool): if True, the full calculated result package is returned, otherwise only
                the results dict is returned

        Returns:
            aitoolbox.experiment.result_package.abstract_result_packages.AbstractResultPackage or dict: calculated
                result package or results dict
        """
        y_pred, y_test, additional_results = self.train_loop.predict_on_test_set()

        result_package.prepare_result_package(y_test, y_pred,
                                              hyperparameters={}, additional_results=additional_results)

        if return_result_package:
            return result_package
        else:
            return result_package.get_results()

    def execute_batch_end_callbacks(self):
        """Execute provided callbacks which are triggered at the end of the batch in train loop

        Returns:
            None
        """
        if len(self.train_loop.callbacks) == 0:
            print('execute_batch_end_callbacks has no effect as there are no registered callbacks')
        self.train_loop.callbacks_handler.execute_batch_end()

    def execute_epoch_end_callbacks(self):
        """Execute provided callbacks which are triggered at the end of the epoch in train loop

        Returns:
            None
        """
        if len(self.train_loop.callbacks) == 0:
            print('execute_epoch_end_callbacks has no effect as there are no registered callbacks')
        self.train_loop.callbacks_handler.execute_epoch_end()

    def evaluate_metric(self, metric_class, return_metric=True):
        """Evaluate a model with a single performance metric

        Only for really simple cases where the output from the network can be directly used for metric calculation.
        For more advanced cases where the network output needs to be preprocessed before the metric evaluation,
        the use of the result package is preferred.

        Args:
            metric_class (aitoolbox.experiment.core_metrics.abstract_metric.AbstractBaseMetric): metric class not
                the object
            return_metric (bool): if True, the full performance metric object is returned, otherwise only metric result
                dict is returned

        Returns:
            aitoolbox.experiment.core_metrics.abstract_metric.AbstractBaseMetric or dict: calculated performance metric
                or result dict
        """
        y_pred, y_test, additional_results = self.train_loop.predict_on_test_set()

        metric_result = metric_class(y_test, y_pred)

        if return_metric:
            return metric_result
        else:
            return metric_result.get_metric_dict()

    def evaluate_metric_list(self, metrics_class_list, return_metric_list=True):
        """Evaluate a model with a list of performance metrics

        Args:
            metrics_class_list (list): list of metric classes not the objects
            return_metric_list (bool): if True, the full performance metrics objects are returned, otherwise only
                metric results dict is returned

        Returns:
            list or dict: list of calculated performance metrics or results dict
        """
        y_pred, y_test, additional_results = self.train_loop.predict_on_test_set()

        metric_final_results = [] if return_metric_list else {}

        for metric_class in metrics_class_list:
            metric_result = metric_class(y_test, y_pred)

            if return_metric_list:
                metric_final_results.append(metric_result)
            else:
                metric_final_results = metric_final_results + metric_result

        return metric_final_results
