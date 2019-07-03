import os
import numpy as np

from AIToolbox.cloud.AWS.model_save import PyTorchS3ModelSaver
from AIToolbox.cloud.GoogleCloud.model_save import PyTorchGoogleStorageModelSaver
from AIToolbox.experiment_save.local_save.local_model_save import PyTorchLocalModelSaver, LocalSubOptimalModelRemover
from AIToolbox.experiment_save.experiment_saver import FullPyTorchExperimentS3Saver, FullPyTorchExperimentGoogleStorageSaver
from AIToolbox.experiment_save.local_experiment_saver import FullPyTorchExperimentLocalSaver
from AIToolbox.experiment_save.result_package.abstract_result_packages import AbstractResultPackage


class AbstractCallback:
    def __init__(self, callback_name, execution_order=0):
        """Abstract callback class that all actual callback classes have to inherit from

        In the inherited callback classes the callback methods should be overwritten and used to implement desired
        callback functionality at specific points of the train loop.

        Args:
            callback_name (str): name of the callback
            execution_order (int): order of the callback execution. If all the used callbacks have the orders set to 0,
                than the callbacks are executed in the order they were registered.
        """
        self.callback_name = callback_name
        self.execution_order = execution_order
        self.train_loop_obj = None

    def register_train_loop_object(self, train_loop_obj):
        """Introduce the reference to the encapsulating trainloop so that the callback has access to the
            low level functionality of the trainloop

        The registration is normally handled by the callback handler found inside the train loops. The handler is
        responsible for all the callback orchestration of the callbacks inside the trainloops.

        Args:
            train_loop_obj (AIToolbox.torchtrain.train_loop.TrainLoop): reference to the encapsulating trainloop

        Returns:
            AbstractCallback: return the reference to the callback after it is registered
        """
        self.train_loop_obj = train_loop_obj
        self.on_train_loop_registration()
        return self

    def on_train_loop_registration(self):
        """Execute callback initialization / preparation after the train_loop_object becomes available

        Returns:

        """
        pass

    def on_epoch_begin(self):
        pass

    def on_epoch_end(self):
        pass

    def on_train_begin(self):
        pass

    def on_train_end(self):
        pass

    def on_batch_begin(self):
        pass

    def on_batch_end(self):
        pass


class ListRegisteredCallbacks(AbstractCallback):
    def __init__(self):
        AbstractCallback.__init__(self, 'Print the list of registered callbacks')

    def on_train_begin(self):
        self.train_loop_obj.callbacks_handler.print_registered_callback_names()


class EarlyStopping(AbstractCallback):
    def __init__(self, monitor='val_loss', min_delta=0., patience=0):
        """Early stopping of the training if the performance stops improving

        Args:
            monitor (str): performance measure that is tracked to decide if performance is  improving during training
            min_delta (float): by how much the performance has to improve to still keep training the model
            patience (int): how many epochs the early stopper waits after the performance stopped improving
        """
        # execution_order=99 makes sure that any performance calculation callbacks are executed before and the most
        # recent results can already be found in the train_history
        AbstractCallback.__init__(self, 'EarlyStopping', execution_order=99)
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience

        self.patience_count = self.patience
        self.best_performance = None
        self.best_epoch = 0

    def on_epoch_end(self):
        history_data = self.train_loop_obj.train_history[self.monitor]
        current_performance = history_data[-1]

        if self.best_performance is None:
            self.best_performance = current_performance
            self.best_epoch = self.train_loop_obj.epoch
        else:
            if 'loss' in self.monitor:
                if current_performance < self.best_performance - self.min_delta:
                    self.best_performance = current_performance
                    self.best_epoch = self.train_loop_obj.epoch
                    self.patience_count = self.patience
                else:
                    self.patience_count -= 1
            else:
                if current_performance > self.best_performance + self.min_delta:
                    self.best_performance = current_performance
                    self.best_epoch = self.train_loop_obj.epoch
                    self.patience_count = self.patience
                else:
                    self.patience_count -= 1

            if self.patience_count < 0:
                self.train_loop_obj.early_stop = True
                print(f'Early stopping at epoch: {self.train_loop_obj.epoch}. Best recorded epoch: {self.best_epoch}.')


class TerminateOnNaN(AbstractCallback):
    def __init__(self, monitor='loss'):
        """

        Args:
            monitor (str):
        """
        AbstractCallback.__init__(self, 'TerminateOnNaN', execution_order=98)
        self.monitor = monitor

    def on_batch_end(self):
        last_measure = self.train_loop_obj.train_history[self.monitor][-1]

        if last_measure is not None:
            if np.isnan(last_measure) or np.isinf(last_measure):
                self.train_loop_obj.early_stop = True
                print(f'Terminating on {self.monitor} = {last_measure} at epoch: {self.train_loop_obj.epoch}.')


class ModelCheckpoint(AbstractCallback):
    def __init__(self, project_name, experiment_name, local_model_result_folder_path,
                 cloud_save_mode='s3', bucket_name='model-result',
                 rm_subopt_local_models=False, num_best_checkpoints_kept=2):
        """Check-point save the model during training to disk or also to S3 / GCS cloud storage

        Args:
            project_name (str): root name of the project
            experiment_name (str): name of the particular experiment
            local_model_result_folder_path (str): root local path where project folder will be created
            cloud_save_mode (str or None): Storage destination selector.
                For AWS S3: 's3' / 'aws_s3' / 'aws'
                For Google Cloud Storage: 'gcs' / 'google_storage' / 'google storage'
                Everything else results just in local storage to disk
            bucket_name (str): name of the bucket in the cloud storage
            rm_subopt_local_models (bool or str): if True, the deciding metric is set to 'loss'. Give string metric name
                to set it as a deciding metric for suboptimal model removal. If metric name consists of substring 'loss'
                the metric minimization is done otherwise metric maximization is done
            num_best_checkpoints_kept (int): number of best performing models which are kept when removing suboptimal
                model checkpoints
        """
        AbstractCallback.__init__(self, 'Model checkpoint at end of epoch')
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.local_model_result_folder_path = os.path.expanduser(local_model_result_folder_path)
        self.rm_subopt_local_models = rm_subopt_local_models

        if self.rm_subopt_local_models is not False:
            metric_name = 'loss' if self.rm_subopt_local_models is True else self.rm_subopt_local_models
            self.subopt_model_remover = LocalSubOptimalModelRemover(metric_name,
                                                                    num_best_checkpoints_kept)

        if cloud_save_mode == 's3' or cloud_save_mode == 'aws_s3' or cloud_save_mode == 'aws':
            self.model_checkpointer = PyTorchS3ModelSaver(
                bucket_name=bucket_name, local_model_result_folder_path=self.local_model_result_folder_path,
                checkpoint_model=True
            )
        elif cloud_save_mode == 'gcs' or cloud_save_mode == 'google_storage' or cloud_save_mode == 'google storage':
            self.model_checkpointer = PyTorchGoogleStorageModelSaver(
                bucket_name=bucket_name, local_model_result_folder_path=self.local_model_result_folder_path,
                checkpoint_model=True
            )
        else:
            self.model_checkpointer = PyTorchLocalModelSaver(
                local_model_result_folder_path=self.local_model_result_folder_path, checkpoint_model=True
            )

    def on_epoch_end(self):
        model_paths = self.model_checkpointer.save_model(model=self.train_loop_obj.model,
                                                         project_name=self.project_name,
                                                         experiment_name=self.experiment_name,
                                                         experiment_timestamp=self.train_loop_obj.experiment_timestamp,
                                                         epoch=self.train_loop_obj.epoch,
                                                         protect_existing_folder=True)

        if self.rm_subopt_local_models is not False:
            _, _, model_local_path, model_weights_local_path = model_paths
            self.subopt_model_remover.decide_if_remove_suboptimal_model(self.train_loop_obj.train_history,
                                                                        [model_local_path, model_weights_local_path])


class ModelTrainEndSave(AbstractCallback):
    def __init__(self, project_name, experiment_name, local_model_result_folder_path,
                 args, val_result_package=None, test_result_package=None,
                 cloud_save_mode='s3', bucket_name='model-result'):
        """At the end of training execute model performance evaluation, build result package repot and save it
            together with the final model to local disk and possibly to S3 / GCS cloud storage

        Args:
            project_name (str): root name of the project
            experiment_name (str): name of the particular experiment
            local_model_result_folder_path (str): root local path where project folder will be created
            args (dict): used hyper-parameters
            val_result_package (AIToolbox.experiment_save.result_package.abstract_result_packages.AbstractResultPackage):
            test_result_package (AIToolbox.experiment_save.result_package.abstract_result_packages.AbstractResultPackage):
            cloud_save_mode (str or None): Storage destination selector.
                For AWS S3: 's3' / 'aws_s3' / 'aws'
                For Google Cloud Storage: 'gcs' / 'google_storage' / 'google storage'
                Everything else results just in local storage to disk
            bucket_name (str): name of the bucket in the cloud storage
        """
        # execution_order=100 to make sure that this callback is the very last one to be executed when all the
        # evaluations are already stored in the train_history
        AbstractCallback.__init__(self, 'Model save at the end of training', execution_order=100)
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.local_model_result_folder_path = os.path.expanduser(local_model_result_folder_path)
        self.args = args
        self.val_result_package = val_result_package
        self.test_result_package = test_result_package
        self.result_package = None

        self.check_result_packages()

        if cloud_save_mode == 's3' or cloud_save_mode == 'aws_s3' or cloud_save_mode == 'aws':
            self.results_saver = FullPyTorchExperimentS3Saver(self.project_name, self.experiment_name,
                                                              bucket_name=bucket_name,
                                                              local_model_result_folder_path=self.local_model_result_folder_path)

        elif cloud_save_mode == 'gcs' or cloud_save_mode == 'google_storage' or cloud_save_mode == 'google storage':
            self.results_saver = FullPyTorchExperimentGoogleStorageSaver(self.project_name, self.experiment_name,
                                                                         bucket_name=bucket_name,
                                                                         local_model_result_folder_path=self.local_model_result_folder_path)
        else:
            self.results_saver = FullPyTorchExperimentLocalSaver(self.project_name, self.experiment_name,
                                                                 local_model_result_folder_path=self.local_model_result_folder_path)

    def on_train_end(self):
        if self.val_result_package is not None:
            y_test, y_pred, additional_results = self.train_loop_obj.predict_on_validation_set()
            self.val_result_package.pkg_name += '_VAL'
            self.val_result_package.prepare_result_package(y_test, y_pred,
                                                           hyperparameters=self.args,
                                                           training_history=self.train_loop_obj.train_history,
                                                           additional_results=additional_results)
            self.result_package = self.val_result_package

        if self.test_result_package is not None:
            y_test_test, y_pred_test, additional_results_test = self.train_loop_obj.predict_on_test_set()
            self.test_result_package.pkg_name += '_TEST'
            self.test_result_package.prepare_result_package(y_test_test, y_pred_test,
                                                            hyperparameters=self.args,
                                                            training_history=self.train_loop_obj.train_history,
                                                            additional_results=additional_results_test)
            self.result_package = self.test_result_package + self.result_package if self.result_package is not None \
                else self.test_result_package

        self.results_saver.save_experiment(self.train_loop_obj.model, self.result_package,
                                           experiment_timestamp=self.train_loop_obj.experiment_timestamp,
                                           save_true_pred_labels=True)

    def on_train_loop_registration(self):
        if self.val_result_package is not None:
            self.val_result_package.set_experiment_dir_path_for_additional_results(self.project_name, self.experiment_name,
                                                                                   self.train_loop_obj.experiment_timestamp,
                                                                                   self.local_model_result_folder_path)
        if self.test_result_package is not None:
            self.test_result_package.set_experiment_dir_path_for_additional_results(self.project_name,
                                                                                    self.experiment_name,
                                                                                    self.train_loop_obj.experiment_timestamp,
                                                                                    self.local_model_result_folder_path)

    def check_result_packages(self):
        if self.val_result_package is None and self.test_result_package is None:
            raise ValueError("Both val_result_package and test_result_package are None. "
                             "At least one of these should be not None but actual result package.")

        if self.val_result_package is not None and not isinstance(self.val_result_package, AbstractResultPackage):
            raise TypeError(f'val_result_package {self.val_result_package} is not inherited from AbstractResultPackage')

        if self.test_result_package is not None and not isinstance(self.test_result_package, AbstractResultPackage):
            raise TypeError(f'test_result_package {self.test_result_package} is not inherited from AbstractResultPackage')
