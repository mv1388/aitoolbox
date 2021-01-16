import os

from aitoolbox.cloud.AWS.model_save import PyTorchS3ModelSaver
from aitoolbox.cloud.GoogleCloud.model_save import PyTorchGoogleStorageModelSaver
from aitoolbox.experiment.experiment_saver import FullPyTorchExperimentS3Saver, \
    FullPyTorchExperimentGoogleStorageSaver
from aitoolbox.experiment.local_experiment_saver import FullPyTorchExperimentLocalSaver
from aitoolbox.experiment.local_save.local_model_save import LocalSubOptimalModelRemover, PyTorchLocalModelSaver
from aitoolbox.experiment.result_package.abstract_result_packages import AbstractResultPackage
from aitoolbox.experiment.result_reporting.hyperparam_reporter import HyperParamSourceReporter
from aitoolbox.torchtrain.callbacks.abstract import AbstractCallback
from aitoolbox.utils import util


class ModelCheckpoint(AbstractCallback):
    def __init__(self, project_name, experiment_name, local_model_result_folder_path,
                 hyperparams,
                 cloud_save_mode='s3', bucket_name='model-result', cloud_dir_prefix='',
                 rm_subopt_local_models=False, num_best_checkpoints_kept=2):
        """Check-point save the model during training to disk or also to S3 / GCS cloud storage

        Args:
            project_name (str): root name of the project
            experiment_name (str): name of the particular experiment
            local_model_result_folder_path (str): root local path where project folder will be created
            hyperparams (dict): used hyper-parameters. When running the TrainLoop from jupyter notebook in order to
                ensure the python experiment file copying to the experiment folder, the user needs to manually
                specify the python file path as the value for the `experiment_file_path` key. If running the training
                directly from the terminal the path deduction is done automatically.
            cloud_save_mode (str or None): Storage destination selector.
                For AWS S3: 's3' / 'aws_s3' / 'aws'
                For Google Cloud Storage: 'gcs' / 'google_storage' / 'google storage'
                Everything else results just in local storage to disk
            bucket_name (str): name of the bucket in the cloud storage
            cloud_dir_prefix (str): path to the folder inside the bucket where the experiments are going to be saved
            rm_subopt_local_models (bool or str): if True, the deciding metric is set to 'loss'. Give string metric name
                to set it as a deciding metric for suboptimal model removal. If metric name consists of substring 'loss'
                the metric minimization is done otherwise metric maximization is done
            num_best_checkpoints_kept (int): number of best performing models which are kept when removing suboptimal
                model checkpoints
        """
        # execution_order=100 to make sure that this callback is the very last one to be executed when all the
        # evaluations are already stored in the train_history and especially also when schedulers have the updated state
        AbstractCallback.__init__(self, 'Model checkpoint at end of epoch', execution_order=100, device_idx_execution=0)
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.local_model_result_folder_path = os.path.expanduser(local_model_result_folder_path)
        self.hyperparams = hyperparams
        self.rm_subopt_local_models = rm_subopt_local_models

        self._hyperparams_already_saved = False

        if self.rm_subopt_local_models is not False:
            metric_name = 'loss' if self.rm_subopt_local_models is True else self.rm_subopt_local_models
            self.subopt_model_remover = LocalSubOptimalModelRemover(metric_name,
                                                                    num_best_checkpoints_kept)
        self.model_checkpointer = None
        self.cloud_save_mode = cloud_save_mode
        self.bucket_name = bucket_name
        self.cloud_dir_prefix = cloud_dir_prefix

    def on_epoch_end(self):
        self.save_hyperparams()
        model_checkpoint = {
            'model_state_dict': self.train_loop_obj.model.state_dict(),
            'optimizer_state_dict': self.train_loop_obj.optimizer.state_dict(),
            'schedulers_state_dict': [scheduler.state_dict() for scheduler in self.train_loop_obj.get_schedulers()],
            'epoch': self.train_loop_obj.epoch,
            'iteration_idx': self.train_loop_obj.total_iteration_idx,
            'hyperparams': self.hyperparams
        }
        # If AMP is used
        if self.train_loop_obj.use_amp:
            model_checkpoint['amp'] = self.train_loop_obj.amp_scaler.state_dict()

        model_paths = self.model_checkpointer.save_model(model=model_checkpoint,
                                                         project_name=self.project_name,
                                                         experiment_name=self.experiment_name,
                                                         experiment_timestamp=self.train_loop_obj.experiment_timestamp,
                                                         epoch=self.train_loop_obj.epoch,
                                                         protect_existing_folder=True)

        if self.rm_subopt_local_models is not False:
            *_, model_local_path = model_paths
            self.subopt_model_remover.decide_if_remove_suboptimal_model(self.train_loop_obj.train_history,
                                                                        [model_local_path])

    def on_train_loop_registration(self):
        if not util.function_exists(self.train_loop_obj.optimizer, 'state_dict'):
            raise AttributeError('Provided optimizer does not have the required state_dict() method which is needed'
                                 'for the saving of the model and the optimizer.')

        if self.cloud_save_mode in ['s3', 'aws_s3', 'aws']:
            self.model_checkpointer = PyTorchS3ModelSaver(
                bucket_name=self.bucket_name, cloud_dir_prefix=self.cloud_dir_prefix,
                local_model_result_folder_path=self.local_model_result_folder_path,
                checkpoint_model=True
            )
        elif self.cloud_save_mode in ['gcs', 'google_storage', 'google storage']:
            self.model_checkpointer = PyTorchGoogleStorageModelSaver(
                bucket_name=self.bucket_name, cloud_dir_prefix=self.cloud_dir_prefix,
                local_model_result_folder_path=self.local_model_result_folder_path,
                checkpoint_model=True
            )
        else:
            self.model_checkpointer = PyTorchLocalModelSaver(
                local_model_result_folder_path=self.local_model_result_folder_path, checkpoint_model=True
            )

        if not self.train_loop_obj.lazy_experiment_save:
            self.save_hyperparams()

    def save_hyperparams(self):
        if not self._hyperparams_already_saved:
            param_reporter = HyperParamSourceReporter(self.project_name, self.experiment_name,
                                                      self.train_loop_obj.experiment_timestamp,
                                                      self.local_model_result_folder_path)

            if not os.path.isfile(param_reporter.local_hyperparams_file_path):
                local_hyperparams_file_path = param_reporter.save_hyperparams_to_text_file(self.hyperparams)
                local_experiment_python_file_path = param_reporter.save_experiment_python_file(self.hyperparams)
                local_source_code_zip_path = param_reporter.save_experiment_source_files(self.hyperparams)

                # Should also save to cloud
                if type(self.model_checkpointer) != PyTorchLocalModelSaver:
                    param_reporter.copy_to_cloud_storage(local_hyperparams_file_path, self.model_checkpointer)

                    if local_experiment_python_file_path is not None:
                        param_reporter.copy_to_cloud_storage(local_experiment_python_file_path,
                                                             self.model_checkpointer,
                                                             file_name=os.path.basename(local_experiment_python_file_path))
                    if local_source_code_zip_path is not None:
                        param_reporter.copy_to_cloud_storage(local_source_code_zip_path,
                                                             self.model_checkpointer,
                                                             file_name=os.path.basename(local_source_code_zip_path))

                self._hyperparams_already_saved = True


class ModelIterationCheckpoint(ModelCheckpoint):
    def __init__(self, save_frequency,
                 project_name, experiment_name, local_model_result_folder_path,
                 hyperparams,
                 cloud_save_mode='s3', bucket_name='model-result', cloud_dir_prefix='',
                 rm_subopt_local_models=False, num_best_checkpoints_kept=2):
        """Check-point save the model during training to disk or also to S3 / GCS cloud storage

        Args:
            save_frequency (int): frequency of saving the model checkpoint every specified number of training iterations
            project_name (str): root name of the project
            experiment_name (str): name of the particular experiment
            local_model_result_folder_path (str): root local path where project folder will be created
            hyperparams (dict): used hyper-parameters. When running the TrainLoop from jupyter notebook in order to
                ensure the python experiment file copying to the experiment folder, the user needs to manually
                specify the python file path as the value for the `experiment_file_path` key. If running the training
                directly from the terminal the path deduction is done automatically.
            cloud_save_mode (str or None): Storage destination selector.
                For AWS S3: 's3' / 'aws_s3' / 'aws'
                For Google Cloud Storage: 'gcs' / 'google_storage' / 'google storage'
                Everything else results just in local storage to disk
            bucket_name (str): name of the bucket in the cloud storage
            cloud_dir_prefix (str): path to the folder inside the bucket where the experiments are going to be saved
            rm_subopt_local_models (bool or str): if True, the deciding metric is set to 'loss'. Give string metric name
                to set it as a deciding metric for suboptimal model removal. If metric name consists of substring 'loss'
                the metric minimization is done otherwise metric maximization is done
            num_best_checkpoints_kept (int): number of best performing models which are kept when removing suboptimal
                model checkpoints
        """
        super().__init__(
            project_name, experiment_name, local_model_result_folder_path,
            hyperparams,
            cloud_save_mode, bucket_name, cloud_dir_prefix,
            rm_subopt_local_models, num_best_checkpoints_kept
        )
        self.save_frequency = save_frequency

        if save_frequency < 0:
            raise ValueError(f'save_frequency can have values only >= 0. But received value {save_frequency}.')

    def on_batch_end(self):
        if self.train_loop_obj.total_iteration_idx % self.save_frequency == 0 and \
                self.train_loop_obj.total_iteration_idx > 0:
            print(f'--> Saving model checkpoint at the training iteration: {self.train_loop_obj.total_iteration_idx}')
            self.save_hyperparams()

            model_checkpoint = {
                'model_state_dict': self.train_loop_obj.model.state_dict(),
                'optimizer_state_dict': self.train_loop_obj.optimizer.state_dict(),
                'schedulers_state_dict': [scheduler.state_dict() for scheduler in
                                          self.train_loop_obj.get_schedulers()],
                'epoch': self.train_loop_obj.epoch,
                'iteration_idx': self.train_loop_obj.total_iteration_idx,
                'hyperparams': self.hyperparams
            }
            # If AMP is used
            if self.train_loop_obj.use_amp:
                model_checkpoint['amp'] = self.train_loop_obj.amp_scaler.state_dict()

            model_paths = self.model_checkpointer.save_model(
                model=model_checkpoint,
                project_name=self.project_name,
                experiment_name=self.experiment_name,
                experiment_timestamp=self.train_loop_obj.experiment_timestamp,
                epoch=self.train_loop_obj.epoch,
                iteration_idx=self.train_loop_obj.total_iteration_idx,
                protect_existing_folder=True
            )


class ModelTrainEndSave(AbstractCallback):
    def __init__(self, project_name, experiment_name, local_model_result_folder_path,
                 hyperparams, val_result_package=None, test_result_package=None,
                 cloud_save_mode='s3', bucket_name='model-result', cloud_dir_prefix=''):
        """At the end of training execute model performance evaluation, build result package report and save it
            together with the final model to local disk and possibly to S3 / GCS cloud storage

        Args:
            project_name (str): root name of the project
            experiment_name (str): name of the particular experiment
            local_model_result_folder_path (str): root local path where project folder will be created
            hyperparams (dict): used hyper-parameters. When running the TrainLoop from jupyter notebook in order to
                ensure the python experiment file copying to the experiment folder, the user needs to manually
                specify the python file path as the value for the `experiment_file_path` key. If running the training
                directly from the terminal the path deduction is done automatically.
            val_result_package (aitoolbox.experiment.result_package.abstract_result_packages.AbstractResultPackage):
            test_result_package (aitoolbox.experiment.result_package.abstract_result_packages.AbstractResultPackage):
            cloud_save_mode (str or None): Storage destination selector.
                For AWS S3: 's3' / 'aws_s3' / 'aws'
                For Google Cloud Storage: 'gcs' / 'google_storage' / 'google storage'
                Everything else results just in local storage to disk
            bucket_name (str): name of the bucket in the cloud storage
            cloud_dir_prefix (str): path to the folder inside the bucket where the experiments are going to be saved
        """
        # execution_order=101 to make sure that this callback is the very last one to be executed when all the
        # evaluations are already stored in the train_history
        AbstractCallback.__init__(self, 'Model save at the end of training', execution_order=101)
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.local_model_result_folder_path = os.path.expanduser(local_model_result_folder_path)
        self.hyperparams = hyperparams
        self.val_result_package = val_result_package
        self.test_result_package = test_result_package
        self.result_package = None

        self.check_result_packages()
        self._hyperparams_already_saved = False

        self.results_saver = None
        self.cloud_save_mode = cloud_save_mode
        self.bucket_name = bucket_name
        self.cloud_dir_prefix = cloud_dir_prefix

    def on_train_end(self):
        if not self.train_loop_obj.ddp_training_mode or self.train_loop_obj.device.index == 0:
            self.save_hyperparams()
        model_final_state = {
            'model_state_dict': self.train_loop_obj.model.state_dict(),
            'optimizer_state_dict': self.train_loop_obj.optimizer.state_dict(),
            'schedulers_state_dict': [scheduler.state_dict() for scheduler in self.train_loop_obj.get_schedulers()],
            'epoch': self.train_loop_obj.epoch,
            'iteration_idx': self.train_loop_obj.total_iteration_idx,
            'hyperparams': self.hyperparams
        }
        # If AMP is used
        if self.train_loop_obj.use_amp:
            model_final_state['amp'] = self.train_loop_obj.amp_scaler.state_dict()

        if self.val_result_package is not None:
            y_pred, y_test, additional_results = self.train_loop_obj.predict_on_validation_set()
            self.val_result_package.pkg_name += '_VAL'
            if self.val_result_package.requires_loss:
                additional_results['loss'] = self.train_loop_obj.evaluate_loss_on_validation_set()
            self.val_result_package.prepare_result_package(y_test, y_pred,
                                                           hyperparameters=self.hyperparams,
                                                           additional_results=additional_results)
            self.result_package = self.val_result_package

        if self.test_result_package is not None:
            y_pred_test, y_test_test, additional_results_test = self.train_loop_obj.predict_on_test_set()
            self.test_result_package.pkg_name += '_TEST'
            if self.test_result_package.requires_loss:
                additional_results_test['loss'] = self.train_loop_obj.evaluate_loss_on_test_set()
            self.test_result_package.prepare_result_package(y_test_test, y_pred_test,
                                                            hyperparameters=self.hyperparams,
                                                            additional_results=additional_results_test)
            self.result_package = self.test_result_package + self.result_package if self.result_package is not None \
                else self.test_result_package

        if not self.train_loop_obj.ddp_training_mode or self.train_loop_obj.device.index == 0:
            self.results_saver.save_experiment(model_final_state, self.result_package,
                                               self.train_loop_obj.train_history,
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
        if not util.function_exists(self.train_loop_obj.optimizer, 'state_dict'):
            raise AttributeError('Provided optimizer does not have the required state_dict() method which is needed'
                                 'for the saving of the model and the optimizer.')

        if self.cloud_save_mode in ['s3', 'aws_s3', 'aws']:
            self.results_saver = FullPyTorchExperimentS3Saver(
                self.project_name, self.experiment_name,
                bucket_name=self.bucket_name, cloud_dir_prefix=self.cloud_dir_prefix,
                local_model_result_folder_path=self.local_model_result_folder_path
            )
        elif self.cloud_save_mode in ['gcs', 'google_storage', 'google storage']:
            self.results_saver = FullPyTorchExperimentGoogleStorageSaver(
                self.project_name, self.experiment_name,
                bucket_name=self.bucket_name, cloud_dir_prefix=self.cloud_dir_prefix,
                local_model_result_folder_path=self.local_model_result_folder_path
            )
        else:
            self.results_saver = FullPyTorchExperimentLocalSaver(
                self.project_name, self.experiment_name,
                local_model_result_folder_path=self.local_model_result_folder_path
            )

        if not self.train_loop_obj.lazy_experiment_save and \
                (not self.train_loop_obj.ddp_training_mode or self.train_loop_obj.device.index == 0):
            self.save_hyperparams()

    def save_hyperparams(self):
        if not self._hyperparams_already_saved:
            param_reporter = HyperParamSourceReporter(self.project_name, self.experiment_name,
                                                      self.train_loop_obj.experiment_timestamp,
                                                      self.local_model_result_folder_path)

            if not os.path.isfile(param_reporter.local_hyperparams_file_path):
                local_hyperparams_file_path = param_reporter.save_hyperparams_to_text_file(self.hyperparams)
                local_experiment_python_file_path = param_reporter.save_experiment_python_file(self.hyperparams)
                local_source_code_zip_path = param_reporter.save_experiment_source_files(self.hyperparams)

                # Should also save to cloud
                if type(self.results_saver) != FullPyTorchExperimentLocalSaver:
                    param_reporter.copy_to_cloud_storage(local_hyperparams_file_path, self.results_saver.model_saver)

                    if local_experiment_python_file_path is not None:
                        param_reporter.copy_to_cloud_storage(local_experiment_python_file_path,
                                                             self.results_saver.model_saver,
                                                             file_name=os.path.basename(local_experiment_python_file_path))
                    if local_source_code_zip_path is not None:
                        param_reporter.copy_to_cloud_storage(local_source_code_zip_path,
                                                             self.results_saver.model_saver,
                                                             file_name=os.path.basename(local_source_code_zip_path))

                self._hyperparams_already_saved = True

    def check_result_packages(self):
        if self.val_result_package is None and self.test_result_package is None:
            raise ValueError("Both val_result_package and test_result_package are None. "
                             "At least one of these should be not None but actual result package.")

        if self.val_result_package is not None and not isinstance(self.val_result_package, AbstractResultPackage):
            raise TypeError(f'val_result_package {self.val_result_package} is not inherited from AbstractResultPackage')

        if self.test_result_package is not None and not isinstance(self.test_result_package, AbstractResultPackage):
            raise TypeError(f'test_result_package {self.test_result_package} is not inherited from AbstractResultPackage')
