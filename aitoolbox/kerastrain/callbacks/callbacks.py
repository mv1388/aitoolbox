import os
from keras.callbacks import Callback

from aitoolbox.cloud.AWS.model_save import KerasS3ModelSaver
from aitoolbox.cloud.GoogleCloud.model_save import KerasGoogleStorageModelSaver
from aitoolbox.experiment.local_save.local_model_save import KerasLocalModelSaver, LocalSubOptimalModelRemover
from aitoolbox.experiment.experiment_saver import FullKerasExperimentS3Saver, FullKerasExperimentGoogleStorageSaver
from aitoolbox.experiment.local_experiment_saver import FullKerasExperimentLocalSaver
from aitoolbox.experiment.training_history import TrainingHistory


class AbstractKerasCallback(Callback):
    def __init__(self, callback_name):
        Callback.__init__(self)
        self.callback_name = callback_name
        self.train_loop_obj = None

    def register_train_loop_object(self, train_loop_obj):
        """

        Args:
            train_loop_obj (aitoolbox.kerastrain.train_loop.TrainLoop):

        Returns:

        """
        self.train_loop_obj = train_loop_obj
        self.on_train_loop_registration()
        return self

    def on_train_loop_registration(self):
        """Execute callback initialization / preparation after the train_loop_object becomes available

        Returns:

        """
        pass
    
    def on_train_end_train_loop(self):
        pass


class ModelCheckpoint(AbstractKerasCallback):
    def __init__(self, project_name, experiment_name, local_model_result_folder_path, cloud_save_mode='s3',
                 rm_subopt_local_models=False, num_best_checkpoints_kept=2):
        """

        Args:
            project_name (str):
            experiment_name (str):
            local_model_result_folder_path (str):
            cloud_save_mode (str or None): Storage destination selector.
                For AWS S3: 's3' / 'aws_s3' / 'aws'
                For Google Cloud Storage: 'gcs' / 'google_storage' / 'google storage'
                Everything else results just in local storage to disk
            rm_subopt_local_models (bool or str): if True, the deciding metric is set to 'loss'. Give string metric name
                to set it as a deciding metric for suboptimal model removal. If metric name consists of substring 'loss'
                the metric minimization is done otherwise metric maximization is done
            num_best_checkpoints_kept (int): number of best performing models which are kept when removing suboptimal
                model checkpoints
        """
        AbstractKerasCallback.__init__(self, 'Model checkpoint at end of epoch')
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.local_model_result_folder_path = os.path.expanduser(local_model_result_folder_path)
        self.cloud_save_mode = cloud_save_mode
        self.rm_subopt_local_models = rm_subopt_local_models
        
        if self.rm_subopt_local_models is not False:
            metric_name = 'loss' if self.rm_subopt_local_models is True else self.rm_subopt_local_models
            self.subopt_model_remover = LocalSubOptimalModelRemover(metric_name,
                                                                    num_best_checkpoints_kept)

        if self.cloud_save_mode == 's3' or self.cloud_save_mode == 'aws_s3' or self.cloud_save_mode == 'aws':
            self.model_checkpointer = KerasS3ModelSaver(
                local_model_result_folder_path=self.local_model_result_folder_path,
                checkpoint_model=True
            )
        elif self.cloud_save_mode == 'gcs' or self.cloud_save_mode == 'google_storage' or self.cloud_save_mode == 'google storage':
            self.model_checkpointer = KerasGoogleStorageModelSaver(
                local_model_result_folder_path=self.local_model_result_folder_path,
                checkpoint_model=True
            )
        else:
            self.model_checkpointer = KerasLocalModelSaver(
                local_model_result_folder_path=self.local_model_result_folder_path, checkpoint_model=True
            )

    def on_epoch_end(self, epoch, logs=None):
        model_paths = self.model_checkpointer.save_model(model=self.model,
                                                         project_name=self.project_name,
                                                         experiment_name=self.experiment_name,
                                                         experiment_timestamp=self.train_loop_obj.experiment_timestamp,
                                                         epoch=epoch,
                                                         protect_existing_folder=True)
        
        if self.rm_subopt_local_models is not False:
            _, _, model_local_path, model_weights_local_path = model_paths
            self.subopt_model_remover.decide_if_remove_suboptimal_model(self.train_loop_obj.train_history,
                                                                        [model_local_path, model_weights_local_path])


class ModelTrainEndSave(AbstractKerasCallback):
    def __init__(self, project_name, experiment_name, local_model_result_folder_path,
                 args, val_result_package, test_result_package, cloud_save_mode='s3'):
        """

        Args:
            project_name (str):
            experiment_name (str):
            local_model_result_folder_path (str):
            args (dict):
            val_result_package (aitoolbox.experiment.result_package.abstract_result_packages.AbstractResultPackage):
            test_result_package (aitoolbox.experiment.result_package.abstract_result_packages.AbstractResultPackage):
            cloud_save_mode (str or None): Storage destination selector.
                For AWS S3: 's3' / 'aws_s3' / 'aws'
                For Google Cloud Storage: 'gcs' / 'google_storage' / 'google storage'
                Everything else results just in local storage to disk
        """
        AbstractKerasCallback.__init__(self, 'Model save at the end of training')
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.local_model_result_folder_path = os.path.expanduser(local_model_result_folder_path)
        self.args = args
        self.val_result_package = val_result_package
        self.test_result_package = test_result_package
        self.result_package = None
        self.cloud_save_mode = cloud_save_mode

        self.check_result_packages()
        
        if self.cloud_save_mode == 's3' or self.cloud_save_mode == 'aws_s3' or self.cloud_save_mode == 'aws':
            self.results_saver = FullKerasExperimentS3Saver(self.project_name, self.experiment_name,
                                                            local_model_result_folder_path=self.local_model_result_folder_path)
            
        elif self.cloud_save_mode == 'gcs' or self.cloud_save_mode == 'google_storage' or self.cloud_save_mode == 'google storage':
            self.results_saver = FullKerasExperimentGoogleStorageSaver(self.project_name, self.experiment_name,
                                                                       local_model_result_folder_path=self.local_model_result_folder_path)
        else:
            self.results_saver = FullKerasExperimentLocalSaver(self.project_name, self.experiment_name,
                                                               local_model_result_folder_path=self.local_model_result_folder_path)

    def on_train_end_train_loop(self):
        """

        Returns:

        """
        train_history = self.train_loop_obj.train_history.history
        train_hist_pkg = TrainingHistory().wrap_pre_prepared_history(train_history)

        if self.val_result_package is not None:
            y_pred, y_test, additional_results = self.train_loop_obj.predict_on_validation_set()
            self.val_result_package.pkg_name += '_VAL'
            self.val_result_package.prepare_result_package(y_test, y_pred,
                                                           hyperparameters=self.args, training_history=train_hist_pkg,
                                                           additional_results=additional_results)
            self.result_package = self.val_result_package

        if self.test_result_package is not None:
            y_pred_test, y_test_test, additional_results_test = self.train_loop_obj.predict_on_test_set()
            self.test_result_package.pkg_name += '_TEST'
            self.test_result_package.prepare_result_package(y_test_test, y_pred_test,
                                                            hyperparameters=self.args, training_history=train_hist_pkg,
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
