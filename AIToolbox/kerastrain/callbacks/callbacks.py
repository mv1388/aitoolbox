from keras.callbacks import Callback

from AIToolbox.AWS.model_save import KerasS3ModelSaver
from AIToolbox.GoogleCloud.model_save import KerasGoogleStorageModelSaver
from AIToolbox.experiment_save.local_model_save import KerasLocalModelSaver
from AIToolbox.experiment_save.experiment_saver import FullKerasExperimentS3Saver, FullKerasExperimentGoogleStorageSaver
from AIToolbox.experiment_save.local_experiment_saver import FullKerasExperimentLocalSaver
from AIToolbox.experiment_save.training_history import TrainingHistory


class AbstractKerasCallback(Callback):
    def __init__(self, callback_name):
        Callback.__init__(self)
        self.callback_name = callback_name
        self.train_loop_obj = None

    def register_train_loop_object(self, train_loop_obj):
        """

        Args:
            train_loop_obj (AIToolbox.kerastrain.train_loop.TrainLoop):

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


class ModelCheckpointCallback(AbstractKerasCallback):
    def __init__(self, project_name, experiment_name, local_model_result_folder_path, cloud_save_mode='s3'):
        """

        Args:
            project_name (str):
            experiment_name (str):
            local_model_result_folder_path (str):
            cloud_save_mode (str or None):
        """
        AbstractKerasCallback.__init__(self, 'Model checkpoint at end of epoch')
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.local_model_result_folder_path = local_model_result_folder_path
        self.cloud_save_mode = cloud_save_mode

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
        self.model_checkpointer.save_model(model=self.model,
                                           project_name=self.project_name,
                                           experiment_name=self.experiment_name,
                                           experiment_timestamp=self.train_loop_obj.experiment_timestamp,
                                           epoch=epoch,
                                           protect_existing_folder=True)


class ModelTrainEndSaveCallback(AbstractKerasCallback):
    def __init__(self, project_name, experiment_name, local_model_result_folder_path,
                 args, result_package, cloud_save_mode='s3'):
        """

        Args:
            project_name (str):
            experiment_name (str):
            local_model_result_folder_path (str):
            args (dict):
            result_package (AIToolbox.experiment_save.result_package.abstract_result_packages.AbstractResultPackage):
            cloud_save_mode (str or None):
        """
        AbstractKerasCallback.__init__(self, 'Model save at the end of training')
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.local_model_result_folder_path = local_model_result_folder_path
        self.args = args
        self.result_package = result_package
        self.cloud_save_mode = cloud_save_mode

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
        epoch_list = self.train_loop_obj.train_history.epoch
        train_hist_pkg = TrainingHistory(train_history, epoch_list)

        y_test, y_pred, additional_results = self.train_loop_obj.predict_on_validation_set()

        self.result_package.prepare_result_package(y_test, y_pred,
                                                   hyperparameters=self.args, training_history=train_hist_pkg,
                                                   additional_results=additional_results)

        self.results_saver.save_experiment(self.train_loop_obj.model, self.result_package,
                                           experiment_timestamp=self.train_loop_obj.experiment_timestamp,
                                           save_true_pred_labels=True)

    def on_train_loop_registration(self):
        self.result_package.set_experiment_dir_path_for_additional_results(self.project_name, self.experiment_name,
                                                                           self.train_loop_obj.experiment_timestamp,
                                                                           self.local_model_result_folder_path)
