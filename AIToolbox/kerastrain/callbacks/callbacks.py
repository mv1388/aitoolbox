from keras.callbacks import Callback

from AIToolbox.AWS.model_save import KerasS3ModelSaver
from AIToolbox.experiment_save.local_model_save import KerasLocalModelSaver
from AIToolbox.experiment_save.experiment_saver import FullKerasExperimentS3Saver
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
    def __init__(self, project_name, experiment_name, local_model_result_folder_path, save_to_s3=True):
        """

        Args:
            project_name (str):
            experiment_name (str):
            local_model_result_folder_path (str):
            save_to_s3 (bool):
        """
        AbstractKerasCallback.__init__(self, 'Model checkpoint at end of epoch')
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.local_model_result_folder_path = local_model_result_folder_path
        self.save_to_s3 = save_to_s3

        if self.save_to_s3:
            self.model_checkpointer = KerasS3ModelSaver(
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
                 args, val_result_package, test_result_package, save_to_s3=True):
        """

        Args:
            project_name (str):
            experiment_name (str):
            local_model_result_folder_path (str):
            args (dict):
            val_result_package (AIToolbox.experiment_save.result_package.abstract_result_packages.AbstractResultPackage):
            test_result_package (AIToolbox.experiment_save.result_package.abstract_result_packages.AbstractResultPackage):
            save_to_s3 (bool):
        """
        AbstractKerasCallback.__init__(self, 'Model save at the end of training')
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.local_model_result_folder_path = local_model_result_folder_path
        self.args = args
        self.val_result_package = val_result_package
        self.test_result_package = test_result_package
        self.result_package = None
        self.save_to_s3 = save_to_s3

        self.check_result_packages()

        if self.save_to_s3:
            self.results_saver = FullKerasExperimentS3Saver(self.project_name, self.experiment_name,
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

        # y_test, y_pred, additional_results = self.train_loop_obj.predict_on_validation_set()
        #
        # self.result_package.prepare_result_package(y_test, y_pred,
        #                                            hyperparameters=self.args, training_history=train_hist_pkg,
        #                                            additional_results=additional_results)
        #
        # self.results_saver.save_experiment(self.train_loop_obj.model, self.result_package,
        #                                    experiment_timestamp=self.train_loop_obj.experiment_timestamp,
        #                                    save_true_pred_labels=True)

        if self.val_result_package is not None:
            y_test, y_pred, additional_results = self.train_loop_obj.predict_on_validation_set()
            self.val_result_package.pkg_name += '_VAL'
            self.val_result_package.prepare_result_package(y_test, y_pred,
                                                           hyperparameters=self.args, training_history=train_hist_pkg,
                                                           additional_results=additional_results)
            self.result_package = self.val_result_package

        if self.test_result_package is not None:
            y_test_test, y_pred_test, additional_results_test = self.train_loop_obj.predict_on_test_set()
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
        self.result_package.set_experiment_dir_path_for_additional_results(self.project_name, self.experiment_name,
                                                                           self.train_loop_obj.experiment_timestamp,
                                                                           self.local_model_result_folder_path)

    def check_result_packages(self):
        if self.val_result_package is None and self.test_result_package is None:
            raise ValueError("Both val_result_package and test_result_package are None. "
                             "At least one of these should be not None but actual result package.")
