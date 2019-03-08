from AIToolbox.AWS.model_save import PyTorchS3ModelSaver
from AIToolbox.experiment_save.experiment_saver import FullPyTorchExperimentS3Saver
from AIToolbox.experiment_save.training_history import TrainingHistory


class AbstractCallback:
    def __init__(self, callback_name):
        self.callback_name = callback_name
        self.train_loop_obj = None

    def register_train_loop_object(self, train_loop_obj):
        """

        Args:
            train_loop_obj (AIToolbox.torchtrain.train_loop.TrainLoop):

        Returns:

        """
        self.train_loop_obj = train_loop_obj
        return self

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


class EarlyStoppingCallback(AbstractCallback):
    def __init__(self, monitor='val_loss', min_delta=0., patience=0):
        """

        Args:
            monitor (str):
            min_delta (float):
            patience (int):
        """
        AbstractCallback.__init__(self, 'EarlyStopping')
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience

        self.patience_count = self.patience
        self.best_performance = None
        self.best_epoch = 0

    def on_epoch_end(self):
        """

        Returns:

        """
        history_data = self.train_loop_obj.train_history[self.monitor]
        current_performance = history_data[-1]

        # if len(history_data) > self.patience:
        #     history_window = history_data[-self.patience:]
        #
        #     if 'loss' in self.monitor:
        #         if history_window[0] == min(history_window) and history_window[0] < history_window[-1]-self.patience:
        #             train_loop_obj.early_stop = True
        #     else:
        #         if history_window[0] == max(history_window) and history_window[0] > history_window[-1]+self.patience:
        #             train_loop_obj.early_stop = True

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

    def on_train_end(self):
        """

        Returns:

        """
        if self.train_loop_obj.early_stop:
            print(f'Early stopping at epoch: {self.train_loop_obj.epoch}. Best recorded epoch: {self.best_epoch}.')


class ModelCheckpointCallback(AbstractCallback):
    def __init__(self, project_name, experiment_name, local_model_result_folder_path):
        """

        Args:
            project_name (str):
            experiment_name (str):
            local_model_result_folder_path (str):
        """
        AbstractCallback.__init__(self, 'Model checkpoint at end of epoch')
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.local_model_result_folder_path = local_model_result_folder_path

        self.model_checkpointer = PyTorchS3ModelSaver(
            local_model_result_folder_path=self.local_model_result_folder_path,
            checkpoint_model=True
        )

    def on_epoch_end(self):
        """

        Returns:

        """
        self.model_checkpointer.save_model(model=self.train_loop_obj.model,
                                           project_name=self.project_name,
                                           experiment_name=self.experiment_name,
                                           experiment_timestamp=self.train_loop_obj.experiment_timestamp,
                                           epoch=self.train_loop_obj.epoch,
                                           protect_existing_folder=True)


class ModelTrainEndSaveCallback(AbstractCallback):
    def __init__(self, project_name, experiment_name, local_model_result_folder_path,
                 args, result_package):
        """

        Args:
            project_name (str):
            experiment_name (str):
            local_model_result_folder_path (str):
            args (dict):
            result_package (AIToolbox.experiment_save.result_package.AbstractResultPackage):
        """
        AbstractCallback.__init__(self, 'Model save at the end of training')
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.local_model_result_folder_path = local_model_result_folder_path
        self.args = args
        self.result_package = result_package

        self.results_saver = FullPyTorchExperimentS3Saver(self.project_name, self.experiment_name,
                                                          local_model_result_folder_path=self.local_model_result_folder_path)

    def on_train_end(self):
        """

        Returns:

        """
        train_history = self.train_loop_obj.train_history
        epoch_list = list(range(len(self.train_loop_obj.train_history[list(self.train_loop_obj.train_history.keys())[0]])))
        train_hist_pkg = TrainingHistory(train_history, epoch_list)

        y_test, y_pred = self.train_loop_obj.predict_on_validation_set()

        # result_pkg = self.result_package_class(y_test, y_pred,
        #                                        hyperparameters=self.args, training_history=train_hist_pkg)
        self.result_package.prepare_result_package(y_test, y_pred,
                                                   hyperparameters=self.args, training_history=train_hist_pkg)

        self.results_saver.save_experiment(self.train_loop_obj.model, self.result_package,
                                           experiment_timestamp=self.train_loop_obj.experiment_timestamp,
                                           save_true_pred_labels=True)
