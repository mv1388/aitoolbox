from AIToolbox.AWS.model_save import PyTorchS3ModelSaver


class AbstractCallback:
    def __init__(self, callback_name):
        self.callback_name = callback_name

    def on_epoch_begin(self, train_loop_obj):
        pass

    def on_epoch_end(self, train_loop_obj):
        pass

    def on_train_begin(self, train_loop_obj):
        pass

    def on_train_end(self, train_loop_obj):
        pass

    # Not used yet to prevent training slowdown
    # def on_batch_begin(self, batch):
    #     pass
    #
    # def on_batch_end(self, batch):
    #     pass


class DummyCallback(AbstractCallback):
    def __init__(self):
        AbstractCallback.__init__(self, 'DummyCallback')

    def on_epoch_end(self, train_loop_obj):
        """

        Args:
            train_loop_obj (AIToolbox.torchtrain.train_loop.TrainLoop):

        Returns:

        """
        if self.callback_name not in train_loop_obj.train_history:
            train_loop_obj.train_history[self.callback_name] = []

        train_loop_obj.train_history[self.callback_name].append(1000)


class EarlyStoppingCallback(AbstractCallback):
    def __init__(self, monitor='val_loss', min_delta=0, patience=0):
        AbstractCallback.__init__(self, 'EarlyStopping')
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience

        self.patience_count = self.patience
        self.best_performance = None
        self.best_epoch = 0

    def on_epoch_end(self, train_loop_obj):
        """

        Args:
            train_loop_obj (AIToolbox.torchtrain.train_loop.TrainLoop):

        Returns:

        """
        history_data = train_loop_obj.train_history[self.monitor]
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
            self.best_epoch = train_loop_obj.epoch

        else:
            if 'loss' in self.monitor:
                if current_performance < self.best_performance - self.min_delta:
                    self.best_performance = current_performance
                    self.best_epoch = train_loop_obj.epoch
                    self.patience_count = self.patience
                else:
                    self.patience_count -= 1
            else:
                if current_performance > self.best_performance + self.min_delta:
                    self.best_performance = current_performance
                    self.best_epoch = train_loop_obj.epoch
                    self.patience_count = self.patience
                else:
                    self.patience_count -= 1

            if self.patience_count < 0:
                train_loop_obj.early_stop = True

    def on_train_end(self, train_loop_obj):
        """

        Args:
            train_loop_obj (AIToolbox.torchtrain.train_loop.TrainLoop):

        Returns:

        """
        if train_loop_obj.early_stop:
            print(f'Early stopping at epoch: {train_loop_obj.epoch}. Best recorded epoch: {self.best_epoch}.')


class ModelCheckpointCallback(AbstractCallback):
    def __init__(self, local_model_result_folder_path):
        AbstractCallback.__init__(self, 'Model checkpoint at end of epoch')
        self.local_model_result_folder_path = local_model_result_folder_path

        self.model_checkpointer = PyTorchS3ModelSaver(
            local_model_result_folder_path=self.local_model_result_folder_path,
            checkpoint_model=True
        )

    def on_epoch_end(self, train_loop_obj):
        """

        Args:
            train_loop_obj (AIToolbox.torchtrain.train_loop.TrainLoopModelCheckpointEndSave):

        Returns:

        """
        self.model_checkpointer.save_model(model=train_loop_obj.model,
                                           project_name=train_loop_obj.project_name,
                                           experiment_name=train_loop_obj.experiment_name,
                                           experiment_timestamp=train_loop_obj.experiment_timestamp,
                                           epoch=train_loop_obj.epoch,
                                           protect_existing_folder=True)
