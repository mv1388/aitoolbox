
class CallbacksHandler:
    def __init__(self, train_loop_obj):
        """

        TODO: Not an optimal implementation... repeated for loops

        Args:
            train_loop_obj (AIToolbox.torchtrain.train_loop.TrainLoop):
        """
        self.train_loop_obj = train_loop_obj

    def register_callbacks(self, callbacks):
        """

        Args:
            callbacks (list):

        Returns:

        """
        if callbacks is not None and len(callbacks) > 0:
            self.train_loop_obj.callbacks += callbacks

    def execute_epoch_begin(self):
        for callback in self.train_loop_obj.callbacks:
            callback.on_epoch_begin(self.train_loop_obj)

    def execute_epoch_end(self):
        for callback in self.train_loop_obj.callbacks:
            callback.on_epoch_end(self.train_loop_obj)

    def execute_train_begin(self):
        for callback in self.train_loop_obj.callbacks:
            callback.on_train_begin(self.train_loop_obj)

    def execute_train_end(self):
        for callback in self.train_loop_obj.callbacks:
            callback.on_train_end(self.train_loop_obj)


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


# TODO: implement:
class EarlyStoppingCallback(AbstractCallback):
    def __init__(self, monitor='val_loss', min_delta=0, patience=0):
        AbstractCallback.__init__(self, 'EarlyStopping')
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience

    def on_epoch_end(self, train_loop_obj):
        """

        Args:
            train_loop_obj (AIToolbox.torchtrain.train_loop.TrainLoop):

        Returns:

        """
        history_data = train_loop_obj.train_history[self.monitor]
        history_window = history_data if len(history_data) < self.patience else history_data[-self.patience:]

        if 'loss' in self.monitor:
            pass

        raise NotImplementedError

        # train_loop_obj.early_stop = True
