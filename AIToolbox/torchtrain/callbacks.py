from abc import ABC, abstractmethod


class AbstractCallback(ABC):
    def __init__(self, callback_name):
        self.callback_name = callback_name

    @abstractmethod
    def execute(self, train_loop_obj):
        """

        Args:
            train_loop_obj (AIToolbox.torchtrain.train_loop.TrainLoop):

        Returns:

        """
        pass


class DummyCallback(AbstractCallback):
    def __init__(self):
        AbstractCallback.__init__(self, 'DummyCallback')

    def execute(self, train_loop_obj):
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

    def execute(self, train_loop_obj):
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
