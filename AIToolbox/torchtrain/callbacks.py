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
