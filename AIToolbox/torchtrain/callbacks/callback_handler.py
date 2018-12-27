
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

    def print_registered_callback_names(self):
        print('CALLBACKS:')
        for callback in self.train_loop_obj.callbacks:
            print(callback.callback_name)

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
