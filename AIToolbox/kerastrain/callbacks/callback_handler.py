from AIToolbox.kerastrain.callbacks.callbacks import AbstractKerasCallback


class CallbacksHandler:
    def __init__(self, train_loop_obj):
        """

        Args:
            train_loop_obj (AIToolbox.kerastrain.train_loop.TrainLoop):

        """
        self.train_loop_obj = train_loop_obj

    def register_callbacks(self, callbacks):
        """

        Args:
            callbacks (list):

        Returns:

        """
        if callbacks is not None and len(callbacks) > 0:
            self.train_loop_obj.callbacks += \
                [cb.register_train_loop_object(self.train_loop_obj) if isinstance(cb, AbstractKerasCallback) else cb
                 for cb in callbacks]

    def print_registered_callback_names(self):
        """
        
        Returns:
            None

        """
        print('CALLBACKS:')
        for callback in self.train_loop_obj.callbacks:
            print(callback.callback_name)

    def execute_train_end_train_loop(self):
        for callback in self.train_loop_obj.callbacks:
            if isinstance(callback, AbstractKerasCallback):
                callback.on_train_end_train_loop()
