
class CallbacksHandler:
    def __init__(self, train_loop_obj):
        """

        Args:
            train_loop_obj (AIToolbox.torchtrain.train_loop.TrainLoop):
        """
        self.train_loop_obj = train_loop_obj

    def register_callbacks(self, callbacks):
        """

        Args:
            callbacks (list):

        Returns:
            None
        """
        if callbacks is not None and len(callbacks) > 0:
            self.train_loop_obj.callbacks += [cb.register_train_loop_object(self.train_loop_obj) for cb in callbacks]

        if not all(0 == cb.execution_order for cb in self.train_loop_obj.callbacks):
            self.train_loop_obj.callbacks = sorted(self.train_loop_obj.callbacks, key=lambda cb: cb.execution_order)

    def print_registered_callback_names(self):
        print('CALLBACKS:')
        for callback in self.train_loop_obj.callbacks:
            print(f'\t{callback.callback_name}')

    def execute_epoch_begin(self):
        for callback in self.train_loop_obj.callbacks:
            callback.on_epoch_begin()

    def execute_epoch_end(self):
        for callback in self.train_loop_obj.callbacks:
            callback.on_epoch_end()

    def execute_train_begin(self):
        for callback in self.train_loop_obj.callbacks:
            callback.on_train_begin()

    def execute_train_end(self):
        for callback in self.train_loop_obj.callbacks:
            callback.on_train_end()

    def execute_batch_begin(self):
        for callback in self.train_loop_obj.callbacks:
            callback.on_batch_begin()

    def execute_batch_end(self):
        for callback in self.train_loop_obj.callbacks:
            callback.on_batch_end()

    def __str__(self):
        return 'CALLBACKS:\n' + '\n'.join([f'\t{callback.callback_name}' for callback in self.train_loop_obj.callbacks])

    def __len__(self):
        return len(self.train_loop_obj.callbacks)

    def __add__(self, other):
        """

        Args:
            other (list): callbacks list

        Returns:
            None
        """
        self.register_callbacks(other)

    def __iadd__(self, other):
        """

        Args:
            other (list): callbacks list

        Returns:
            None
        """
        self.register_callbacks(other)

    def __contains__(self, item):
        """

        Args:
            item:

        Returns:
            bool:
        """
        if type(item) == str:
            for cb in self.train_loop_obj.callbacks:
                if cb.callback_name == item:
                    return True
        else:
            for cb in self.train_loop_obj.callbacks:
                if type(cb) == item:
                    return True
        return False
