from AIToolbox.torchtrain.callbacks.callbacks import AbstractCallback


class CallbacksHandler:
    def __init__(self, train_loop_obj):
        """Callback handler used for the callback orchestration inside the TrainLoop

        Common use of this handler is to call different methods inside the TrainLoop at different stages of the training
        process. Thus execute desired callbacks' functionality at the desired point of the training process.

        Args:
            train_loop_obj (AIToolbox.torchtrain.train_loop.TrainLoop): reference to the encapsulating TrainLoop
        """
        self.train_loop_obj = train_loop_obj

    def register_callbacks(self, callbacks):
        """Register TrainLoop object reference inside the listed callbacks when the TrainLoop is created

        Normally, this is called from inside of the train loop by the TrainLoop itself. Basically train loop "registers"
        itself.

        Args:
            callbacks (list): list of callbacks

        Returns:
            None
        """
        if callbacks is not None and len(callbacks) > 0:
            self.enforce_callback_type(callbacks)
            self.train_loop_obj.callbacks += [cb.register_train_loop_object(self.train_loop_obj) for cb in callbacks]

        if not all(0 == cb.execution_order for cb in self.train_loop_obj.callbacks):
            self.train_loop_obj.callbacks = sorted(self.train_loop_obj.callbacks, key=lambda cb: cb.execution_order)

    def print_registered_callback_names(self):
        print('CALLBACKS:')
        for callback in self.train_loop_obj.callbacks:
            print(f'\t{callback.callback_name}: {type(callback)}, execution_order: {callback.execution_order}')

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

    def execute_gradient_update(self):
        for callback in self.train_loop_obj.callbacks:
            callback.on_gradient_update()
    
    @staticmethod
    def enforce_callback_type(callbacks):
        for cb in callbacks:
            if not isinstance(cb, AbstractCallback):
                raise TypeError(f'Callback {cb} is not inherited from the AbstractCallback')
    
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
