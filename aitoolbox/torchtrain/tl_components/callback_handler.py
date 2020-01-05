from aitoolbox.torchtrain.callbacks.abstract import AbstractCallback
from aitoolbox.utils.util import is_empty_function


class BasicCallbacksHandler:
    def __init__(self, train_loop_obj):
        """Callback handler used for the callback orchestration inside the TrainLoop

        Common use of this handler is to call different methods inside the TrainLoop at different stages of the training
        process. Thus execute desired callbacks' functionality at the desired point of the training process.

        Args:
            train_loop_obj (aitoolbox.torchtrain.train_loop.TrainLoop): reference to the encapsulating TrainLoop
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
            callback.on_after_gradient_update()

    def execute_optimizer_step(self):
        for callback in self.train_loop_obj.callbacks:
            callback.on_after_optimizer_step()

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
        return self

    def __iadd__(self, other):
        """

        Args:
            other (list): callbacks list

        Returns:
            None
        """
        self.register_callbacks(other)
        return self

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


class CallbacksHandler(BasicCallbacksHandler):
    def __init__(self, train_loop_obj):
        """Callback handler used for the callback orchestration inside the TrainLoop

        Common use of this handler is to call different methods inside the TrainLoop at different stages of the training
        process. Thus execute desired callbacks' functionality at the desired point of the training process.

        Args:
            train_loop_obj (aitoolbox.torchtrain.train_loop.TrainLoop): reference to the encapsulating TrainLoop
        """
        super().__init__(train_loop_obj)

        self.cbs_on_epoch_begin = []
        self.cbs_on_epoch_end = []
        self.cbs_on_train_begin = []
        self.cbs_on_train_end = []
        self.cbs_on_batch_begin = []
        self.cbs_on_batch_end = []
        self.cbs_on_after_gradient_update = []
        self.cbs_on_after_optimizer_step = []
        
        self.registered_cbs = [
            self.cbs_on_epoch_begin, self.cbs_on_epoch_end,
            self.cbs_on_train_begin, self.cbs_on_train_end,
            self.cbs_on_batch_begin,  self.cbs_on_batch_end,
            self.cbs_on_after_gradient_update, self.cbs_on_after_optimizer_step
        ]

    def register_callbacks(self, callbacks):
        """Register TrainLoop object reference inside the listed callbacks when the TrainLoop is created

        Normally, this is called from inside of the train loop by the TrainLoop itself. Basically train loop "registers"
        itself.

        Args:
            callbacks (list): list of callbacks

        Returns:
            None
        """
        super().register_callbacks(callbacks)
        self.split_on_execution_position(callbacks, register_train_loop=False)

    def execute_epoch_begin(self):
        for callback in self.cbs_on_epoch_begin:
            callback.on_epoch_begin()

    def execute_epoch_end(self):
        for callback in self.cbs_on_epoch_end:
            callback.on_epoch_end()

    def execute_train_begin(self):
        for callback in self.cbs_on_train_begin:
            callback.on_train_begin()

    def execute_train_end(self):
        for callback in self.cbs_on_train_end:
            callback.on_train_end()

    def execute_batch_begin(self):
        for callback in self.cbs_on_batch_begin:
            callback.on_batch_begin()

    def execute_batch_end(self):
        for callback in self.cbs_on_batch_end:
            callback.on_batch_end()

    def execute_gradient_update(self):
        for callback in self.cbs_on_after_gradient_update:
            callback.on_after_gradient_update()

    def execute_optimizer_step(self):
        for callback in self.cbs_on_after_optimizer_step:
            callback.on_after_optimizer_step()

    def split_on_execution_position(self, callbacks, register_train_loop=False):
        if callbacks is not None and len(callbacks) > 0:
            for callback in callbacks:
                if register_train_loop:
                    callback = callback.register_train_loop_object(self.train_loop_obj)

                if not is_empty_function(callback.on_epoch_begin):
                    self.cbs_on_epoch_begin.append(callback)

                if not is_empty_function(callback.on_epoch_end):
                    self.cbs_on_epoch_end.append(callback)

                if not is_empty_function(callback.on_train_begin):
                    self.cbs_on_train_begin.append(callback)

                if not is_empty_function(callback.on_train_end):
                    self.cbs_on_train_end.append(callback)

                if not is_empty_function(callback.on_batch_begin):
                    self.cbs_on_batch_begin.append(callback)

                if not is_empty_function(callback.on_batch_end):
                    self.cbs_on_batch_end.append(callback)

                if not is_empty_function(callback.on_after_gradient_update):
                    self.cbs_on_after_gradient_update.append(callback)

                if not is_empty_function(callback.on_after_optimizer_step):
                    self.cbs_on_after_optimizer_step.append(callback)

        for cbs_at_position in self.registered_cbs:
            if not all(0 == cb.execution_order for cb in cbs_at_position):
                cbs_at_position.sort(key=lambda cb: cb.execution_order)

    def print_registered_callback_names(self):
        print('CALLBACKS:')
        self.print_callback_info(self.train_loop_obj.callbacks)

    def print_registered_callbacks_execution_position(self):
        print('CALLBACKS:')
        print('At on_epoch_begin')
        self.print_callback_info(self.cbs_on_epoch_begin)
        print('At on_epoch_end')
        self.print_callback_info(self.cbs_on_epoch_end)
        print('At on_train_begin')
        self.print_callback_info(self.cbs_on_train_begin)
        print('At on_train_end')
        self.print_callback_info(self.cbs_on_train_end)
        print('At on_batch_begin')
        self.print_callback_info(self.cbs_on_batch_begin)
        print('At on_batch_end')
        self.print_callback_info(self.cbs_on_batch_end)
        print('At on_after_gradient_update')
        self.print_callback_info(self.cbs_on_after_gradient_update)
        print('At on_after_optimizer_step')
        self.print_callback_info(self.cbs_on_after_optimizer_step)

    @staticmethod
    def print_callback_info(callback_list):
        for callback in callback_list:
            print(f'\t{callback.callback_name}: {type(callback)}, execution_order: {callback.execution_order}')
    
    @staticmethod
    def enforce_callback_type(callbacks):
        for cb in callbacks:
            if not isinstance(cb, AbstractCallback):
                raise TypeError(f'Callback {cb} is not inherited from the AbstractCallback')
    
    def __str__(self):
        return 'CALLBACKS:\n' + '\n'.join([f'\t{callback.callback_name}' for callback in self.train_loop_obj.callbacks])
