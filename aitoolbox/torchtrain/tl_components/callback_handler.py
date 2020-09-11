import torch

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
        self.callbacks_cache = []

    def register_callbacks(self, callbacks, cache_callbacks=False):
        """Register TrainLoop object reference inside the listed callbacks when the TrainLoop is created

        Normally, this is called from inside of the train loop by the TrainLoop itself. Basically train loop "registers"
        itself.

        Args:
            callbacks (list or None): list of callbacks
            cache_callbacks (bool): should provided callbacks be cached and not yet registered. First subsequent time
                this method is called without ``cache_callbacks`` enabled all the previously cached callbacks are added
                and also registered with the current list of callbacks.

        Returns:
            None
        """
        if cache_callbacks:
            self.callbacks_cache += callbacks if callbacks is not None else []
        else:
            if len(self.callbacks_cache) > 0:
                callbacks = self.callbacks_cache + (callbacks if callbacks is not None else [])

            self.callbacks_cache = []

            if callbacks is not None and len(callbacks) > 0:
                self.enforce_callbacks_quality(callbacks)
                self.train_loop_obj.callbacks += \
                    [cb.register_train_loop_object(self.train_loop_obj) for cb in callbacks
                     if self.train_loop_obj.device.index is None or
                     cb.device_idx_execution is None or
                     (cb.device_idx_execution is not None and cb.device_idx_execution == self.train_loop_obj.device.index)]

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

    def execute_gradient_update(self, optimizer_idx=0):
        for callback in self.train_loop_obj.callbacks:
            callback.on_after_gradient_update(optimizer_idx)

    def execute_optimizer_step(self):
        for callback in self.train_loop_obj.callbacks:
            callback.on_after_optimizer_step()

    def execute_multiprocess_start(self):
        for callback in self.train_loop_obj.callbacks:
            callback.on_multiprocess_start()

    def mp_filter_callbacks(self):
        self.train_loop_obj.callbacks = self._mp_filter_cb_list(self.train_loop_obj.callbacks)

    def _mp_filter_cb_list(self, callbacks_list):
        return [cb for cb in callbacks_list
                if cb.device_idx_execution is None or
                (cb.device_idx_execution is not None and cb.device_idx_execution == self.train_loop_obj.device.index)]

    def enforce_callbacks_quality(self, callbacks):
        for cb in callbacks:
            if not isinstance(cb, AbstractCallback):
                raise TypeError(f'Callback {cb} is not inherited from the AbstractCallback')
            
            if cb.device_idx_execution is not None and self.train_loop_obj.device.index is not None:
                if cb.device_idx_execution >= torch.cuda.device_count():
                    raise ValueError(f'Selected device_idx_execution of {cb.device_idx_execution} is too high. '
                                     f'There are only {torch.cuda.device_count()} available GPU devices. '
                                     f'Select index ranging from 0 to {torch.cuda.device_count() - 1}')

    @staticmethod
    def print_callback_info(callback_list):
        return '\n'.join([f'\t{callback.callback_name}: {type(callback)}, execution_order: {callback.execution_order}'
                          for callback in callback_list])

    def print_registered_callback_names(self):
        print(self)

    def __str__(self):
        return 'CALLBACKS:\n' + self.print_callback_info(self.train_loop_obj.callbacks)

    def __len__(self):
        return len(self.train_loop_obj.callbacks)

    def __add__(self, other):
        """

        Args:
            other (list): callbacks list

        Returns:
            BasicCallbacksHandler:
        """
        self.register_callbacks(other)
        return self

    def __iadd__(self, other):
        """

        Args:
            other (list): callbacks list

        Returns:
            BasicCallbacksHandler:
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

        Compared to `BasicCallbacksHandler`, this handler will at certain TrainLoop stage only execute those
        callbacks which have implemented the functionality intended to be executed at this particular stage.
        Thus, `CallbacksHandler` doesn't unnecessarily execute callbacks at stages they are not implemented at.

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
        self.cbs_on_multiprocess_start = []
        
        self.registered_cbs = [
            self.cbs_on_epoch_begin, self.cbs_on_epoch_end,
            self.cbs_on_train_begin, self.cbs_on_train_end,
            self.cbs_on_batch_begin,  self.cbs_on_batch_end,
            self.cbs_on_after_gradient_update, self.cbs_on_after_optimizer_step,
            self.cbs_on_multiprocess_start
        ]

    def register_callbacks(self, callbacks, cache_callbacks=False):
        """Register TrainLoop object reference inside the listed callbacks when the TrainLoop is created

        Normally, this is called from inside of the train loop by the TrainLoop itself. Basically train loop "registers"
        itself.

        Args:
            callbacks (list or None): list of callbacks
            cache_callbacks (bool): should provided callbacks be cached and not yet registered. First subsequent time
                this method is called without ``cache_callbacks`` enabled all the previously cached callbacks are added
                and also registered with the current list of callbacks.

        Returns:
            None
        """
        if cache_callbacks:
            # Just filling the self.callbacks_cache list with callbacks
            super().register_callbacks(callbacks, cache_callbacks=cache_callbacks)
        else:
            # hack since self.callbacks_cache is emptied by the super().register_callbacks()
            callbacks_cache = self.callbacks_cache
            super().register_callbacks(callbacks, cache_callbacks=cache_callbacks)
            # Retrieve cached callbacks from self.callbacks_cache and combine with current callbacks
            callbacks = callbacks_cache + (callbacks if callbacks is not None else [])
            self.callbacks_cache = []
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

    def execute_gradient_update(self, optimizer_idx=0):
        for callback in self.cbs_on_after_gradient_update:
            callback.on_after_gradient_update(optimizer_idx)

    def execute_optimizer_step(self):
        for callback in self.cbs_on_after_optimizer_step:
            callback.on_after_optimizer_step()

    def execute_multiprocess_start(self):
        for callback in self.cbs_on_multiprocess_start:
            callback.on_multiprocess_start()

    def split_on_execution_position(self, callbacks, register_train_loop=False):
        if callbacks is not None and len(callbacks) > 0:
            for callback in callbacks:
                if self.train_loop_obj.device.index is None or \
                        callback.device_idx_execution is None or \
                        (callback.device_idx_execution is not None and
                         callback.device_idx_execution == self.train_loop_obj.device.index):

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

                    if not is_empty_function(callback.on_multiprocess_start):
                        self.cbs_on_multiprocess_start.append(callback)

        for cbs_at_position in self.registered_cbs:
            if not all(0 == cb.execution_order for cb in cbs_at_position):
                cbs_at_position.sort(key=lambda cb: cb.execution_order)
                
    def mp_filter_callbacks(self):
        super().mp_filter_callbacks()
        self.cbs_on_epoch_begin = self._mp_filter_cb_list(self.cbs_on_epoch_begin)
        self.cbs_on_epoch_end = self._mp_filter_cb_list(self.cbs_on_epoch_end)
        self.cbs_on_train_begin = self._mp_filter_cb_list(self.cbs_on_train_begin)
        self.cbs_on_train_end = self._mp_filter_cb_list(self.cbs_on_train_end)
        self.cbs_on_batch_begin = self._mp_filter_cb_list(self.cbs_on_batch_begin)
        self.cbs_on_batch_end = self._mp_filter_cb_list(self.cbs_on_batch_end)
        self.cbs_on_after_gradient_update = self._mp_filter_cb_list(self.cbs_on_after_gradient_update)
        self.cbs_on_after_optimizer_step = self._mp_filter_cb_list(self.cbs_on_after_optimizer_step)
        self.cbs_on_multiprocess_start = self._mp_filter_cb_list(self.cbs_on_multiprocess_start)

        self.registered_cbs = [
            self.cbs_on_epoch_begin, self.cbs_on_epoch_end,
            self.cbs_on_train_begin, self.cbs_on_train_end,
            self.cbs_on_batch_begin, self.cbs_on_batch_end,
            self.cbs_on_after_gradient_update, self.cbs_on_after_optimizer_step,
            self.cbs_on_multiprocess_start
        ]

    def __str__(self):
        return 'CALLBACKS\n' \
               f'At on_epoch_begin:\n{self.print_callback_info(self.cbs_on_epoch_begin)}\n' \
               f'At on_epoch_end:\n{self.print_callback_info(self.cbs_on_epoch_end)}\n' \
               f'At on_train_begin:\n{self.print_callback_info(self.cbs_on_train_begin)}\n' \
               f'At on_train_end:\n{self.print_callback_info(self.cbs_on_train_end)}\n' \
               f'At on_batch_begin:\n{self.print_callback_info(self.cbs_on_batch_begin)}\n' \
               f'At on_batch_end:\n{self.print_callback_info(self.cbs_on_batch_end)}\n' \
               f'At on_after_gradient_update:\n{self.print_callback_info(self.cbs_on_after_gradient_update)}\n' \
               f'At on_after_optimizer_step:\n{self.print_callback_info(self.cbs_on_after_optimizer_step)}\n' \
               f'At on_multiprocess_start:\n{self.print_callback_info(self.cbs_on_multiprocess_start)}'
