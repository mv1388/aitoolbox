import torch

from aitoolbox.torchtrain.callbacks.abstract import AbstractCallback
from aitoolbox.utils.util import is_empty_function


class CallbacksHandler:
    def __init__(self, train_loop_obj):
        """Callback handler used for the callback orchestration inside the TrainLoop

        The use of this handler is to call specified callback methods inside the TrainLoop at different stages of
        the training process. This executes desired callbacks' functionality at the desired point of the training
        process.

        The ``CallbacksHandler`` handler will at certain TrainLoop stage only execute those
        callback methods which have implemented the functionality intended to be executed at this particular stage.
        Thus, `CallbacksHandler` doesn't unnecessarily execute callbacks at stages they are not implemented at -
        their respective callback methods are left as ``pass`` and aren't overridden with some desired code logic.

        Args:
            train_loop_obj (aitoolbox.torchtrain.train_loop.TrainLoop): reference to the encapsulating TrainLoop
        """
        self.train_loop_obj = train_loop_obj
        self.callbacks_cache = []

        self.cbs_on_epoch_begin = []
        self.cbs_on_epoch_end = []
        self.cbs_on_train_begin = []
        self.cbs_on_train_end = []
        self.cbs_on_batch_begin = []
        self.cbs_on_batch_end = []
        self.cbs_on_after_gradient_update = []
        self.cbs_on_after_optimizer_step = []
        self.cbs_on_multiprocess_start = []
        self.cbs_on_after_batch_prediction = []

        self.registered_cbs = [
            self.cbs_on_epoch_begin, self.cbs_on_epoch_end,
            self.cbs_on_train_begin, self.cbs_on_train_end,
            self.cbs_on_batch_begin, self.cbs_on_batch_end,
            self.cbs_on_after_gradient_update, self.cbs_on_after_optimizer_step,
            self.cbs_on_multiprocess_start,
            self.cbs_on_after_batch_prediction
        ]

    def register_callbacks(self, callbacks, cache_callbacks=False, print_callbacks=False):
        """Register TrainLoop object reference inside the listed callbacks when the TrainLoop is created

        Normally, this is called from inside the train loop by the TrainLoop itself. Basically train loop "registers"
        itself with each of the provided callbacks.

        Add via append new provided callbacks to the existing ones.

        Args:
            callbacks (list or None): list of new callbacks to be added (appended)
            cache_callbacks (bool): should the provided callbacks be cached and not yet registered. First subsequent
                time this method is called without ``cache_callbacks`` enabled all the previously cached callbacks
                are added and also registered with the current list of callbacks.
            print_callbacks (bool): after registering the provided callbacks also print the list of registered callbacks
                which will be executed during the run of the train loop

        Returns:
            None
        """
        if cache_callbacks:
            # Just filling the self.callbacks_cache list with callbacks
            self.callbacks_cache += callbacks if callbacks is not None else []
        else:
            # Combine any previously cached callbacks with new callbacks
            # If there aren't any callbacks cached then the callback cache is just an empty list
            callbacks = self.callbacks_cache + (callbacks if callbacks is not None else [])
            # Empty the callbacks cache
            self.callbacks_cache = []

            if callbacks is not None and len(callbacks) > 0:
                self.enforce_callbacks_quality(callbacks)

                self.train_loop_obj.callbacks += [
                    cb.register_train_loop_object(self.train_loop_obj) for cb in callbacks
                    if self.should_enable_callback(cb)
                ]

            if not all(0 == cb.execution_order for cb in self.train_loop_obj.callbacks):
                self.train_loop_obj.callbacks = sorted(self.train_loop_obj.callbacks, key=lambda cb: cb.execution_order)

            # Note: using `callbacks` here instead of `self.train_loop_obj.callbacks` is correct.
            #   Provide original input `callbacks` to this method instead of `self.train_loop_obj.callbacks`
            #   which we added new callbacks to above. In case some callbacks were already registered at some earlier
            #   time this prevents their duplication int the execution-position-split self.registered_cbs.
            self.split_on_execution_position(callbacks, register_train_loop=False)

        if print_callbacks:
            self.print_registered_callback_names()

    def should_enable_callback(self, callback):
        """Determine if callback should be enabled and executed to be in accordance with the GPU device setting

        Always true in case of training on single device (CPU or one GPU).

        In case of multi (GPU) device training such as DDP, this function checks if a callback should be executed on
        the particular GPU device. If the callback doesn't have any ``device_idx_execution`` set than it is executed
        on all the GPUs. In case the parameter is set in the callback than this function will only be True when the set
        ``device_idx_execution`` in the callback and the train loop's GPU device index match. In other words
        the callback will be executed only in the DDP process which sits on the matching GPU.

        Args:
            callback (AbstractCallback): callback which will be checked if it should be enabled during the particular
                train loop run

        Returns:
            bool: if the provided callback should be enabled or disabled based on (GPU) device index matching.
        """
        return self.train_loop_obj.device.index is None or \
            callback.device_idx_execution is None or \
            (
                callback.device_idx_execution is not None and
                callback.device_idx_execution == self.train_loop_obj.device.index
            )

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

    def execute_after_batch_prediction(self, y_pred_batch, y_test_batch, metadata_batch, dataset_info):
        for callback in self.cbs_on_after_batch_prediction:
            callback.on_after_batch_prediction(y_pred_batch, y_test_batch, metadata_batch, dataset_info)

    def split_on_execution_position(self, callbacks, register_train_loop=False):
        if callbacks is not None and len(callbacks) > 0:
            for callback in callbacks:
                if self.should_enable_callback(callback):

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

                    if not is_empty_function(callback.on_after_batch_prediction):
                        self.cbs_on_after_batch_prediction.append(callback)

        for cbs_at_position in self.registered_cbs:
            if not all(0 == cb.execution_order for cb in cbs_at_position):
                cbs_at_position.sort(key=lambda cb: cb.execution_order)
                
    def mp_filter_callbacks(self):
        self.train_loop_obj.callbacks = self._mp_filter_cb_list(self.train_loop_obj.callbacks)

        self.cbs_on_epoch_begin = self._mp_filter_cb_list(self.cbs_on_epoch_begin)
        self.cbs_on_epoch_end = self._mp_filter_cb_list(self.cbs_on_epoch_end)
        self.cbs_on_train_begin = self._mp_filter_cb_list(self.cbs_on_train_begin)
        self.cbs_on_train_end = self._mp_filter_cb_list(self.cbs_on_train_end)
        self.cbs_on_batch_begin = self._mp_filter_cb_list(self.cbs_on_batch_begin)
        self.cbs_on_batch_end = self._mp_filter_cb_list(self.cbs_on_batch_end)
        self.cbs_on_after_gradient_update = self._mp_filter_cb_list(self.cbs_on_after_gradient_update)
        self.cbs_on_after_optimizer_step = self._mp_filter_cb_list(self.cbs_on_after_optimizer_step)
        self.cbs_on_multiprocess_start = self._mp_filter_cb_list(self.cbs_on_multiprocess_start)
        self.cbs_on_after_batch_prediction = self._mp_filter_cb_list(self.cbs_on_after_batch_prediction)

        self.registered_cbs = [
            self.cbs_on_epoch_begin, self.cbs_on_epoch_end,
            self.cbs_on_train_begin, self.cbs_on_train_end,
            self.cbs_on_batch_begin, self.cbs_on_batch_end,
            self.cbs_on_after_gradient_update, self.cbs_on_after_optimizer_step,
            self.cbs_on_multiprocess_start,
            self.cbs_on_after_batch_prediction
        ]

    def _mp_filter_cb_list(self, callbacks_list):
        return [cb for cb in callbacks_list if self.should_enable_callback(cb)]

    def enforce_callbacks_quality(self, callbacks):
        for cb in callbacks:
            if not isinstance(cb, AbstractCallback):
                raise TypeError(f'Callback {cb} is not inherited from the AbstractCallback')

            if cb.device_idx_execution is not None and self.train_loop_obj.device.index is not None:
                if cb.device_idx_execution >= torch.cuda.device_count():
                    raise ValueError(f'Selected device_idx_execution of {cb.device_idx_execution} is too high. '
                                     f'There are only {torch.cuda.device_count()} available GPU devices. '
                                     f'Select index ranging from 0 to {torch.cuda.device_count() - 1}')

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
               f'At on_multiprocess_start:\n{self.print_callback_info(self.cbs_on_multiprocess_start)}\n' \
               f'At cbs_on_after_batch_prediction:\n{self.print_callback_info(self.cbs_on_after_batch_prediction)}\n'

    @staticmethod
    def print_callback_info(callback_list):
        return '\n'.join([f'\t{callback.callback_name}: {type(callback)}, execution_order: {callback.execution_order}'
                          for callback in callback_list])

    def print_registered_callback_names(self):
        if self.train_loop_obj.ddp_training_mode:
            print(f'*** On device {self.train_loop_obj.device.index} ({self.train_loop_obj.device}) ***')
        print(self)

    def __len__(self):
        return len(self.train_loop_obj.callbacks)

    def __add__(self, other):
        """

        Args:
            other (list): callbacks list

        Returns:
            CallbacksHandler:
        """
        self.register_callbacks(other)
        return self

    def __iadd__(self, other):
        """

        Args:
            other (list): callbacks list

        Returns:
            CallbacksHandler:
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
