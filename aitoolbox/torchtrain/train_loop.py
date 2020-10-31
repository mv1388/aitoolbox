from tqdm import tqdm
import os
import time
import datetime
import inspect
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules import Module
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import torch.cuda.amp as amp
try:
    import deepspeed
    from aitoolbox.torchtrain.parallel import TTDeepSpeedLight
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False

from aitoolbox.utils import dict_util
from aitoolbox.torchtrain.model import TTModel, ModelWrap
from aitoolbox.torchtrain.parallel import TTDataParallel, TTDistributedDataParallel
from aitoolbox.torchtrain.multi_loss_optim import MultiLoss, MultiOptimizer
from aitoolbox.torchtrain.data.batch_model_feed_defs import AbstractModelFeedDefinition
from aitoolbox.torchtrain.tl_components.callback_handler import CallbacksHandler
from aitoolbox.torchtrain.tl_components.ddp_handler import DDPHandler
from aitoolbox.torchtrain.callbacks.model_save import ModelCheckpoint, ModelTrainEndSave
from aitoolbox.torchtrain.schedulers.basic import AbstractScheduler
from aitoolbox.experiment.training_history import TrainingHistory
from aitoolbox.torchtrain.tl_components.model_prediction_store import ModelPredictionStore
from aitoolbox.torchtrain.tl_components.message_passing import MessageService
from aitoolbox.torchtrain.tl_components.pred_collate_fns import append_predictions, torch_cat_transf
from aitoolbox.experiment.result_package.abstract_result_packages import AbstractResultPackage


class TrainLoop:
    def __init__(self, model,
                 train_loader, validation_loader, test_loader,
                 optimizer, criterion,
                 collate_batch_pred_fn=append_predictions, pred_transform_fn=torch_cat_transf,
                 end_auto_eval=True, lazy_experiment_save=True,
                 gpu_mode='single', cuda_device_idx=None, use_amp=False):
        """Core PyTorch TrainLoop supporting the model training and target prediction

        Implements core training procedures: batch feeding into the network as part of (multi)epoch train loop,
        calculation of the loss & gradients. Apart from training related functionality the TrainLoop also implements
        the logic needed for prediction of target variables.

        Args:
            model (TTModel or ModelWrap or TTDataParallel): neural network model
            train_loader (torch.utils.data.DataLoader): data loader for train data set
            validation_loader (torch.utils.data.DataLoader or None): data loader for validation data set
            test_loader (torch.utils.data.DataLoader or None): data loader for test data set
            optimizer (torch.optim.optimizer.Optimizer or MultiOptimizer): optimizer algorithm.
            criterion (torch.nn.modules.loss._Loss or MultiLoss or None): criterion during the training procedure
            collate_batch_pred_fn (callable): collate function transforming batch predictions as they come out from the
                model
            pred_transform_fn (callable): function transforming all the produced predictions after all the batches have
                been run through the model
            end_auto_eval (bool or int): used to optionally disable otherwise automatic end of epoch/training val/test
                loss calculations. This is useful when conducting very costly experiments to save on compute time.
                Specify either True/False boolean to always run or never run after each epoch or specify an int to
                execute only every specified number of epochs.
            lazy_experiment_save (bool): when in lazy mode experiment tracking components will create the experiment
                folder only after some training results are available (possibly at the end of the first epoch) instead
                of at the beginning of training.
            gpu_mode (str): GPU training mode selection. TrainLoop supports different GPU training modes by
                specifying one of the following:

                * ``'single'``: single GPU training
                * ``'dp'``: multi-GPU training via DataParallel
                * ``'ddp'``: multi-GPU training via DistributedDataParallel
                * ``'deepspeed'``: training via the Microsoft DeepSpeed

            cuda_device_idx (int or None): CUDA device index used when training on multiple GPUs
            use_amp (bool or dict): use 16-bit Automatic Mixed Precision (AMP)

                To switch to AMP mode either:

                * set this parameter to ``True`` to use default AMP ``torch.cuda.amp.GradScaler`` initialization params
                * provide custom AMP ``torch.cuda.amp.GradScaler`` initialization parameters as a dict as this parameter
        """
        if isinstance(model, TTModel) or isinstance(model, TTDataParallel):
            self.model = model
            self.batch_model_feed_def = None
        elif type(model) == ModelWrap:
            self.model = model.model
            self.batch_model_feed_def = model.batch_model_feed_def
        else:
            raise TypeError(f"Provided model is not either inherited from TTModel or ModelWrap. "
                            f"Provided type is: {type(model)}.")

        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.collate_batch_pred_fn = collate_batch_pred_fn
        self.pred_transform_fn = pred_transform_fn
        self.end_auto_eval = end_auto_eval
        self.lazy_experiment_save = lazy_experiment_save

        self.num_optimizers = 1 if not isinstance(self.optimizer, MultiOptimizer) else len(self.optimizer)

        self.gpu_mode = gpu_mode
        self.use_amp = use_amp is True or type(use_amp) == dict
        self.amp_scaler_init = {} if use_amp is True else use_amp
        self.amp_scaler = amp.GradScaler(**self.amp_scaler_init) if self.use_amp and self.gpu_mode != 'ddp' else None
        self.use_deepspeed = False

        USE_CUDA = torch.cuda.is_available()
        cuda_suffix = ''
        if USE_CUDA and cuda_device_idx is not None:
            if cuda_device_idx >= torch.cuda.device_count():
                raise ValueError(f'Selected cuda_device_idx of {cuda_device_idx} is too high. There are only '
                                 f'{torch.cuda.device_count()} available GPU devices. Select index ranging from '
                                 f'0 to {torch.cuda.device_count() - 1}')
            cuda_suffix = f':{cuda_device_idx}'
        self.device = torch.device(f"cuda{cuda_suffix}" if USE_CUDA else "cpu")

        self.experiment_timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S')
        self.loss_batch_accum = []
        self.epoch = 0

        self.train_history = TrainingHistory(has_validation=self.validation_loader is not None)
        self.prediction_store = ModelPredictionStore(auto_purge=True)
        self.message_service = MessageService()

        self.ddp_training_mode = False
        self.ddp_handler: Optional[DDPHandler] = None

        self.callbacks = []
        self.callbacks_handler = CallbacksHandler(self)
        self.early_stop = False

        self.grad_cb_used = False

        if not isinstance(self.model, TTModel) and not isinstance(self.model, TTDataParallel) and \
                not isinstance(self.model, Module):
            raise TypeError('Provided model is not inherited from TTModel/TTDataParallel and base PyTorch Module')
        if not isinstance(self.model, TTModel) and not isinstance(self.model, TTDataParallel) and \
                isinstance(self.model, Module) and not isinstance(self.batch_model_feed_def, AbstractModelFeedDefinition):
            raise TypeError('Provided the base PyTorch model but did not give the batch_model_feed_def')
        if self.gpu_mode not in ['single', 'dp', 'ddp', 'deepspeed']:
            raise ValueError("gpu_mode parameter set to the non-supported value. Can use only the following values: "
                             "'single', 'dp', 'ddp' and 'deepspeed'")

    def fit(self, num_epochs, callbacks=None, grad_accumulation=1, **kwargs):
        """Train the model using the train loop

        This is the general API method which starts the model training. By calling this method and depending on
        the selected training mode provided as the TrainLoop's ``gpu_mode`` parameter the training will start in one
        of the following training modes:

        * Basic (CPU or single GPU) mode
        * DataParallel mode
        * DistributedDataParallel mode
        * Microsoft DeepSpeed mode

        Args:
            num_epochs (int): how many epochs the network will be trained
            callbacks (list or None): callbacks that are executed during the training run
            grad_accumulation (int): number of batches the gradients are accumulated before updating weights
            **kwargs: additional parameters for training methods:

                * :meth:`aitoolbox.torchtrain.train_loop.TrainLoop._train_dp`
                * :meth:`aitoolbox.torchtrain.train_loop.TrainLoop._train_ddp`
                * :meth:`aitoolbox.torchtrain.train_loop.TrainLoop._train_deepspeed`

                These training methods are called by the TrainLoop depending on the specified setting of the TrainLoop's
                ``gpu_mode`` parameter.

        Returns:
            TTModel or torch.nn.modules.Module or TTDataParallel or deepspeed.DeepSpeedLight: trained model
        """
        if self.gpu_mode == 'single':
            return self._train(num_epochs, callbacks=callbacks, grad_accumulation=grad_accumulation)
        elif self.gpu_mode == 'dp':
            return self._train_dp(num_epochs, callbacks=callbacks, grad_accumulation=grad_accumulation, **kwargs)
        elif self.gpu_mode == 'ddp':
            return self._train_ddp(num_epochs, callbacks=callbacks, grad_accumulation=grad_accumulation, **kwargs)
        elif self.gpu_mode == 'deepspeed':
            return self._train_deepspeed(num_epochs=num_epochs, callbacks=callbacks, **kwargs)
        else:
            raise ValueError("gpu_mode parameter set to the non-supported value. Can use only the following values: "
                             "'single', 'dp', 'ddp' and 'deepspeed'")

    def _train(self, num_epochs, callbacks=None, grad_accumulation=1):
        """Train the model using the train loop

        Args:
            num_epochs (int): how many epochs the network will be trained
            callbacks (list or None): callbacks that are executed during the training run
            grad_accumulation (int): number of batches the gradients are accumulated before updating weights

        Returns:
            TTModel or torch.nn.modules.Module or TTDataParallel or deepspeed.DeepSpeedLight: trained model
        """
        self.callbacks_handler.register_callbacks(callbacks)

        self.model = self.model.to(self.device)
        if self.criterion is not None:
            self.criterion = self.criterion.to(self.device)

        self.model.train()

        self.callbacks_handler.execute_train_begin()

        for self.epoch in range(self.epoch, num_epochs):
            if not self.ddp_training_mode or self.device.index == 0:
                print('\n\n================================================================================')
                print('================================================================================')
                print(f'Epoch: {self.epoch}')
            self.callbacks_handler.execute_epoch_begin()

            for iteration, batch_data in enumerate(tqdm(self.train_loader)):
                self.callbacks_handler.execute_batch_begin()

                # Feed batch into the model
                loss_batch = self._calculate_batch_loss(batch_data, grad_accumulation)

                # Iterate over potentially multiple optimizers
                for optimizer_idx in range(self.num_optimizers):
                    # Backward pass through the model
                    self._backward_pass(loss_batch, iteration, optimizer_idx)
                    if self.grad_cb_used:
                        self.callbacks_handler.execute_gradient_update(optimizer_idx)

                    # Optimizer step
                    self._optimizer_step(iteration, grad_accumulation, optimizer_idx)
                    # Optimizer zero grad
                    self._optimizer_zero_grad(iteration, grad_accumulation, optimizer_idx)

                if self.use_amp:
                    self.amp_scaler.update()

                self.callbacks_handler.execute_batch_end()

            # Automatic end of epoch code - reports the train and if available validation loss and executes callbacks
            self.auto_execute_end_of_epoch()
            self.callbacks_handler.execute_epoch_end()

            self.message_service.end_of_epoch_trigger()

            if self.ddp_training_mode:
                # Sync early stopping setting between multiple processes when using DDP
                # Triggers overall early stopping if at least one of the processes has triggered early stopping
                self.early_stop = sum(self.ddp_handler.mp_sync(self.early_stop).numpy()) > 0
            # self.early_stop is changed from the early stopper callback
            if self.early_stop:
                break

        self.auto_execute_end_of_training()
        self.callbacks_handler.execute_train_end()

        return self.model

    def _calculate_batch_loss(self, batch_data, grad_accumulation):
        """Push batch data through the model and calculate the batch loss

        Args:
            batch_data: input data batch
            grad_accumulation (int): number of batches the gradients are accumulated before updating weights

        Returns:
            loss: loss calculated on current batch
        """
        with amp.autocast(enabled=self.use_amp):
            if self.batch_model_feed_def is None:
                loss_batch = self.model.get_loss(batch_data, self.criterion, self.device)
            else:
                loss_batch = self.batch_model_feed_def.get_loss(self.model, batch_data,
                                                                self.criterion, self.device)
            loss_batch_log = loss_batch

            # Need to divide by the number of accumulation steps if our loss is averaged over the training samples
            loss_batch = loss_batch / grad_accumulation

        self.loss_batch_accum.append(loss_batch_log.item())

        return loss_batch

    def _backward_pass(self, loss_batch, iteration, optimizer_idx):
        """Execute backward pass from the current batch loss

        Args:
            loss_batch: loss calculated on current batch
            iteration (int): current iteration index
            optimizer_idx (int): index of the current optimizer. Mostly useful when using multiple optimizers. When
                only a single optimizer is used this parameter can be ignored.

        Returns:
            None
        """
        if self.use_amp:
            if not isinstance(loss_batch, MultiLoss):
                self.amp_scaler.scale(loss_batch).backward()
            else:
                # Multi-loss Apex AMP calculation
                loss_batch.backward_amp(self.optimizer, optimizer_idx, iteration)
        elif self.use_deepspeed:
            self.model.backward(loss_batch)
        else:
            if not isinstance(loss_batch, MultiLoss):
                loss_batch.backward()
            else:
                loss_batch.backward(optimizer_idx, iteration)

    def _optimizer_step(self, iteration, grad_accumulation, optimizer_idx):
        """Execute the optimizer step

        Args:
            iteration (int): current iteration index
            grad_accumulation (int): number of batches the gradients are accumulated before updating weights
            optimizer_idx (int): index of the current optimizer. Mostly useful when using multiple optimizers. When
                only a single optimizer is used this parameter can be ignored.

        Returns:
            None
        """
        # if (iteration + 1) % grad_accumulation == 0 or iteration == len(self.train_loader) - 1:
        if (iteration + 1) % grad_accumulation == 0:
            if self.use_amp:
                if not isinstance(self.optimizer, MultiOptimizer):
                    self.amp_scaler.step(self.optimizer)
                else:
                    raise ValueError

            elif self.use_deepspeed:
                self.model.step()
            else:
                if not isinstance(self.optimizer, MultiOptimizer):
                    self.optimizer.step()
                else:
                    self.optimizer.step(optimizer_idx, iteration)

            if self.grad_cb_used:
                self.callbacks_handler.execute_optimizer_step()

    def _optimizer_zero_grad(self, iteration, grad_accumulation, optimizer_idx):
        """Execute optimizer zero grad

        Args:
            iteration (int): current iteration index
            grad_accumulation (int): number of batches the gradients are accumulated before updating weights
            optimizer_idx (int): index of the current optimizer. Mostly useful when using multiple optimizers. When
                only a single optimizer is used this parameter can be ignored.

        Returns:
            None
        """
        # if (iteration + 1) % grad_accumulation == 0 or iteration == len(self.train_loader) - 1:
        if (iteration + 1) % grad_accumulation == 0:
            if not self.use_deepspeed:
                if not isinstance(self.optimizer, MultiOptimizer):
                    self.optimizer.zero_grad()
                else:
                    self.optimizer.zero_grad(optimizer_idx, iteration)

    def auto_execute_end_of_epoch(self):
        """Basic performance evaluation executed by default at the end of each epoch

        Mainly evaluation of the loss functions which are always present as part of the training loop.

        Returns:
            None
        """
        loss_parsed = self.parse_loss(self.loss_batch_accum)
        self._print_save_loss(loss_parsed,
                              loss_type_name='accumulated_loss',
                              loss_print_description='AVG BATCH ACCUMULATED TRAIN LOSS')
        self.loss_batch_accum = []

        if (type(self.end_auto_eval) is bool and self.end_auto_eval) or \
                (type(self.end_auto_eval) is int and self.epoch % self.end_auto_eval == 0):
            train_loss = self.evaluate_loss_on_train_set()
            self._print_save_loss(train_loss, loss_type_name='loss', loss_print_description='TRAIN LOSS')

            if self.validation_loader is not None:
                val_loss = self.evaluate_loss_on_validation_set()
                self._print_save_loss(val_loss, loss_type_name='val_loss', loss_print_description='VAL LOSS')

    def auto_execute_end_of_training(self):
        """Basic performance evaluation executed by default at the end of the training process

        Returns:
            None
        """
        if self.test_loader is not None and \
                ((type(self.end_auto_eval) is bool and self.end_auto_eval) or type(self.end_auto_eval) is int):
            test_loss = self.evaluate_loss_on_test_set()
            # To keep TrainingHistory from complaining due to the non-matching metric result lengths the checking
            # has been turned off
            self._print_save_loss(test_loss, loss_type_name='train_end_test_loss', loss_print_description='TEST LOSS')

    def parse_loss(self, loss_record):
        """Helper function to process different possible loss formats

        Primarily useful for parsing between single loss representation and the multi-loss representation.

        Args:
            loss_record (list): list losses from each processed batch

        Returns:
            np.array or dict: in the case of single loss numpy array is returned, otherwise the dict of multiple losses
                is returned
        """
        loss_names = None

        if isinstance(self.optimizer, MultiOptimizer):
            loss_names = sorted(loss_record[0].keys())
            loss_record = [[loss_dict[k] for k in loss_names] for loss_dict in loss_record]

            loss_batch_accum_avg = np.mean(loss_record, axis=0)
        else:
            loss_batch_accum_avg = np.mean(loss_record)

        if self.ddp_training_mode:
            loss_ddp_synced = self.ddp_handler.mp_sync(loss_batch_accum_avg).numpy()
            if isinstance(self.optimizer, MultiOptimizer):
                loss_batch_accum_avg = np.mean(loss_ddp_synced, axis=0)
            else:
                loss_batch_accum_avg = np.mean(loss_ddp_synced)

        if loss_names is None:
            return loss_batch_accum_avg
        else:
            return dict(zip(loss_names, loss_batch_accum_avg))

    def _print_save_loss(self, loss_parsed, loss_type_name, loss_print_description):
        """Helper function which prints information about parsed loss and saves the loss results into the history

        Args:
            loss_parsed (np.array or dict): parsed loss result either as a single value or as a dict of multiple losses
            loss_type_name (str): type of the provided loss result
            loss_print_description (str): presentation description text of the provided loss result

        Returns:
            None
        """
        # Results reporting to terminal
        if not self.ddp_training_mode or self.device.index == 0:
            loss_avg = np.mean(list(loss_parsed.values())) if isinstance(loss_parsed, dict) else loss_parsed
            print(f'{loss_print_description}: {loss_avg}')

            if isinstance(self.optimizer, MultiOptimizer) and isinstance(loss_parsed, dict):
                print(f'MULTI-LOSS {loss_print_description}:')
                for loss_name, loss_val in loss_parsed.items():
                    print(f'\t{loss_name}: {loss_val}')

        # Insert results into history
        if isinstance(self.optimizer, MultiOptimizer) and isinstance(loss_parsed, dict):
            for loss_name, loss_val in loss_parsed.items():
                self.insert_metric_result_into_history(f'{loss_type_name}_{loss_name}', loss_val)
        else:
            self.insert_metric_result_into_history(loss_type_name, loss_parsed)

    def evaluate_loss_on_train_set(self, force_prediction=False):
        """Run train dataset through the network without updating the weights and return the loss

        Args:
            force_prediction (bool): recompute the loss even if it is available in the prediction cache. This causes
                the old cached value to be overwritten.

        Returns:
            float or dict: loss, in the case of multi loss, the dict gets returned
        """
        if not self.prediction_store.has_train_loss(self.epoch) or force_prediction:
            loss = self.evaluate_model_loss(self.train_loader)
            self.prediction_store.insert_train_loss(loss, self.epoch, force_prediction)
        else:
            loss = self.prediction_store.get_train_loss(self.epoch)

        return loss

    def evaluate_loss_on_validation_set(self, force_prediction=False):
        """Run validation dataset through the network without updating the weights and return the loss

        Args:
            force_prediction (bool): recompute the loss even if it is available in the prediction cache. This causes
                the old cached value to be overwritten.

        Returns:
            float or dict: loss, in the case of multi loss, the dict gets returned
        """
        if not self.prediction_store.has_val_loss(self.epoch) or force_prediction:
            loss = self.evaluate_model_loss(self.validation_loader)
            self.prediction_store.insert_val_loss(loss, self.epoch, force_prediction)
        else:
            loss = self.prediction_store.get_val_loss(self.epoch)

        return loss

    def evaluate_loss_on_test_set(self, force_prediction=False):
        """Run test dataset through the network without updating the weights and return the loss

        Args:
            force_prediction (bool): recompute the loss even if it is available in the prediction cache. This causes
                the old cached value to be overwritten.

        Returns:
            float or dict: loss, in the case of multi loss, the dict gets returned
        """
        if not self.prediction_store.has_test_loss(self.epoch) or force_prediction:
            loss = self.evaluate_model_loss(self.test_loader)
            self.prediction_store.insert_test_loss(loss, self.epoch, force_prediction)
        else:
            loss = self.prediction_store.get_test_loss(self.epoch)

        return loss

    def evaluate_model_loss(self, data_loader):
        """Run given dataset through the network without updating the weights and return the loss

        Args:
            data_loader (torch.utils.data.DataLoader): dataloader containing the data on which the loss is calculated

        Returns:
            float or dict: loss, in the case of multi loss, the dict gets returned
        """
        self.model = self.model.to(self.device)
        if self.criterion is not None:
            self.criterion = self.criterion.to(self.device)

        self.model.eval()
        loss_avg = []

        with torch.no_grad():
            for batch_data in tqdm(data_loader):
                with amp.autocast(enabled=self.use_amp):
                    if self.batch_model_feed_def is None:
                        loss_batch = self.model.get_loss_eval(batch_data, self.criterion, self.device)
                    else:
                        loss_batch = self.batch_model_feed_def.get_loss_eval(self.model, batch_data, self.criterion,
                                                                             self.device)

                loss_avg.append(loss_batch.item())

            loss_avg = self.parse_loss(loss_avg)

        self.model.train()

        return loss_avg

    def predict_on_train_set(self, force_prediction=False):
        """Run train dataset through the network and return true target values, target predictions and metadata

        Args:
            force_prediction (bool): recompute the output prediction even if it is available in the prediction cache.
                This causes the old cached predictions to be overwritten.

        Returns:
            (torch.Tensor, torch.Tensor, dict): y_pred, y_true, metadata
        """
        if not self.prediction_store.has_train_predictions(self.epoch) or force_prediction:
            predictions = self.predict_with_model(self.train_loader)
            self.prediction_store.insert_train_predictions(predictions, self.epoch, force_prediction)
        else:
            predictions = self.prediction_store.get_train_predictions(self.epoch)

        return predictions

    def predict_on_validation_set(self, force_prediction=False):
        """Run validation dataset through the network and return true target values, target predictions and metadata

        Args:
            force_prediction (bool): recompute the output prediction even if it is available in the prediction cache.
                This causes the old cached predictions to be overwritten.

        Returns:
            (torch.Tensor, torch.Tensor, dict): y_pred, y_true, metadata
        """
        if not self.prediction_store.has_val_predictions(self.epoch) or force_prediction:
            predictions = self.predict_with_model(self.validation_loader)
            self.prediction_store.insert_val_predictions(predictions, self.epoch, force_prediction)
        else:
            predictions = self.prediction_store.get_val_predictions(self.epoch)

        return predictions

    def predict_on_test_set(self, force_prediction=False):
        """Run test dataset through the network and return true target values, target predictions and metadata

        Args:
            force_prediction (bool): recompute the output prediction even if it is available in the prediction cache.
                This causes the old cached predictions to be overwritten.

        Returns:
            (torch.Tensor, torch.Tensor, dict): y_pred, y_true, metadata
        """
        if not self.prediction_store.has_test_predictions(self.epoch) or force_prediction:
            predictions = self.predict_with_model(self.test_loader)
            self.prediction_store.insert_test_predictions(predictions, self.epoch, force_prediction)
        else:
            predictions = self.prediction_store.get_test_predictions(self.epoch)

        return predictions

    def predict_with_model(self, data_loader):
        """Run given dataset through the network and return true target values, target predictions and metadata

        Args:
            data_loader (torch.utils.data.DataLoader): dataloader containing the data on which the output predictions
                are calculated

        Returns:
            (torch.Tensor, torch.Tensor, dict): y_pred, y_true, metadata
        """
        self.model = self.model.to(self.device)

        self.model.eval()
        y_pred, y_test, metadata_list = [], [], []

        with torch.no_grad():
            for batch_data in tqdm(data_loader):
                with amp.autocast(enabled=self.use_amp):
                    if self.batch_model_feed_def is None:
                        y_pred_batch, y_test_batch, metadata_batch = self.model.get_predictions(batch_data, self.device)
                    else:
                        y_pred_batch, y_test_batch, metadata_batch = \
                            self.batch_model_feed_def.get_predictions(self.model, batch_data, self.device)

                y_pred = self.collate_batch_pred_fn(y_pred_batch, y_pred)
                y_test = self.collate_batch_pred_fn(y_test_batch, y_test)

                if metadata_batch is not None:
                    metadata_list.append(metadata_batch)

            y_pred = self.pred_transform_fn(y_pred)
            y_test = self.pred_transform_fn(y_test)

            metadata = dict_util.combine_prediction_metadata_batches(metadata_list) if len(metadata_list) > 0 else None

            if self.ddp_training_mode:
                y_pred = self.ddp_handler.mp_sync(y_pred).cpu()
                y_test = self.ddp_handler.mp_sync(y_test).cpu()
                metadata = self.ddp_handler.mp_sync_dict_of_lists(metadata) if metadata is not None else None

        self.model.train()

        return y_pred, y_test, metadata

    def insert_metric_result_into_history(self, metric_name, metric_result):
        """Insert a metric result into the train history

        This is the main and preferred API function for metric insertion as part of the train loop.

        Args:
            metric_name (str): name of the metric to be inserted
            metric_result (float or dict): new result for the corresponding metric
        """
        self.train_history.insert_single_result_into_history(metric_name, metric_result)

    def get_schedulers(self):
        """Get the registered schedulers

        Schedulers in TrainLoop training are implemented as callbacks under the hood.

        Returns:
            list: list of scheduler (callbacks)
        """
        return [cb for cb in self.callbacks if isinstance(cb, AbstractScheduler)]

    def _train_dp(self, num_epochs, callbacks=None, grad_accumulation=1, dp_model_args=None):
        """Train the model on multi-GPU with DataParallel auto wrapping

        Args:
            num_epochs (int): how many epochs the network will be trained
            callbacks (list or None): callbacks that are executed during the training run
            grad_accumulation (int): number of batches the gradients are accumulated before updating weights
            dp_model_args (dict or None): parameters for :class:`aitoolbox.torchtrain.parallel.TTDataParallel` /
                ``nn.DataParallel`` DP model wrap.

        Returns:
            TTDataParallel or nn.DataParallel: trained model
        """
        dp_model_args = dp_model_args if dp_model_args is not None else {}

        if not isinstance(self.model, TTDataParallel) and not isinstance(self.model, nn.DataParallel):
            if isinstance(self.model, TTModel):
                self.model = TTDataParallel(self.model, **dp_model_args)
            else:
                self.model = nn.DataParallel(self.model, **dp_model_args)

        return self._train(num_epochs, callbacks, grad_accumulation)

    def _train_ddp(self, num_epochs, callbacks=None, grad_accumulation=1,
                   ddp_model_args=None, in_process_data_load=None,
                   num_nodes=1, node_rank=0, num_gpus=torch.cuda.device_count()):
        """Train the model using the train loop in the Distributed Data Parallel setting

        During the training, multiple processes will be spawned, one for each of the available GPUs.

        Args:
            num_epochs (int): how many epochs the network will be trained
            callbacks (list or None): callbacks that are executed during the training run
            grad_accumulation (int): number of batches the gradients are accumulated before updating weights
            ddp_model_args (dict or None): parameters for DistributedDataParallel model
                Available parameters for DistributedDataParallel:
                    https://pytorch.org/docs/master/nn.html#torch.nn.parallel.DistributedDataParallel
            in_process_data_load (AbstractCallback or list or None):
                in-process data loading logic implemented as a torchtrain callback. The logic should be placed inside
                the on_multiprocess_start() callback function.
                When using this data loading option bare in mind that loaded dataset will be replicated in memory for
                every spawned training process. This can in turn in cause extensive overall memory consumption.
            num_nodes (int): number of nodes in the cluster
            node_rank (int): rank of the current node
            num_gpus (int): number of GPUs in the node
        """
        self.ddp_training_mode = True
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '8888'
        # Based on:
        # https://blog.exxactcorp.com/pytorch-1-5-1-bug-fix-release/
        # https://github.com/pytorch/pytorch/issues/37377
        os.environ['MKL_THREADING_LAYER'] = 'GNU'
        ddp_args = {
            'node_rank': node_rank,
            'num_gpus': num_gpus,
            'world_size': num_nodes * num_gpus,
            'ddp_model_args': ddp_model_args if ddp_model_args is not None else {}
        }

        from aitoolbox.torchtrain.callbacks.abstract import AbstractCallback
        if isinstance(in_process_data_load, AbstractCallback):
            in_process_data_load = [in_process_data_load]

        mp.spawn(self._spawn_fit,
                 args=(
                     ddp_args, num_epochs, callbacks, grad_accumulation, in_process_data_load
                 ),
                 nprocs=ddp_args['world_size'])

    def _spawn_fit(self, gpu, ddp_args, num_epochs, callbacks, grad_accumulation, in_process_data_load):
        """Helper function that prepares the TrainLoop state inside each of the spawned processes and initiates training

        Args:
            gpu (int): provided by the mp.spawn(); index of the GPU allocated to the current process
            ddp_args (dict): parameters dict needed for the distributed training setup
            num_epochs (int): how many epochs the network will be trained
            callbacks (list or None): callbacks that are executed during the training run
            grad_accumulation (int): number of batches the gradients are accumulated before updating weights
            in_process_data_load (list or None): in-process data loading logic implemented as a torchtrain callback.
                The logic should be placed inside the on_multiprocess_start() callback function.
                When using this data loading option bare in mind that loaded dataset will be replicated in memory for
                every spawned training process. This can in turn in cause extensive overall memory consumption.
        """
        rank = ddp_args['node_rank'] * ddp_args['num_gpus'] + gpu
        dist.init_process_group(backend='nccl', init_method='env://', world_size=ddp_args['world_size'], rank=rank)
        torch.manual_seed(0)
        torch.cuda.set_device(gpu)
        self.device = torch.device(f"cuda:{gpu}")
        self.callbacks_handler.mp_filter_callbacks()

        # Optionally load data in-process
        self.callbacks_handler.register_callbacks(in_process_data_load)
        self.callbacks_handler.execute_multiprocess_start()
        # Add DistributedSampler to the data loaders
        self.ddp_handler = DDPHandler(self)
        self.ddp_handler.add_distributed_samplers(ddp_args['world_size'], rank)

        # Move to the GPU belonging to the process
        self.model = self.model.to(self.device)
        if self.criterion is not None:
            self.criterion = self.criterion.to(self.device)

        # Optionally initialize AMP scaler inside each of the processes
        if self.use_amp:
            self.amp_scaler = amp.GradScaler(**self.amp_scaler_init)

        # Wrap models into DDP module
        if isinstance(self.model, TTModel):
            self.model = TTDistributedDataParallel(self.model, device_ids=[gpu], **ddp_args['ddp_model_args'])
        else:
            self.model = DistributedDataParallel(self.model, device_ids=[gpu], **ddp_args['ddp_model_args'])

        self._train(num_epochs, callbacks, grad_accumulation)

    def _train_deepspeed(self, deepspeed_args, num_epochs, callbacks=None,
                         **ds_model_args):
        """Train the model using Microsoft DeepSpeed package

        Before starting the training the DeepSpeed library needs to be installed on the machine. Find the installation
        instructions on this page: https://www.deepspeed.ai/getting-started/#installation.

        If you want to manually install the DeepSpeed package execute the ``install.sh`` script:
        https://github.com/microsoft/DeepSpeed/blob/master/install.sh

        Args:
            deepspeed_args (argparse.Namespace): argparser results structured as per DeepSpeed requirements.
                A dictionary containing local_rank and deepspeed_config file location.
            num_epochs (int): how many epochs the network will be trained
            callbacks (list): callbacks that are executed during the training run
            **ds_model_args: additional parameters for the underlying ``deepspeed.DeepSpeedLight`` class

                Possible arguments: https://deepspeed.readthedocs.io/en/latest/initialize.html

        Returns:
            deepspeed.DeepSpeedLight: DeepSpeed model engine
        """
        if not DEEPSPEED_AVAILABLE:
            raise ValueError('Trying to use Microsoft DeepSpeed. However, DeepSpeed is not installed.')
        if self.use_amp:
            raise ValueError('Base Nvidia APEX AMP enabled. To use DeepSpeed first disable base AMP and specify '
                             'the AMP as part of DeepSpeed config.')

        self.use_deepspeed = True

        self.model = TTDeepSpeedLight(
            args=deepspeed_args,
            model=self.model, model_parameters=self.model.parameters(),
            training_data=self.train_loader.dataset,
            **ds_model_args
        )
        self.optimizer = self.model.optimizer
        self.train_loader = self.model.training_dataloader

        return self._train(num_epochs, callbacks)

    def __call__(self, num_epochs, callbacks=None, grad_accumulation=1, **kwargs):
        """Train the model using the train loop

        This is a convenience function which calls the main TrainLoop model training method fit().

        Args:
            num_epochs (int): how many epochs the network will be trained
            callbacks (list): callbacks that are executed during the training run
            grad_accumulation (int): number of batches the gradients are accumulated before updating weights
            **kwargs: additional parameters for ``_train_dp()``, ``_train_ddp()`` and ``_train_deepspeed()`` methods.

        Returns:
            TTModel or torch.nn.modules.Module or TTDataParallel: trained model
        """
        return self.fit(num_epochs, callbacks, grad_accumulation, **kwargs)


class TrainLoopCheckpoint(TrainLoop):
    def __init__(self, model,
                 train_loader, validation_loader, test_loader,
                 optimizer, criterion,
                 project_name, experiment_name, local_model_result_folder_path,
                 hyperparams,
                 cloud_save_mode='s3', bucket_name='model-result', cloud_dir_prefix='', source_dirs=(),
                 rm_subopt_local_models=False, num_best_checkpoints_kept=2,
                 collate_batch_pred_fn=append_predictions, pred_transform_fn=torch_cat_transf,
                 end_auto_eval=True, lazy_experiment_save=True,
                 gpu_mode='single', cuda_device_idx=None, use_amp=False):
        """TrainLoop with the automatic model check-pointing at the end of each epoch

        Args:
            model (TTModel or ModelWrap or TTDataParallel): neural network model
            train_loader (torch.utils.data.DataLoader): data loader for train data set
            validation_loader (torch.utils.data.DataLoader or None): data loader for validation data set
            test_loader (torch.utils.data.DataLoader or None): data loader for test data set
            optimizer (torch.optim.optimizer.Optimizer or MultiOptimizer): optimizer algorithm.
            criterion (torch.nn.modules.loss._Loss or MultiLoss or None): criterion during the training procedure
            project_name (str): root name of the project
            experiment_name (str): name of the particular experiment
            local_model_result_folder_path (str): root local path where project folder will be created
            hyperparams (dict): used hyper-parameters. When running the TrainLoop from jupyter notebook in order to
                ensure the python experiment file copying to the experiment folder, the user needs to manually
                specify the python file path as the value for the `experiment_file_path` key. If running the training
                directly from the terminal the path deduction is done automatically.
            cloud_save_mode (str or None): Storage destination selector.
                For AWS S3: 's3' / 'aws_s3' / 'aws'
                For Google Cloud Storage: 'gcs' / 'google_storage' / 'google storage'
                Everything else results just in local storage to disk
            bucket_name (str): name of the bucket in the cloud storage
            cloud_dir_prefix (str): path to the folder inside the bucket where the experiments are going to be saved
            source_dirs (list or tuple): paths to the local folders with the source code files used in experiment
            rm_subopt_local_models (bool or str): if True, the deciding metric is set to 'loss'. Give string metric name
                to set it as a deciding metric for suboptimal model removal. If metric name consists of substring 'loss'
                the metric minimization is done otherwise metric maximization is done
            num_best_checkpoints_kept (int): number of best performing models which are kept when removing suboptimal
                model checkpoints
            collate_batch_pred_fn (callable): collate function transforming batch predictions as they come out from the
                model
            pred_transform_fn (callable): function transforming all the produced predictions after all the batches have
                been run through the model
            end_auto_eval (bool or int): used to optionally disable otherwise automatic end of epoch/training val/test
                loss calculations. This is useful when conducting very costly experiments to save on compute time.
                Specify either True/False boolean to always run or never run after each epoch or specify an int to
                execute only every specified number of epochs.
            lazy_experiment_save (bool): when in lazy mode experiment tracking components will create the experiment
                folder only after some training results are available (possibly at the end of the first epoch) instead
                of at the beginning of training.
            gpu_mode (str): GPU training mode selection. TrainLoop supports different GPU training modes by
                specifying one of the following:

                * ``'single'``: single GPU training
                * ``'dp'``: multi-GPU training via DataParallel
                * ``'ddp'``: multi-GPU training via DistributedDataParallel
                * ``'deepspeed'``: training via the Microsoft DeepSpeed

            cuda_device_idx (int or None): CUDA device index used when training on multiple GPUs
            use_amp (bool or dict): use 16-bit Automatic Mixed Precision (AMP)

                To switch to AMP mode either:

                * set this parameter to ``True`` to use default AMP ``torch.cuda.amp.GradScaler`` initialization params
                * provide custom AMP ``torch.cuda.amp.GradScaler`` initialization parameters as a dict as this parameter
        """
        TrainLoop.__init__(self, model, train_loader, validation_loader, test_loader, optimizer, criterion,
                           collate_batch_pred_fn, pred_transform_fn,
                           end_auto_eval, lazy_experiment_save,
                           gpu_mode, cuda_device_idx, use_amp)
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.local_model_result_folder_path = os.path.expanduser(local_model_result_folder_path)
        self.hyperparams = hyperparams
        self.cloud_save_mode = cloud_save_mode
        self.bucket_name = bucket_name
        self.cloud_dir_prefix = cloud_dir_prefix
        self.rm_subopt_local_models = rm_subopt_local_models

        if 'experiment_file_path' not in self.hyperparams:
            self.hyperparams['experiment_file_path'] = inspect.getframeinfo(inspect.currentframe().f_back).filename
        if 'source_dirs_paths' not in self.hyperparams:
            self.hyperparams['source_dirs_paths'] = source_dirs

        self.callbacks_handler.register_callbacks([
            ModelCheckpoint(self.project_name, self.experiment_name, self.local_model_result_folder_path,
                            self.hyperparams,
                            cloud_save_mode=self.cloud_save_mode,
                            bucket_name=bucket_name, cloud_dir_prefix=cloud_dir_prefix,
                            rm_subopt_local_models=self.rm_subopt_local_models,
                            num_best_checkpoints_kept=num_best_checkpoints_kept)
        ], cache_callbacks=True)


class TrainLoopEndSave(TrainLoop):
    def __init__(self, model,
                 train_loader, validation_loader, test_loader,
                 optimizer, criterion,
                 project_name, experiment_name, local_model_result_folder_path,
                 hyperparams, val_result_package=None, test_result_package=None,
                 cloud_save_mode='s3', bucket_name='model-result', cloud_dir_prefix='', source_dirs=(),
                 collate_batch_pred_fn=append_predictions, pred_transform_fn=torch_cat_transf,
                 end_auto_eval=True, lazy_experiment_save=True,
                 gpu_mode='single', cuda_device_idx=None, use_amp=False):
        """TrainLoop with the model performance evaluation and final model saving at the end of the training process

        Args:
            model (TTModel or ModelWrap or TTDataParallel): neural network model
            train_loader (torch.utils.data.DataLoader): data loader for train data set
            validation_loader (torch.utils.data.DataLoader or None): data loader for validation data set
            test_loader (torch.utils.data.DataLoader or None): data loader for test data set
            optimizer (torch.optim.optimizer.Optimizer or MultiOptimizer): optimizer algorithm.
            criterion (torch.nn.modules.loss._Loss or MultiLoss or None): criterion during the training procedure
            project_name (str): root name of the project
            experiment_name (str): name of the particular experiment
            local_model_result_folder_path (str): root local path where project folder will be created
            hyperparams (dict): used hyper-parameters. When running the TrainLoop from jupyter notebook in order to
                ensure the python experiment file copying to the experiment folder, the user needs to manually
                specify the python file path as the value for the `experiment_file_path` key. If running the training
                directly from the terminal the path deduction is done automatically.
            val_result_package (AbstractResultPackage or None): result package evaluated on validation data at  the end
                of the training
            test_result_package (AbstractResultPackage or None): result package evaluated on test data at the end
                of the training
            cloud_save_mode (str or None): Storage destination selector.
                For AWS S3: 's3' / 'aws_s3' / 'aws'
                For Google Cloud Storage: 'gcs' / 'google_storage' / 'google storage'
                Everything else results just in local storage to disk
            bucket_name (str): name of the bucket in the cloud storage
            cloud_dir_prefix (str): path to the folder inside the bucket where the experiments are going to be saved
            source_dirs (list or tuple): paths to the local folders with the source code files used in experiment
            collate_batch_pred_fn (callable): collate function transforming batch predictions as they come out from the
                model
            pred_transform_fn (callable): function transforming all the produced predictions after all the batches have
                been run through the model
            end_auto_eval (bool or int): used to optionally disable otherwise automatic end of epoch/training val/test
                loss calculations. This is useful when conducting very costly experiments to save on compute time.
                Specify either True/False boolean to always run or never run after each epoch or specify an int to
                execute only every specified number of epochs.
            lazy_experiment_save (bool): when in lazy mode experiment tracking components will create the experiment
                folder only after some training results are available (possibly at the end of the first epoch) instead
                of at the beginning of training.
            gpu_mode (str): GPU training mode selection. TrainLoop supports different GPU training modes by
                specifying one of the following:

                * ``'single'``: single GPU training
                * ``'dp'``: multi-GPU training via DataParallel
                * ``'ddp'``: multi-GPU training via DistributedDataParallel
                * ``'deepspeed'``: training via the Microsoft DeepSpeed

            cuda_device_idx (int or None): CUDA device index used when training on multiple GPUs
            use_amp (bool or dict): use 16-bit Automatic Mixed Precision (AMP)

                To switch to AMP mode either:

                * set this parameter to ``True`` to use default AMP ``torch.cuda.amp.GradScaler`` initialization params
                * provide custom AMP ``torch.cuda.amp.GradScaler`` initialization parameters as a dict as this parameter
        """
        TrainLoop.__init__(self, model, train_loader, validation_loader, test_loader, optimizer, criterion,
                           collate_batch_pred_fn, pred_transform_fn,
                           end_auto_eval, lazy_experiment_save,
                           gpu_mode, cuda_device_idx, use_amp)
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.local_model_result_folder_path = os.path.expanduser(local_model_result_folder_path)
        self.hyperparams = hyperparams
        self.val_result_package = val_result_package
        self.test_result_package = test_result_package
        self.cloud_save_mode = cloud_save_mode
        self.bucket_name = bucket_name
        self.cloud_dir_prefix = cloud_dir_prefix

        if 'experiment_file_path' not in self.hyperparams:
            self.hyperparams['experiment_file_path'] = inspect.getframeinfo(inspect.currentframe().f_back).filename
        if 'source_dirs_paths' not in self.hyperparams:
            self.hyperparams['source_dirs_paths'] = source_dirs
        self.check_if_result_packages_possible()

        self.callbacks_handler.register_callbacks([
            ModelTrainEndSave(self.project_name, self.experiment_name, self.local_model_result_folder_path,
                              self.hyperparams, self.val_result_package, self.test_result_package,
                              cloud_save_mode=self.cloud_save_mode,
                              bucket_name=bucket_name, cloud_dir_prefix=cloud_dir_prefix)
        ], cache_callbacks=True)

    def check_if_result_packages_possible(self):
        if self.val_result_package is not None and self.validation_loader is None:
            raise ValueError('Given the val_result_package but not supplied the validation_loader. '
                             'If you want to calculate the val_result_package the validation_loader has to be provided.')

        if self.test_result_package is not None and self.test_loader is None:
            raise ValueError('Given the test_result_package but not supplied the test_loader. '
                             'If you want to calculate the test_result_package the test_loader has to be provided.')

        if self.val_result_package is None and self.test_result_package is None:
            raise ValueError('Both val_result_package and test_result_package are None. '
                             'At least one of these should be not None but actual result package.')

        if self.val_result_package is not None and not isinstance(self.val_result_package, AbstractResultPackage):
            raise TypeError(f'val_result_package {self.val_result_package} is not inherited from AbstractResultPackage')

        if self.test_result_package is not None and not isinstance(self.test_result_package, AbstractResultPackage):
            raise TypeError(f'test_result_package {self.test_result_package} is not inherited from AbstractResultPackage')


class TrainLoopCheckpointEndSave(TrainLoopEndSave):
    def __init__(self, model,
                 train_loader, validation_loader, test_loader,
                 optimizer, criterion,
                 project_name, experiment_name, local_model_result_folder_path,
                 hyperparams, val_result_package=None, test_result_package=None,
                 cloud_save_mode='s3', bucket_name='model-result', cloud_dir_prefix='', source_dirs=(),
                 rm_subopt_local_models=False, num_best_checkpoints_kept=2,
                 collate_batch_pred_fn=append_predictions, pred_transform_fn=torch_cat_transf,
                 end_auto_eval=True, lazy_experiment_save=True,
                 gpu_mode='single', cuda_device_idx=None, use_amp=False):
        """TrainLoop both saving model check-pointing at the end of each epoch and model performance reporting
            and model saving at the end of the training process

        Args:
            model (TTModel or ModelWrap or TTDataParallel): neural network model
            train_loader (torch.utils.data.DataLoader): data loader for train data set
            validation_loader (torch.utils.data.DataLoader or None): data loader for validation data set
            test_loader (torch.utils.data.DataLoader or None): data loader for test data set
            optimizer (torch.optim.optimizer.Optimizer or MultiOptimizer): optimizer algorithm.
            criterion (torch.nn.modules.loss._Loss or MultiLoss or None): criterion during the training procedure
            project_name (str): root name of the project
            experiment_name (str): name of the particular experiment
            local_model_result_folder_path (str): root local path where project folder will be created
            hyperparams (dict): used hyper-parameters. When running the TrainLoop from jupyter notebook in order to
                ensure the python experiment file copying to the experiment folder, the user needs to manually
                specify the python file path as the value for the `experiment_file_path` key. If running the training
                directly from the terminal the path deduction is done automatically.
            val_result_package (AbstractResultPackage or None): result package evaluated on validation data at the end
                of the training
            test_result_package (AbstractResultPackage or None): result package evaluated on test data at the end
                of the training
            cloud_save_mode (str or None): Storage destination selector.
                For AWS S3: 's3' / 'aws_s3' / 'aws'
                For Google Cloud Storage: 'gcs' / 'google_storage' / 'google storage'
                Everything else results just in local storage to disk
            bucket_name (str): name of the bucket in the cloud storage
            cloud_dir_prefix (str): path to the folder inside the bucket where the experiments are going to be saved
            source_dirs (list or tuple): paths to the local folders with the source code files used in experiment
            rm_subopt_local_models (bool or str): if True, the deciding metric is set to 'loss'. Give string metric name
                to set it as a deciding metric for suboptimal model removal. If metric name consists of substring 'loss'
                the metric minimization is done otherwise metric maximization is done
            num_best_checkpoints_kept (int): number of best performing models which are kept when removing suboptimal
                model checkpoints
            collate_batch_pred_fn (callable): collate function transforming batch predictions as they come out from the
                model
            pred_transform_fn (callable): function transforming all the produced predictions after all the batches have
                been run through the model
            end_auto_eval (bool or int): used to optionally disable otherwise automatic end of epoch/training val/test
                loss calculations. This is useful when conducting very costly experiments to save on compute time.
                Specify either True/False boolean to always run or never run after each epoch or specify an int to
                execute only every specified number of epochs.
            lazy_experiment_save (bool): when in lazy mode experiment tracking components will create the experiment
                folder only after some training results are available (possibly at the end of the first epoch) instead
                of at the beginning of training.
            gpu_mode (str): GPU training mode selection. TrainLoop supports different GPU training modes by
                specifying one of the following:

                * ``'single'``: single GPU training
                * ``'dp'``: multi-GPU training via DataParallel
                * ``'ddp'``: multi-GPU training via DistributedDataParallel
                * ``'deepspeed'``: training via the Microsoft DeepSpeed

            cuda_device_idx (int or None): CUDA device index used when training on multiple GPUs
            use_amp (bool or dict): use 16-bit Automatic Mixed Precision (AMP)

                To switch to AMP mode either:

                * set this parameter to ``True`` to use default AMP ``torch.cuda.amp.GradScaler`` initialization params
                * provide custom AMP ``torch.cuda.amp.GradScaler`` initialization parameters as a dict as this parameter
        """
        if 'experiment_file_path' not in hyperparams:
            hyperparams['experiment_file_path'] = inspect.getframeinfo(inspect.currentframe().f_back).filename
        if 'source_dirs_paths' not in hyperparams:
            hyperparams['source_dirs_paths'] = source_dirs

        TrainLoopEndSave.__init__(self, model, train_loader, validation_loader, test_loader,
                                  optimizer, criterion,
                                  project_name, experiment_name, os.path.expanduser(local_model_result_folder_path),
                                  hyperparams, val_result_package, test_result_package,
                                  cloud_save_mode, bucket_name, cloud_dir_prefix, source_dirs,
                                  collate_batch_pred_fn, pred_transform_fn,
                                  end_auto_eval, lazy_experiment_save,
                                  gpu_mode, cuda_device_idx, use_amp)
        self.rm_subopt_local_models = rm_subopt_local_models

        self.callbacks_handler.register_callbacks([
            ModelCheckpoint(self.project_name, self.experiment_name, self.local_model_result_folder_path,
                            self.hyperparams,
                            cloud_save_mode=self.cloud_save_mode,
                            bucket_name=bucket_name, cloud_dir_prefix=cloud_dir_prefix,
                            rm_subopt_local_models=self.rm_subopt_local_models,
                            num_best_checkpoints_kept=num_best_checkpoints_kept)
        ], cache_callbacks=True)
