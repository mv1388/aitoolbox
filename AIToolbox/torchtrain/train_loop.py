from tqdm import tqdm
import os
import time
import datetime
import inspect
import numpy as np
import torch
from torch.nn.modules import Module

from AIToolbox.utils import dict_util
from AIToolbox.torchtrain.model import TTFullModel, ModelWrap
from AIToolbox.torchtrain.multi_loss import MultiOptimizer
from AIToolbox.torchtrain.data.batch_model_feed_defs import AbstractModelFeedDefinition
from AIToolbox.torchtrain.tl_components.callback_handler import CallbacksHandler
from AIToolbox.torchtrain.callbacks.model_save_callbacks import ModelCheckpoint, ModelTrainEndSave
from AIToolbox.experiment.training_history import TrainingHistory
from AIToolbox.torchtrain.tl_components.model_prediction_store import ModelPredictionStore
from AIToolbox.torchtrain.tl_components.message_passing import MessageService
from AIToolbox.experiment.result_package.abstract_result_packages import AbstractResultPackage


class TrainLoop:
    def __init__(self, model,
                 train_loader, validation_loader, test_loader,
                 optimizer, criterion):
        """

        Args:
            model (AIToolbox.torchtrain.model.TTFullModel or AIToolbox.torchtrain.model.ModelWrap): neural
                network model
            train_loader (torch.utils.data.DataLoader): data loader for train data set
            validation_loader (torch.utils.data.DataLoader): data loader for validation data set
            test_loader (torch.utils.data.DataLoader): data loader for test data set
            optimizer (torch.optim.optimizer.Optimizer or MultiOptimizer): optimizer algorithm.
            criterion (torch.nn.modules.loss._Loss): criterion criterion during the training procedure.
        """
        if isinstance(model, TTFullModel):
            self.model = model
            self.batch_model_feed_def = None
        elif type(model) == ModelWrap:
            self.model = model.model
            self.batch_model_feed_def = model.batch_model_feed_def
        else:
            raise TypeError(f"Provided model is not either inherited from TTFullModel or ModelWrap. "
                            f"Provided type is: {type(model)}.")

        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.criterion = criterion

        USE_CUDA = torch.cuda.is_available()
        self.device = torch.device("cuda" if USE_CUDA else "cpu")

        self.experiment_timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')
        self.loss_batch_accum = []
        self.epoch = 0

        self.train_history = TrainingHistory(has_validation=self.validation_loader is not None)
        self.prediction_store = ModelPredictionStore(auto_purge=True)
        self.message_service = MessageService()

        self.callbacks_handler = CallbacksHandler(self)
        self.callbacks = []
        self.early_stop = False

        if not isinstance(self.model, TTFullModel) and not isinstance(self.model, Module):
            raise TypeError('Provided model is not inherited from TTFullModel or base PyTorch Module')
        if not isinstance(self.model, TTFullModel) and \
                isinstance(self.model, Module) and not isinstance(self.batch_model_feed_def, AbstractModelFeedDefinition):
            raise TypeError('Provided the base PyTorch model but did not give the batch_model_feed_def')

    def __call__(self, num_epoch, callbacks=None, grad_clip=None):
        """Train the model using the train loop

        Args:
            num_epoch (int): how many epochs the network will be trained
            callbacks (list): callbacks that are executed during the training run
            grad_clip (int or float): optional gradient clipping

        Returns:
            torch.nn.modules.Module: trained model
        """
        return self.do_train(num_epoch, callbacks, grad_clip)

    def do_train(self, num_epoch, callbacks=None, grad_clip=None):
        """Train the model using the train loop

        Args:
            num_epoch (int): how many epochs the network will be trained
            callbacks (list): callbacks that are executed during the training run
            grad_clip (int or float): optional gradient clipping

        Returns:
            torch.nn.modules.Module: trained model
        """
        self.callbacks_handler.register_callbacks(callbacks)

        self.model = self.model.to(self.device)
        self.model.train()

        self.callbacks_handler.execute_train_begin()

        for self.epoch in range(self.epoch, num_epoch):
            print('\n\n========================================================================')
            print('========================================================================')
            # print(self.train_history)
            print(f'Epoch: {self.epoch + 1}')
            self.callbacks_handler.execute_epoch_begin()

            for batch_data in tqdm(self.train_loader):
                self.callbacks_handler.execute_batch_begin()

                if isinstance(self.model, TTFullModel):
                    loss_batch = self.model.get_loss(batch_data, self.criterion, self.device)
                else:
                    loss_batch = self.batch_model_feed_def.get_loss(self.model, batch_data, self.criterion, self.device)

                self.loss_batch_accum.append(loss_batch.item())

                self.optimizer.zero_grad()
                loss_batch.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                self.optimizer.step()

                self.callbacks_handler.execute_batch_end()

            # Automatic end of epoch code - reports the train and if available validation loss and executes callbacks
            self.auto_execute_end_of_epoch()
            self.callbacks_handler.execute_epoch_end()

            self.message_service.end_of_epoch_trigger()

            # self.early_stop is changed from the early stopper callback
            if self.early_stop:
                break

        self.auto_execute_end_of_training()
        self.callbacks_handler.execute_train_end()

        return self.model

    def auto_execute_end_of_epoch(self):
        """Basic performance evaluation executed by default at the end of each epoch

        Mainly evaluation of the loss functions which are always present as part of the training loop.

        Returns:
            None
        """
        if type(self.optimizer) == MultiOptimizer:
            train_loss_batch_accum_avg = np.mean(self.loss_batch_accum, axis=0).tolist()
        else:
            train_loss_batch_accum_avg = np.mean(self.loss_batch_accum).item()

        print(f'AVG BATCH ACCUMULATED TRAIN LOSS: {train_loss_batch_accum_avg}')
        self.insert_metric_result_into_history('accumulated_loss', train_loss_batch_accum_avg)
        self.loss_batch_accum = []

        train_loss = self.evaluate_loss_on_train_set()
        print(f'TRAIN LOSS: {train_loss}')
        self.insert_metric_result_into_history('loss', train_loss)

        if self.validation_loader is not None:
            val_loss = self.evaluate_loss_on_validation_set()
            print(f'VAL LOSS: {val_loss}')
            self.insert_metric_result_into_history('val_loss', val_loss)

    def auto_execute_end_of_training(self):
        """Basic performance evaluation executed by default at the end of the training process

        Returns:
            None
        """
        if self.test_loader is not None:
            test_loss = self.evaluate_loss_on_test_set()
            print(f'TEST LOSS: {test_loss}')
            # To keep TrainingHistory from complaining due to the non-matching metric result lengths the checking
            # has been turned off
            self.insert_metric_result_into_history('train_end_test_loss', test_loss)

    def evaluate_loss_on_train_set(self):
        """Run train dataset through the network without updating the weights and return the loss

        Returns:
            float: loss
        """
        return self.evaluate_model_loss(self.train_loader)

    def evaluate_loss_on_validation_set(self):
        """Run validation dataset through the network without updating the weights and return the loss

        Returns:
            float: loss
        """
        return self.evaluate_model_loss(self.validation_loader)

    def evaluate_loss_on_test_set(self):
        """Run test dataset through the network without updating the weights and return the loss

        Returns:
            float: loss
        """
        return self.evaluate_model_loss(self.test_loader)

    def evaluate_model_loss(self, data_loader):
        """Run given dataset through the network without updating the weights and return the loss

        Args:
            data_loader (torch.utils.data.DataLoader):

        Returns:
            float: loss
        """
        self.model.eval()
        loss_avg = []

        with torch.no_grad():
            for batch_data in tqdm(data_loader):
                if isinstance(self.model, TTFullModel):
                    loss_batch = self.model.get_loss_eval(batch_data, self.criterion, self.device)
                else:
                    loss_batch = self.batch_model_feed_def.get_loss_eval(self.model, batch_data, self.criterion,
                                                                         self.device)

                loss_avg.append(loss_batch.item())

        self.model.train()

        return np.mean(loss_avg, axis=0)

    def predict_on_train_set(self, force_prediction=False):
        """Run train dataset through the network and return true target values, target predictions and metadata

        Args:
            force_prediction (bool):

        Returns:
            (torch.Tensor, torch.Tensor, dict): y_true, y_pred, metadata
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
            force_prediction (bool):

        Returns:
            (torch.Tensor, torch.Tensor, dict): y_true, y_pred, metadata
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
            force_prediction (bool):

        Returns:
            (torch.Tensor, torch.Tensor, dict): y_true, y_pred, metadata
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
            data_loader (torch.utils.data.DataLoader):

        Returns:
            (torch.Tensor, torch.Tensor, dict): y_true, y_pred, metadata
        """
        y_test, y_pred, metadata_list = [], [], []

        self.model.eval()

        with torch.no_grad():
            for batch_data in tqdm(data_loader):
                if isinstance(self.model, TTFullModel):
                    y_test_batch, y_pred_batch, metadata_batch = self.model.get_predictions(batch_data, self.device)
                else:
                    y_test_batch, y_pred_batch, metadata_batch = \
                        self.batch_model_feed_def.get_predictions(self.model, batch_data, self.device)

                # TODO: check if it is the best idea to append predictions to the list and not to some torch tensor
                # TODO: also if append is the best option and not the concat
                if type(y_test_batch) is list:
                    y_test += y_test_batch
                else:
                    y_test.append(y_test_batch)

                if type(y_pred_batch) is list:
                    y_pred += y_pred_batch
                else:
                    y_pred.append(y_pred_batch)

                if metadata_batch is not None:
                    metadata_list.append(metadata_batch)

            if type(y_test_batch) is not list:
                y_test = torch.cat(y_test)

            if type(y_pred_batch) is not list:
                y_pred = torch.cat(y_pred)

            metadata = dict_util.combine_prediction_metadata_batches(metadata_list) if len(metadata_list) > 0 else None

        self.model.train()

        return y_test, y_pred, metadata

    def insert_metric_result_into_history(self, metric_name, metric_result):
        """Insert a metric result into the train history

        This is the main and preferred API function for metric insertion as part of the train loop.

        Args:
            metric_name (str): name of the metric to be inserted
            metric_result (float or dict): new result for the corresponding metric
        """
        self.train_history.insert_single_result_into_history(metric_name, metric_result)


class TrainLoopModelCheckpoint(TrainLoop):
    def __init__(self, model,
                 train_loader, validation_loader, test_loader,
                 optimizer, criterion,
                 project_name, experiment_name, local_model_result_folder_path,
                 hyperparams,
                 cloud_save_mode='s3', bucket_name='model-result', cloud_dir_prefix='',
                 rm_subopt_local_models=False, num_best_checkpoints_kept=2):
        """TrainLoop with the automatic model check-pointing at the end of each epoch

        Args:
            model (AIToolbox.torchtrain.model.TTFullModel or AIToolbox.torchtrain.model.ModelWrap): neural
                network model
            train_loader (torch.utils.data.DataLoader):
            validation_loader (torch.utils.data.DataLoader):
            test_loader (torch.utils.data.DataLoader):
            optimizer (torch.optim.optimizer.Optimizer or MultiOptimizer): optimizer algorithm.
            criterion (torch.nn.modules.loss._Loss): criterion criterion during the training procedure.
            project_name (str): root name of the project
            experiment_name (str): name of the particular experiment
            local_model_result_folder_path (str): root local path where project folder will be created
            hyperparams (dict): used hyper-parameters
            cloud_save_mode (str or None): Storage destination selector.
                For AWS S3: 's3' / 'aws_s3' / 'aws'
                For Google Cloud Storage: 'gcs' / 'google_storage' / 'google storage'
                Everything else results just in local storage to disk
            bucket_name (str): name of the bucket in the cloud storage
            cloud_dir_prefix (str): path to the folder inside the bucket where the experiments are going to be saved
            rm_subopt_local_models (bool or str): if True, the deciding metric is set to 'loss'. Give string metric name
                to set it as a deciding metric for suboptimal model removal. If metric name consists of substring 'loss'
                the metric minimization is done otherwise metric maximization is done
            num_best_checkpoints_kept (int): number of best performing models which are kept when removing suboptimal
                model checkpoints
        """
        TrainLoop.__init__(self, model, train_loader, validation_loader, test_loader, optimizer, criterion)
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.local_model_result_folder_path = os.path.expanduser(local_model_result_folder_path)
        self.hyperparams = hyperparams
        self.cloud_save_mode = cloud_save_mode
        self.cloud_dir_prefix = cloud_dir_prefix
        self.rm_subopt_local_models = rm_subopt_local_models

        self.hyperparams['experiment_file_path'] = inspect.getframeinfo(inspect.currentframe().f_back).filename

        self.callbacks_handler.register_callbacks([
            ModelCheckpoint(self.project_name, self.experiment_name, self.local_model_result_folder_path,
                            self.hyperparams,
                            cloud_save_mode=self.cloud_save_mode, bucket_name=bucket_name, cloud_dir_prefix=cloud_dir_prefix,
                            rm_subopt_local_models=self.rm_subopt_local_models,
                            num_best_checkpoints_kept=num_best_checkpoints_kept)
        ])


class TrainLoopModelEndSave(TrainLoop):
    def __init__(self, model,
                 train_loader, validation_loader, test_loader,
                 optimizer, criterion,
                 project_name, experiment_name, local_model_result_folder_path,
                 hyperparams, val_result_package=None, test_result_package=None,
                 cloud_save_mode='s3', bucket_name='model-result', cloud_dir_prefix=''):
        """TrainLoop with the model performance evaluation and final model saving at the end of the training process

        Args:
            model (AIToolbox.torchtrain.model.TTFullModel or AIToolbox.torchtrain.model.ModelWrap): neural
                network model
            train_loader (torch.utils.data.DataLoader):
            validation_loader (torch.utils.data.DataLoader or None):
            test_loader (torch.utils.data.DataLoader or None):
            optimizer (torch.optim.optimizer.Optimizer or MultiOptimizer): optimizer algorithm.
            criterion (torch.nn.modules.loss._Loss): criterion criterion during the training procedure.
            project_name (str): root name of the project
            experiment_name (str): name of the particular experiment
            local_model_result_folder_path (str): root local path where project folder will be created
            hyperparams (dict): used hyper-parameters
            val_result_package (AIToolbox.experiment.result_package.abstract_result_packages.AbstractResultPackage or None):
            test_result_package (AIToolbox.experiment.result_package.abstract_result_packages.AbstractResultPackage or None):
            cloud_save_mode (str or None): Storage destination selector.
                For AWS S3: 's3' / 'aws_s3' / 'aws'
                For Google Cloud Storage: 'gcs' / 'google_storage' / 'google storage'
                Everything else results just in local storage to disk
            bucket_name (str): name of the bucket in the cloud storage
            cloud_dir_prefix (str): path to the folder inside the bucket where the experiments are going to be saved
        """
        TrainLoop.__init__(self, model, train_loader, validation_loader, test_loader, optimizer, criterion)
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.local_model_result_folder_path = os.path.expanduser(local_model_result_folder_path)
        self.hyperparams = hyperparams
        self.val_result_package = val_result_package
        self.test_result_package = test_result_package
        self.cloud_save_mode = cloud_save_mode
        self.cloud_dir_prefix = cloud_dir_prefix

        self.hyperparams['experiment_file_path'] = inspect.getframeinfo(inspect.currentframe().f_back).filename
        self.check_if_result_packages_possible()

        self.callbacks_handler.register_callbacks([
            ModelTrainEndSave(self.project_name, self.experiment_name, self.local_model_result_folder_path,
                              self.hyperparams, self.val_result_package, self.test_result_package,
                              cloud_save_mode=self.cloud_save_mode, bucket_name=bucket_name, cloud_dir_prefix=cloud_dir_prefix)
        ])

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


class TrainLoopModelCheckpointEndSave(TrainLoopModelEndSave):
    def __init__(self, model,
                 train_loader, validation_loader, test_loader,
                 optimizer, criterion,
                 project_name, experiment_name, local_model_result_folder_path,
                 hyperparams, val_result_package=None, test_result_package=None,
                 cloud_save_mode='s3', bucket_name='model-result', cloud_dir_prefix='',
                 rm_subopt_local_models=False, num_best_checkpoints_kept=2):
        """TrainLoop both saving model check-pointing at the end of each epoch and model performance reporting
            and model saving at the end of the training process

        Args:
            model (AIToolbox.torchtrain.model.TTFullModel or AIToolbox.torchtrain.model.ModelWrap): neural
                network model
            train_loader (torch.utils.data.DataLoader):
            validation_loader (torch.utils.data.DataLoader or None):
            test_loader (torch.utils.data.DataLoader or None):
            optimizer (torch.optim.optimizer.Optimizer or MultiOptimizer): optimizer algorithm.
            criterion (torch.nn.modules.loss._Loss): criterion criterion during the training procedure.
            project_name (str): root name of the project
            experiment_name (str): name of the particular experiment
            local_model_result_folder_path (str): root local path where project folder will be created
            hyperparams (dict): used hyper-parameters
            val_result_package (AIToolbox.experiment.result_package.abstract_result_packages.AbstractResultPackage or None):
            test_result_package (AIToolbox.experiment.result_package.abstract_result_packages.AbstractResultPackage or None):
            cloud_save_mode (str or None): Storage destination selector.
                For AWS S3: 's3' / 'aws_s3' / 'aws'
                For Google Cloud Storage: 'gcs' / 'google_storage' / 'google storage'
                Everything else results just in local storage to disk
            bucket_name (str): name of the bucket in the cloud storage
            cloud_dir_prefix (str): path to the folder inside the bucket where the experiments are going to be saved
            rm_subopt_local_models (bool or str): if True, the deciding metric is set to 'loss'. Give string metric name
                to set it as a deciding metric for suboptimal model removal. If metric name consists of substring 'loss'
                the metric minimization is done otherwise metric maximization is done
            num_best_checkpoints_kept (int): number of best performing models which are kept when removing suboptimal
                model checkpoints
        """
        TrainLoopModelEndSave.__init__(self, model, train_loader, validation_loader, test_loader,
                                       optimizer, criterion,
                                       project_name, experiment_name, os.path.expanduser(local_model_result_folder_path),
                                       hyperparams, val_result_package, test_result_package,
                                       cloud_save_mode, bucket_name, cloud_dir_prefix)
        self.rm_subopt_local_models = rm_subopt_local_models
        self.hyperparams['experiment_file_path'] = inspect.getframeinfo(inspect.currentframe().f_back).filename

        self.callbacks_handler.register_callbacks([
            ModelCheckpoint(self.project_name, self.experiment_name, self.local_model_result_folder_path,
                            self.hyperparams,
                            cloud_save_mode=self.cloud_save_mode, bucket_name=bucket_name, cloud_dir_prefix=cloud_dir_prefix,
                            rm_subopt_local_models=self.rm_subopt_local_models,
                            num_best_checkpoints_kept=num_best_checkpoints_kept)
        ])
