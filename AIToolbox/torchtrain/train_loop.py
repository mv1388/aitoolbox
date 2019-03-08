from tqdm import tqdm
import time
import datetime
import numpy as np
import torch

from AIToolbox.torchtrain.callbacks.callback_handler import CallbacksHandler
from AIToolbox.torchtrain.callbacks.callbacks import ModelCheckpointCallback, ModelTrainEndSaveCallback


class TrainLoop:
    def __init__(self, model,
                 train_loader, validation_loader,
                 batch_model_feed_def,
                 optimizer, criterion):
        """

        Args:
            model (torch.nn.modules.Module):
            train_loader (torch.utils.data.DataLoader):
            validation_loader (torch.utils.data.DataLoader):
            batch_model_feed_def (AIToolbox.torchtrain.batch_model_feed_defs.AbstractModelFeedDefinition):
            optimizer:
            criterion:
        """
        self.model = model
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.batch_model_feed_def = batch_model_feed_def
        self.optimizer = optimizer
        self.criterion = criterion

        USE_CUDA = torch.cuda.is_available()
        self.device = torch.device("cuda" if USE_CUDA else "cpu")

        self.experiment_timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')
        self.loss_batch_accum = []
        self.epoch = 0

        self.train_history = {'loss': [], 'accumulated_loss': [], 'val_loss': []} if self.validation_loader is not None \
            else {'loss': [], 'accumulated_loss': []}

        self.callbacks_handler = CallbacksHandler(self)
        self.callbacks = []
        self.early_stop = False

    def __call__(self, num_epoch, callbacks=None):
        """

        Args:
            num_epoch (int):
            callbacks (list):

        Returns:

        """
        return self.do_train(num_epoch, callbacks)

    def do_train(self, num_epoch, callbacks=None):
        """

        Args:
            num_epoch (int):
            callbacks (list):

        Returns:

        """
        self.callbacks_handler.register_callbacks(callbacks)

        self.model = self.model.to(self.device)
        self.model.train()

        self.callbacks_handler.execute_train_begin()

        for self.epoch in range(num_epoch):
            print('=================================================')
            # print(self.train_history)
            print(f'Epoch: {self.epoch + 1}')
            self.callbacks_handler.execute_epoch_begin()

            for batch_data in tqdm(self.train_loader):
                self.callbacks_handler.execute_batch_begin()

                loss_batch = self.batch_model_feed_def.get_loss(self.model, batch_data, self.criterion, self.device)
                self.loss_batch_accum.append(loss_batch.item())

                self.optimizer.zero_grad()
                loss_batch.backward()
                self.optimizer.step()

                self.callbacks_handler.execute_batch_end()

            # Automatic end of epoch code - reports the train and if available validation loss and executes callbacks
            self.auto_execute_end_of_epoch()
            self.callbacks_handler.execute_epoch_end()

            # self.early_stop is changed from the early stopper callback
            if self.early_stop:
                break

        self.callbacks_handler.execute_train_end()

        return self.model

    def auto_execute_end_of_epoch(self):
        train_loss_batch_accum_avg = np.mean(self.loss_batch_accum)
        print(f'AVG BATCH ACCUMULATED TRAIN LOSS: {train_loss_batch_accum_avg}')
        self.train_history['accumulated_loss'].append(train_loss_batch_accum_avg)
        self.loss_batch_accum = []

        train_loss = self.evaluate_loss_on_train_set()
        print(f'TRAIN LOSS: {train_loss}')
        # TODO: test this
        # self.train_history['loss'].append(train_loss)
        self.insert_metric_result_into_history('loss', train_loss)

        if self.validation_loader is not None:
            val_loss = self.evaluate_loss_on_validation_set()
            print(f'VAL LOSS: {val_loss}')
            # TODO: test this
            # self.train_history['val_loss'].append(val_loss)
            self.insert_metric_result_into_history('val_loss', val_loss)

    def evaluate_loss_on_train_set(self):
        """

        Returns:
            float:
        """
        return self.evaluate_model_loss(self.train_loader)

    def evaluate_loss_on_validation_set(self):
        """

        Returns:
            float:
        """
        return self.evaluate_model_loss(self.validation_loader)

    def evaluate_model_loss(self, data_loader):
        """

        Args:
            data_loader (torch.utils.data.DataLoader):

        Returns:
            float:
        """
        self.model.eval()
        loss_avg = []

        with torch.no_grad():
            for batch_data in tqdm(data_loader):
                loss_batch = self.batch_model_feed_def.get_loss(self.model, batch_data, self.criterion, self.device)

                loss_avg.append(loss_batch.item())

        self.model.train()

        return np.mean(loss_avg)

    def predict_on_train_set(self):
        """

        Returns:
            (torch.Tensor, torch.Tensor):
        """
        return self.predict_with_model(self.train_loader)

    def predict_on_validation_set(self):
        """

        Returns:
            (torch.Tensor, torch.Tensor):
        """
        return self.predict_with_model(self.validation_loader)

    def predict_with_model(self, data_loader):
        """

        Args:
            data_loader (torch.utils.data.DataLoader):

        Returns:
            (torch.Tensor, torch.Tensor):
        """
        y_test, y_pred = [], []

        self.model.eval()

        with torch.no_grad():
            for batch_data in tqdm(data_loader):
                y_test_batch, y_pred_batch = self.batch_model_feed_def.get_predictions(self.model, batch_data,
                                                                                       self.device)

                # TODO: check if it is the best idea to append predictions to the list and not to some torch tensor
                # TODO: also if append is the best option and not the concat
                y_test.append(y_test_batch)
                y_pred.append(y_pred_batch)

            y_test = torch.cat(y_test)
            y_pred = torch.cat(y_pred)

        self.model.train()

        return y_test, y_pred

    def insert_metric_result_into_history(self, metric_name, metric_result):
        """

        Args:
            metric_name (str):
            metric_result (float or dict):

        """
        if metric_name not in self.train_history:
            self.train_history[metric_name] = []
        self.train_history[metric_name].append(metric_result)


class TrainLoopModelCheckpoint(TrainLoop):
    def __init__(self, model,
                 train_loader, validation_loader,
                 batch_model_feed_def,
                 optimizer, criterion,
                 project_name, experiment_name, local_model_result_folder_path):
        """

        Args:
            model (torch.nn.modules.Module):
            train_loader (torch.utils.data.DataLoader):
            validation_loader (torch.utils.data.DataLoader):
            batch_model_feed_def (AIToolbox.torchtrain.batch_model_feed_defs.AbstractModelFeedDefinition):
            optimizer:
            criterion:
            project_name (str):
            experiment_name (str):
            local_model_result_folder_path (str):
        """
        TrainLoop.__init__(self, model, train_loader, validation_loader, batch_model_feed_def, optimizer, criterion)
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.local_model_result_folder_path = local_model_result_folder_path

        self.callbacks_handler.register_callbacks([
            ModelCheckpointCallback(self.project_name, self.experiment_name, self.local_model_result_folder_path)
        ])


class TrainLoopModelEndSave(TrainLoop):
    def __init__(self, model,
                 train_loader, validation_loader,
                 batch_model_feed_def,
                 optimizer, criterion,
                 project_name, experiment_name, local_model_result_folder_path,
                 args, result_package):
        """

        Args:
            model (torch.nn.modules.Module):
            train_loader (torch.utils.data.DataLoader):
            validation_loader (torch.utils.data.DataLoader):
            batch_model_feed_def (AIToolbox.torchtrain.batch_model_feed_defs.AbstractModelFeedDefinition):
            optimizer:
            criterion:
            project_name (str):
            experiment_name (str):
            local_model_result_folder_path (str):
            args (dict):
            result_package (AIToolbox.experiment_save.result_package.AbstractResultPackage):
        """
        TrainLoop.__init__(self, model, train_loader, validation_loader, batch_model_feed_def, optimizer, criterion)
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.local_model_result_folder_path = local_model_result_folder_path
        self.args = args
        self.result_package = result_package

        self.callbacks_handler.register_callbacks([
            ModelTrainEndSaveCallback(self.project_name, self.experiment_name, self.local_model_result_folder_path,
                                      self.args, self.result_package)
        ])


class TrainLoopModelCheckpointEndSave(TrainLoopModelEndSave):
    def __init__(self, model,
                 train_loader, validation_loader,
                 batch_model_feed_def,
                 optimizer, criterion,
                 project_name, experiment_name, local_model_result_folder_path,
                 args, result_package):
        """

        Args:
            model (torch.nn.modules.Module):
            train_loader (torch.utils.data.DataLoader):
            validation_loader (torch.utils.data.DataLoader):
            batch_model_feed_def (AIToolbox.torchtrain.batch_model_feed_defs.AbstractModelFeedDefinition):
            optimizer:
            criterion:
            project_name (str):
            experiment_name (str):
            local_model_result_folder_path (str):
            args (dict):
            result_package (AIToolbox.experiment_save.result_package.AbstractResultPackage):
        """
        TrainLoopModelEndSave.__init__(self, model, train_loader, validation_loader, batch_model_feed_def,
                                       optimizer, criterion,
                                       project_name, experiment_name, local_model_result_folder_path,
                                       args, result_package)

        self.callbacks_handler.register_callbacks([
            ModelCheckpointCallback(self.project_name, self.experiment_name, self.local_model_result_folder_path)
        ])
