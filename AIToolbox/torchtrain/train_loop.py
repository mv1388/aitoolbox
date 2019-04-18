from tqdm import tqdm
import time
import datetime
import numpy as np
import torch

from AIToolbox.torchtrain.callbacks.callback_handler import CallbacksHandler
from AIToolbox.torchtrain.callbacks.callbacks import ModelCheckpointCallback, ModelTrainEndSaveCallback

from AIToolbox.logger.logger import Logger
logger = Logger()


class TrainLoop:
    def __init__(self, model,
                 train_loader, validation_loader, test_loader,
                 batch_model_feed_def,
                 optimizer, criterion):
        """

        Args:
            model (torch.nn.modules.Module):
            train_loader (torch.utils.data.DataLoader):
            validation_loader (torch.utils.data.DataLoader):
            test_loader (torch.utils.data.DataLoader):
            batch_model_feed_def (AIToolbox.torchtrain.batch_model_feed_defs.AbstractModelFeedDefinition):
            optimizer:
            criterion:

        """
        self.model = model
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.test_loader = test_loader
        self.batch_model_feed_def = batch_model_feed_def
        self.optimizer = optimizer
        self.criterion = criterion

        self._USE_CUDA = torch.cuda.is_available()
        self.device = torch.device("cuda" if self._USE_CUDA else "cpu")

        self.experiment_timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')
        self.loss_batch_accum = []
        self.epoch = 0

        self.train_history = {'loss': [], 'accumulated_loss': [], 'val_loss': []} if self.validation_loader is not None \
            else {'loss': [], 'accumulated_loss': []}

        self.callbacks_handler = CallbacksHandler(self)
        self.callbacks = []
        self.early_stop = False

    def __call__(self, num_epoch, callbacks=None, grad_clip=None):
        """

        Args:
            num_epoch (int):
            callbacks (list):
            grad_clip (int or float):

        Returns:
            torch.nn.modules.Module:

        """
        return self.do_train(num_epoch, callbacks, grad_clip)

    def do_train(self, num_epoch, callbacks=None, grad_clip=None):
        """

        Args:
            num_epoch (int):
            callbacks (list):
            grad_clip (int or float):

        Returns:
            torch.nn.modules.Module:

        """
        logger.info(f'Train start time: {self.experiment_timestamp}')
        logger.info(f'Device: {"cuda" if self._USE_CUDA else "cpu"}', for_summary=False)
        logger.info(f'TrainLoop type: {type(self)}', for_summary=False)
        logger.info(f'Init Train History: {self.train_history}')

        self.callbacks_handler.register_callbacks(callbacks)
        logger.info(self.callbacks_handler)

        self.model = self.model.to(self.device)
        self.model.train()

        self.callbacks_handler.execute_train_begin()

        for self.epoch in range(num_epoch):
            print('\n\n========================================================================')
            print('========================================================================')
            logger.info(f'Epoch: {self.epoch + 1}')
            self.callbacks_handler.execute_epoch_begin()

            for batch_data in tqdm(self.train_loader):
                self.callbacks_handler.execute_batch_begin()

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

            # self.early_stop is changed from the early stopper callback
            if self.early_stop:
                break

        self.auto_execute_end_of_training()
        self.callbacks_handler.execute_train_end()
        logger.info(f'Final Train History: {self.train_history}', for_summary=False)

        return self.model

    def auto_execute_end_of_epoch(self):
        train_loss_batch_accum_avg = np.mean(self.loss_batch_accum)
        logger.info(f'AVG BATCH ACCUMULATED TRAIN LOSS: {train_loss_batch_accum_avg}')
        self.train_history['accumulated_loss'].append(train_loss_batch_accum_avg)
        self.loss_batch_accum = []

        train_loss = self.evaluate_loss_on_train_set()
        logger.info(f'TRAIN LOSS: {train_loss}')
        self.insert_metric_result_into_history('loss', train_loss)

        if self.validation_loader is not None:
            val_loss = self.evaluate_loss_on_validation_set()
            logger.info(f'VAL LOSS: {val_loss}')
            self.insert_metric_result_into_history('val_loss', val_loss)

    def auto_execute_end_of_training(self):
        if self.test_loader is not None:
            test_loss = self.evaluate_loss_on_test_set()
            logger.info(f'TEST LOSS: {test_loss}')
            # To keep TrainingHistory from complaining due to the non-matching metric result lengths
            # self.insert_metric_result_into_history('test_loss', test_loss)

    def evaluate_loss_on_train_set(self):
        """

        Returns:
            float:

        """
        logger.info('evaluate_loss_on_train_set', for_summary=False)
        return self.evaluate_model_loss(self.train_loader)

    def evaluate_loss_on_validation_set(self):
        """

        Returns:
            float:

        """
        logger.info('evaluate_loss_on_validation_set', for_summary=False)
        return self.evaluate_model_loss(self.validation_loader)

    def evaluate_loss_on_test_set(self):
        """

        Returns:
            float:

        """
        logger.info('evaluate_loss_on_test_set', for_summary=False)
        return self.evaluate_model_loss(self.test_loader)

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
                loss_batch = self.batch_model_feed_def.get_loss_eval(self.model, batch_data, self.criterion, self.device)

                loss_avg.append(loss_batch.item())

        self.model.train()

        return np.mean(loss_avg)

    def predict_on_train_set(self):
        """

        Returns:
            (torch.Tensor, torch.Tensor, dict):

        """
        logger.info('predict_on_train_set', for_summary=False)
        return self.predict_with_model(self.train_loader)

    def predict_on_validation_set(self):
        """

        Returns:
            (torch.Tensor, torch.Tensor, dict):

        """
        logger.info('predict_on_validation_set', for_summary=False)
        return self.predict_with_model(self.validation_loader)

    def predict_on_test_set(self):
        """

        Returns:
            (torch.Tensor, torch.Tensor, dict):

        """
        logger.info('predict_on_test_set', for_summary=False)
        return self.predict_with_model(self.test_loader)

    def predict_with_model(self, data_loader):
        """

        Args:
            data_loader (torch.utils.data.DataLoader):

        Returns:
            (torch.Tensor, torch.Tensor, dict):

        """
        y_test, y_pred, metadata_list = [], [], []

        self.model.eval()

        with torch.no_grad():
            for batch_data in tqdm(data_loader):
                y_test_batch, y_pred_batch, metadata_batch = self.batch_model_feed_def.get_predictions(self.model, batch_data,
                                                                                                       self.device)

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

            metadata = self.combine_prediction_metadata_batches(metadata_list) if len(metadata_list) > 0 else None

        self.model.train()

        return y_test, y_pred, metadata

    def insert_metric_result_into_history(self, metric_name, metric_result):
        """

        Args:
            metric_name (str):
            metric_result (float or dict):

        """
        if metric_name not in self.train_history:
            self.train_history[metric_name] = []
        self.train_history[metric_name].append(metric_result)
        logger.info(f'Train History: {self.train_history}', for_summary=False)

    @staticmethod
    def combine_prediction_metadata_batches(metadata_list):
        """

        Args:
            metadata_list (list):

        Returns:
            dict:

        """
        combined_metadata = {}

        for metadata_batch in metadata_list:
            for meta_el in metadata_batch:
                if meta_el not in combined_metadata:
                    combined_metadata[meta_el] = []
                combined_metadata[meta_el] += metadata_batch[meta_el]

        return combined_metadata


class TrainLoopModelCheckpoint(TrainLoop):
    def __init__(self, model,
                 train_loader, validation_loader, test_loader,
                 batch_model_feed_def,
                 optimizer, criterion,
                 project_name, experiment_name, local_model_result_folder_path, cloud_save_mode='s3',
                 rm_subopt_local_models=False, num_best_checkpoints_kept=2):
        """

        Args:
            model (torch.nn.modules.Module):
            train_loader (torch.utils.data.DataLoader):
            validation_loader (torch.utils.data.DataLoader):
            test_loader (torch.utils.data.DataLoader):
            batch_model_feed_def (AIToolbox.torchtrain.batch_model_feed_defs.AbstractModelFeedDefinition):
            optimizer:
            criterion:
            project_name (str):
            experiment_name (str):
            local_model_result_folder_path (str):
            cloud_save_mode (str or None): Storage destination selector.
                For AWS S3: 's3' / 'aws_s3' / 'aws'
                For Google Cloud Storage: 'gcs' / 'google_storage' / 'google storage'
                Everything else results just in local storage to disk
            rm_subopt_local_models (bool or str): if True, the deciding metric is set to 'loss'. Give string metric name
                to set it as a deciding metric for suboptimal model removal. If metric name consists of substring 'loss'
                the metric minimization is done otherwise metric maximization is done
            num_best_checkpoints_kept (int): number of best performing models which are kept when removing suboptimal
                model checkpoints

        """
        TrainLoop.__init__(self, model, train_loader, validation_loader, test_loader, batch_model_feed_def, optimizer, criterion)
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.local_model_result_folder_path = local_model_result_folder_path
        self.cloud_save_mode = cloud_save_mode
        self.rm_subopt_local_models = rm_subopt_local_models

        self.set_logger_output_dir()

        self.callbacks_handler.register_callbacks([
            ModelCheckpointCallback(self.project_name, self.experiment_name, self.local_model_result_folder_path,
                                    cloud_save_mode=self.cloud_save_mode,
                                    rm_subopt_local_models=self.rm_subopt_local_models,
                                    num_best_checkpoints_kept=num_best_checkpoints_kept)
        ])

    def set_logger_output_dir(self):
        logs_folder_path = logger.create_experiment_logs_local_folder_structure(self.local_model_result_folder_path,
                                                                                self.project_name, self.experiment_name,
                                                                                self.experiment_timestamp)
        logger.setup_logger(logs_folder_path)


class TrainLoopModelEndSave(TrainLoop):
    def __init__(self, model,
                 train_loader, validation_loader, test_loader,
                 batch_model_feed_def,
                 optimizer, criterion,
                 project_name, experiment_name, local_model_result_folder_path,
                 args, val_result_package=None, test_result_package=None, cloud_save_mode='s3'):
        """

        Args:
            model (torch.nn.modules.Module):
            train_loader (torch.utils.data.DataLoader):
            validation_loader (torch.utils.data.DataLoader or None):
            test_loader (torch.utils.data.DataLoader or None):
            batch_model_feed_def (AIToolbox.torchtrain.batch_model_feed_defs.AbstractModelFeedDefinition):
            optimizer:
            criterion:
            project_name (str):
            experiment_name (str):
            local_model_result_folder_path (str):
            args (dict):
            val_result_package (AIToolbox.experiment_save.result_package.abstract_result_packages.AbstractResultPackage or None):
            test_result_package (AIToolbox.experiment_save.result_package.abstract_result_packages.AbstractResultPackage or None):
            cloud_save_mode (str or None): Storage destination selector.
                For AWS S3: 's3' / 'aws_s3' / 'aws'
                For Google Cloud Storage: 'gcs' / 'google_storage' / 'google storage'
                Everything else results just in local storage to disk

        """
        TrainLoop.__init__(self, model, train_loader, validation_loader, test_loader, batch_model_feed_def, optimizer, criterion)
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.local_model_result_folder_path = local_model_result_folder_path
        self.args = args
        self.val_result_package = val_result_package
        self.test_result_package = test_result_package
        self.cloud_save_mode = cloud_save_mode

        self.check_if_result_packages_possible()
        self.set_logger_output_dir()
        logger.info(f'Args: {self.args}', for_summary=False)

        self.callbacks_handler.register_callbacks([
            ModelTrainEndSaveCallback(self.project_name, self.experiment_name, self.local_model_result_folder_path,
                                      self.args, self.val_result_package, self.test_result_package, 
                                      cloud_save_mode=self.cloud_save_mode)
        ])

    def check_if_result_packages_possible(self):
        if self.val_result_package is not None and self.validation_loader is None:
            raise ValueError('Given the val_result_package but not supplied the validation_loader. '
                             'If you want to calculate the val_result_package the validation_loader has to be provided.')

        if self.test_result_package is not None and self.test_loader is None:
            raise ValueError('Given the test_result_package but not supplied the test_loader. '
                             'If you want to calculate the test_result_package the test_loader has to be provided.')

        if self.val_result_package is None and self.test_result_package is None:
            raise ValueError("Both val_result_package and test_result_package are None. "
                             "At least one of these should be not None but actual result package.")

    def set_logger_output_dir(self):
        logs_folder_path = logger.create_experiment_logs_local_folder_structure(self.local_model_result_folder_path,
                                                                                self.project_name, self.experiment_name,
                                                                                self.experiment_timestamp)
        logger.setup_logger(logs_folder_path)


class TrainLoopModelCheckpointEndSave(TrainLoopModelEndSave):
    def __init__(self, model,
                 train_loader, validation_loader, test_loader,
                 batch_model_feed_def,
                 optimizer, criterion,
                 project_name, experiment_name, local_model_result_folder_path,
                 args, val_result_package=None, test_result_package=None, cloud_save_mode='s3',
                 rm_subopt_local_models=False, num_best_checkpoints_kept=2):
        """

        Args:
            model (torch.nn.modules.Module):
            train_loader (torch.utils.data.DataLoader):
            validation_loader (torch.utils.data.DataLoader or None):
            test_loader (torch.utils.data.DataLoader or None):
            batch_model_feed_def (AIToolbox.torchtrain.batch_model_feed_defs.AbstractModelFeedDefinition):
            optimizer:
            criterion:
            project_name (str):
            experiment_name (str):
            local_model_result_folder_path (str):
            args (dict):
            val_result_package (AIToolbox.experiment_save.result_package.abstract_result_packages.AbstractResultPackage or None):
            test_result_package (AIToolbox.experiment_save.result_package.abstract_result_packages.AbstractResultPackage or None):
            cloud_save_mode (str or None): Storage destination selector.
                For AWS S3: 's3' / 'aws_s3' / 'aws'
                For Google Cloud Storage: 'gcs' / 'google_storage' / 'google storage'
                Everything else results just in local storage to disk
            rm_subopt_local_models (bool or str): if True, the deciding metric is set to 'loss'. Give string metric name
                to set it as a deciding metric for suboptimal model removal. If metric name consists of substring 'loss'
                the metric minimization is done otherwise metric maximization is done
            num_best_checkpoints_kept (int): number of best performing models which are kept when removing suboptimal
                model checkpoints

        """
        TrainLoopModelEndSave.__init__(self, model, train_loader, validation_loader, test_loader, batch_model_feed_def,
                                       optimizer, criterion,
                                       project_name, experiment_name, local_model_result_folder_path,
                                       args, val_result_package, test_result_package, cloud_save_mode)
        self.rm_subopt_local_models = rm_subopt_local_models

        self.callbacks_handler.register_callbacks([
            ModelCheckpointCallback(self.project_name, self.experiment_name, self.local_model_result_folder_path,
                                    cloud_save_mode=self.cloud_save_mode,
                                    rm_subopt_local_models=self.rm_subopt_local_models,
                                    num_best_checkpoints_kept=num_best_checkpoints_kept)
        ])
