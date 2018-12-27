from tqdm import tqdm
import time
import datetime
import numpy as np
import torch

from AIToolbox.AWS.model_save import PyTorchS3ModelSaver
from AIToolbox.experiment_save.experiment_saver import FullPyTorchExperimentS3Saver
from AIToolbox.experiment_save.training_history import PyTorchTrainingHistory


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
        self.loss_avg = []

        self.train_history = {'loss': [], 'val_loss': []} if self.validation_loader is not None else {'loss': []}
        self.callbacks = []

    def __call__(self, num_epoch, callbacks=None):
        """

        Args:
            num_epoch (int):
            callbacks (list):

        Returns:

        """
        self.do_train(num_epoch, callbacks)

    def do_train(self, num_epoch, callbacks=None):
        """

        Args:
            num_epoch (int):
            callbacks (list):

        Returns:

        """
        self.register_callbacks(callbacks)

        self.model = self.model.to(self.device)
        self.model.train()

        for epoch in range(num_epoch):
            print(f'Epoch: {epoch + 1}')

            for batch_data in tqdm(self.train_loader):
                loss_batch = self.batch_model_feed_def.get_loss(self.model, batch_data, self.criterion, self.device)

                # print(f'Loss: {loss_batch}')
                self.loss_avg.append(float(loss_batch))

                self.optimizer.zero_grad()
                loss_batch.backward()
                self.optimizer.step()

            # Automatic end of epoch code - reports the train and if available validation loss and executes callbacks
            self.auto_execute_end_of_epoch()
            # Customized end of epoch code
            self.on_end_of_epoch(epoch)

        # Customized end of training code
        self.on_end_of_training()

    def on_end_of_epoch(self, epoch):
        pass

    def on_end_of_training(self):
        pass

    def auto_execute_end_of_epoch(self):
        train_loss_avg = np.mean(self.loss_avg)
        print(f'AVG TRAIN LOSS: {train_loss_avg}')
        self.loss_avg = []
        self.train_history['loss'].append(train_loss_avg)

        if self.validation_loader is not None:
            val_loss_batch = self.evaluate_loss_on_validation()
            print(f'VAL LOSS: {val_loss_batch}')
            self.train_history['val_loss'].append(val_loss_batch)

        # Execute registered callbacks
        self.execute_callbacks()

    def evaluate_loss_on_validation(self):
        """

        Returns:
            float:
        """
        self.model.eval()
        val_loss_avg = []

        with torch.no_grad():
            for batch_data in tqdm(self.validation_loader):
                val_loss_batch = self.batch_model_feed_def.get_loss(self.model, batch_data, self.criterion, self.device)

                val_loss_avg.append(float(val_loss_batch))

        self.model.train()

        return np.mean(val_loss_avg)
    
    def predict_on_validation_set(self):
        y_test, y_pred = [], []

        self.model.eval()

        with torch.no_grad():
            for batch_data in tqdm(self.validation_loader):
                y_test_batch, y_pred_batch = self.batch_model_feed_def.get_predictions(self.model, batch_data, self.device)

                # TODO: check if it is the best idea to append predictions to the list and not to some torch tensor
                # TODO: also if append is the best option and not the concat
                y_test.append(y_test_batch)
                y_pred.append(y_pred_batch)

        self.model.train()

        return y_test, y_pred

    def register_callbacks(self, callbacks):
        if callbacks is not None and len(callbacks) > 0:
            self.callbacks = callbacks

    def execute_callbacks(self):
        for callback in self.callbacks:
            callback.execute(self)


class TrainLoopModelEndSave(TrainLoop):
    def __init__(self, model,
                 train_loader, validation_loader,
                 batch_model_feed_def,
                 optimizer, criterion,
                 project_name, experiment_name, local_model_result_folder_path, args,
                 result_package_class):
        """

        Args:
            model (torch.nn.modules.Module):
            train_loader (torch.utils.data.DataLoader):
            validation_loader (torch.utils.data.DataLoader):
            batch_model_feed_def:
            optimizer:
            criterion:
            project_name (str):
            experiment_name (str):
            local_model_result_folder_path (str):
            result_package_class:
        """
        TrainLoop.__init__(self, model, train_loader, validation_loader, batch_model_feed_def, optimizer, criterion)
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.local_model_result_folder_path = local_model_result_folder_path
        self.args = args
        self.result_package_class = result_package_class

        self.results_saver = FullPyTorchExperimentS3Saver(self.project_name, self.experiment_name,
                                                          experiment_timestamp=self.experiment_timestamp,
                                                          local_model_result_folder_path=self.local_model_result_folder_path)

    def on_end_of_training(self):
        train_hist_pkg = PyTorchTrainingHistory(self.train_history, 
                                                list(range(len(self.train_history[list(self.train_history.keys())[0]]))))

        y_test, y_pred = self.predict_on_validation_set()
        result_pkg = self.result_package_class(y_test, y_pred, 
                                               hyperparameters=self.args, training_history=train_hist_pkg)
        
        self.results_saver.save_experiment(self.model, result_pkg, save_true_pred_labels=True)


class TrainLoopModelCheckpointEndSave(TrainLoopModelEndSave):
    def __init__(self, model,
                 train_loader, validation_loader,
                 batch_model_feed_def,
                 optimizer, criterion,
                 project_name, experiment_name, local_model_result_folder_path, args,
                 result_package_class):
        """

        Args:
            model (torch.nn.modules.Module):
            train_loader (torch.utils.data.DataLoader):
            validation_loader (torch.utils.data.DataLoader):
            batch_model_feed_def:
            optimizer:
            criterion:
            project_name (str):
            experiment_name (str):
            local_model_result_folder_path (str):
            result_package_class:
        """
        TrainLoopModelEndSave.__init__(self, model, train_loader, validation_loader, batch_model_feed_def,
                                       optimizer, criterion,
                                       project_name, experiment_name, local_model_result_folder_path,
                                       args, result_package_class)

        self.model_checkpointer = PyTorchS3ModelSaver(local_model_result_folder_path=self.local_model_result_folder_path,
                                                      checkpoint_model=True)

    def on_end_of_epoch(self, epoch):
        """

        Args:
            epoch (int):

        Returns:

        """
        self.model_checkpointer.save_model(model=self.model,
                                           project_name=self.project_name,
                                           experiment_name=self.experiment_name,
                                           experiment_timestamp=self.experiment_timestamp,
                                           epoch=epoch,
                                           protect_existing_folder=True)


class TrainLoopModelCheckpoint(TrainLoopModelCheckpointEndSave):
    def __init__(self, model,
                 train_loader, validation_loader,
                 batch_model_feed_def,
                 optimizer, criterion,
                 project_name, experiment_name, local_model_result_folder_path, args,
                 result_package_class):
        """

        Args:
            model (torch.nn.modules.Module):
            train_loader (torch.utils.data.DataLoader):
            validation_loader (torch.utils.data.DataLoader):
            batch_model_feed_def:
            optimizer:
            criterion:
            project_name (str):
            experiment_name (str):
            local_model_result_folder_path (str):
            result_package_class:
        """
        TrainLoopModelCheckpointEndSave.__init__(self, model, train_loader, validation_loader, batch_model_feed_def,
                                                 optimizer, criterion,
                                                 project_name, experiment_name, local_model_result_folder_path,
                                                 args, result_package_class)

    def on_end_of_training(self):
        """Disable on end of training hook which is inherited

        The train loop consequently only checkpoints the models after each epoch but does nothing at the end of training

        Returns:
            None
        """
        return None
