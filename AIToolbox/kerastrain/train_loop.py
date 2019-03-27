import time
import datetime
import numpy as np

from AIToolbox.kerastrain.callbacks.callbacks import AbstractCallback, ModelCheckpointCallback, ModelTrainEndSaveCallback


class TrainLoop:
    def __init__(self, model,
                 optimizer, criterion, metrics, use_fit_generator=False,
                 test_loader=None):
        """

        Args:
            model (keras.engine.training.Model):
            optimizer:
            criterion:
            metrics:
            use_fit_generator:
            test_loader:
        """
        self.model = model
        
        self.optimizer = optimizer
        self.criterion = criterion
        self.metrics = metrics
        self.use_fit_generator = use_fit_generator

        self.test_loader = test_loader
        self.x_test, self.y_test = test_loader if test_loader is not None or not callable(test_loader) else (None, None)

        self.callbacks = []
        self.train_history = None

        self.experiment_timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')

    def __call__(self, num_epoch, batch_size, callbacks=None, **kwargs):
        """

        Args:
            num_epoch (int):
            batch_size (int):
            callbacks (list):
            kwargs (dict):

        Returns:

        """
        return self.do_train(num_epoch, batch_size, callbacks, **kwargs)

    def do_train(self, num_epoch, batch_size, callbacks=None, **kwargs):
        """

        Args:
            num_epoch (int):
            batch_size (int):
            callbacks (list):
            kwargs (dict):

        Returns:

        """
        self.register_callbacks(callbacks)
        self.model.compile(optimizer=self.optimizer, loss=self.criterion, metrics=self.metrics)
        
        if not self.use_fit_generator:
            self.train_history = self.model.fit(epochs=num_epoch, batch_size=batch_size, callbacks=self.callbacks, **kwargs)
        else:
            self.train_history = self.model.fit_generator(epochs=num_epoch, callbacks=self.callbacks, **kwargs)

        self.execute_callbacks_on_train_end_train_loop()
        return self.model
        
    def predict_on_validation_set(self):
        """

        In fact in keras mode it predicts on test set

        todo: some time down the line make the dataset names correct: train, val, test

        Returns:

        """
        if not self.use_fit_generator:
            y_pred = self.model.predict(self.x_test)
            y_test = self.y_test
        else:
            y_pred = self.model.predict_generator(self.test_loader)
            y_test = [y_batch for _, y_batch in self.test_loader]

        metadata = None

        return y_test, y_pred, metadata

    def register_callbacks(self, callbacks):
        """

        Args:
            callbacks (list):

        Returns:

        """
        if callbacks is not None and len(callbacks) > 0:
            self.callbacks += [cb.register_train_loop_object(self) if isinstance(cb, AbstractCallback) else cb 
                               for cb in callbacks]
            
    def execute_callbacks_on_train_end_train_loop(self):
        for cb in self.callbacks:
            if isinstance(cb, AbstractCallback):
                cb.on_train_end_train_loop()


class TrainLoopModelCheckpoint(TrainLoop):
    def __init__(self, model,
                 optimizer, criterion, metrics,
                 project_name, experiment_name, local_model_result_folder_path,
                 use_fit_generator=False, test_loader=None, save_to_s3=True):
        TrainLoop.__init__(self, model, optimizer, criterion, metrics, use_fit_generator, test_loader)
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.local_model_result_folder_path = local_model_result_folder_path
        self.save_to_s3 = save_to_s3

        self.register_callbacks([
            ModelCheckpointCallback(self.project_name, self.experiment_name, self.local_model_result_folder_path,
                                    save_to_s3=self.save_to_s3)
        ])


class TrainLoopModelEndSave(TrainLoop):
    def __init__(self, model,
                 optimizer, criterion, metrics,
                 project_name, experiment_name, local_model_result_folder_path,
                 args, result_package,
                 use_fit_generator=False, test_loader=None, save_to_s3=True):
        TrainLoop.__init__(self, model, optimizer, criterion, metrics, use_fit_generator, test_loader)
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.local_model_result_folder_path = local_model_result_folder_path
        self.args = args
        self.result_package = result_package
        self.save_to_s3 = save_to_s3

        self.register_callbacks([
            ModelTrainEndSaveCallback(self.project_name, self.experiment_name, self.local_model_result_folder_path,
                                      self.args, self.result_package, save_to_s3=self.save_to_s3)
        ])


class TrainLoopModelCheckpointEndSave(TrainLoopModelEndSave):
    def __init__(self, model,
                 optimizer, criterion, metrics,
                 project_name, experiment_name, local_model_result_folder_path,
                 args, result_package,
                 use_fit_generator=False, test_loader=None, save_to_s3=True):
        TrainLoopModelEndSave.__init__(self, model, optimizer, criterion, metrics,
                                       project_name, experiment_name, local_model_result_folder_path,
                                       args, result_package,
                                       use_fit_generator, test_loader, save_to_s3)
        self.register_callbacks([
            ModelCheckpointCallback(self.project_name, self.experiment_name, self.local_model_result_folder_path,
                                    save_to_s3=self.save_to_s3)
        ])
