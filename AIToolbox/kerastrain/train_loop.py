import os
import time
import datetime
import types

from AIToolbox.kerastrain.callbacks.callback_handler import CallbacksHandler
from AIToolbox.kerastrain.callbacks.callbacks import ModelCheckpoint, ModelTrainEndSave


class TrainLoop:
    def __init__(self, model,
                 train_loader, validation_loader, test_loader,
                 optimizer, criterion, metrics):
        """

        Args:
            model (keras.engine.training.Model):
            train_loader:
            validation_loader:
            test_loader:
            optimizer:
            criterion:
            metrics:
        """
        self.model = model
        
        self.optimizer = optimizer
        self.criterion = criterion
        self.metrics = metrics

        self.model.compile(optimizer=self.optimizer, loss=self.criterion, metrics=self.metrics)

        self.train_loader = train_loader
        self.x_train, self.y_train = train_loader if train_loader is not None and not self.is_generator(train_loader) else (None, None)

        self.validation_loader = validation_loader
        self.x_val, self.y_val = validation_loader if validation_loader is not None and not self.is_generator(validation_loader) else (None, None)

        self.test_loader = test_loader
        self.x_test, self.y_test = test_loader if test_loader is not None and not self.is_generator(test_loader) else (None, None)

        self.check_provided_data_loaders()

        self.callbacks_handler = CallbacksHandler(self)
        self.callbacks = []
        self.train_history = None

        self.experiment_timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S')

    def __call__(self, num_epoch, batch_size, callbacks=None, **kwargs):
        """

        Args:
            num_epoch (int):
            batch_size (int):
            callbacks (list):
            kwargs (dict):

        Returns:
            keras.engine.training.Model:
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
            keras.engine.training.Model:
        """
        self.callbacks_handler.register_callbacks(callbacks)

        if not self.is_generator(self.train_loader):
            self.train_history = self.model.fit(x=self.x_train, y=self.y_train,
                                                epochs=num_epoch, batch_size=batch_size, callbacks=self.callbacks,
                                                validation_data=self.validation_loader,
                                                **kwargs)
        else:
            self.train_history = self.model.fit_generator(generator=self.train_loader,
                                                          epochs=num_epoch, callbacks=self.callbacks,
                                                          validation_data=self.validation_loader,
                                                          **kwargs)

        self.callbacks_handler.execute_train_end_train_loop()
        return self.model

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

    def evaluate_loss_on_test_set(self):
        """

        Returns:
            float:
        """
        return self.evaluate_model_loss(self.test_loader)

    def evaluate_model_loss(self, data_loader):
        """

        Args:
            data_loader:

        Returns:
            float:
        """
        if not self.is_generator(data_loader):
            x_data = data_loader[0]
            y_data = data_loader[1]
            scores = self.model.evaluate(x=x_data, y=y_data)
        else:
            scores = self.model.evaluate_generator(data_loader)

        loss = scores[0]
        return loss

    def predict_on_train_set(self):
        """

        Returns:
            (numpy.array, numpy.array, dict):
        """
        return self.predict_with_model(self.train_loader)

    def predict_on_validation_set(self):
        """

        Returns:
            (numpy.array, numpy.array, dict):
        """
        return self.predict_with_model(self.validation_loader)

    def predict_on_test_set(self):
        """

        Returns:
            (numpy.array, numpy.array, dict):
        """
        return self.predict_with_model(self.test_loader)

    def predict_with_model(self, data_loader):
        """

        In fact in keras mode it predicts on test set

        todo: some time down the line make the dataset names correct: train, val, test

        Returns:
            (numpy.array, numpy.array, dict):
        """
        if not self.is_generator(data_loader):
            x_data = data_loader[0]
            y_data = data_loader[1]
            y_pred = self.model.predict(x_data)
        else:
            y_data = [y_batch for _, y_batch in data_loader]
            y_pred = self.model.predict_generator(data_loader)

        metadata = None

        return y_data, y_pred, metadata

    @staticmethod
    def is_generator(data_loader):
        """

        Args:
            data_loader:

        Returns:
            bool:
        """
        return isinstance(data_loader, types.GeneratorType)

    def check_provided_data_loaders(self):
        """
        
        Raises:
            ValueError
        
        Returns:
            None
        """
        if not self.is_generator(self.train_loader) and self.is_generator(self.validation_loader):
            raise ValueError('train_loader is not generator, but validation_loader is. '
                             'When train_loader is not generator, the validation_loader also can not be')


class TrainLoopModelCheckpoint(TrainLoop):
    def __init__(self, model,
                 train_loader, validation_loader, test_loader,
                 optimizer, criterion, metrics,
                 project_name, experiment_name, local_model_result_folder_path, cloud_save_mode='s3'):
        """

        Args:
            model (keras.engine.training.Model):
            train_loader:
            validation_loader:
            test_loader:
            optimizer:
            criterion:
            metrics:
            project_name (str):
            experiment_name (str):
            local_model_result_folder_path (str):
            cloud_save_mode (str or None): Storage destination selector.
                For AWS S3: 's3' / 'aws_s3' / 'aws'
                For Google Cloud Storage: 'gcs' / 'google_storage' / 'google storage'
                Everything else results just in local storage to disk
        """
        TrainLoop.__init__(self, model, train_loader, validation_loader, test_loader, optimizer, criterion, metrics)
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.local_model_result_folder_path = local_model_result_folder_path
        self.cloud_save_mode = cloud_save_mode

        self.callbacks_handler.register_callbacks([
            ModelCheckpoint(self.project_name, self.experiment_name, self.local_model_result_folder_path,
                            cloud_save_mode=self.cloud_save_mode)
        ])


class TrainLoopModelEndSave(TrainLoop):
    def __init__(self, model,
                 train_loader, validation_loader, test_loader,
                 optimizer, criterion, metrics,
                 project_name, experiment_name, local_model_result_folder_path,
                 args, val_result_package=None, test_result_package=None, cloud_save_mode='s3'):
        """

        Args:
            model (keras.engine.training.Model):
            train_loader:
            validation_loader:
            test_loader:
            optimizer:
            criterion:
            metrics:
            project_name (str):
            experiment_name (str):
            local_model_result_folder_path (str):
            args (dict):
            val_result_package (AIToolbox.experiment.result_package.abstract_result_packages.AbstractResultPackage or None):
            test_result_package (AIToolbox.experiment.result_package.abstract_result_packages.AbstractResultPackage or None):
            cloud_save_mode (str or None): Storage destination selector.
                For AWS S3: 's3' / 'aws_s3' / 'aws'
                For Google Cloud Storage: 'gcs' / 'google_storage' / 'google storage'
                Everything else results just in local storage to disk
        """
        TrainLoop.__init__(self, model, train_loader, validation_loader, test_loader, optimizer, criterion, metrics)
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.local_model_result_folder_path = os.path.expanduser(local_model_result_folder_path)
        self.args = args
        self.val_result_package = val_result_package
        self.test_result_package = test_result_package
        self.cloud_save_mode = cloud_save_mode
        
        self.check_if_result_packages_possible()

        self.callbacks_handler.register_callbacks([
            ModelTrainEndSave(self.project_name, self.experiment_name, self.local_model_result_folder_path,
                              self.args, self.val_result_package, self.test_result_package,
                              cloud_save_mode=self.cloud_save_mode)
        ])
        
    def check_if_result_packages_possible(self):
        """
        
        Raises:
            ValueError
        
        Returns:
            None
        """
        if self.val_result_package is not None and self.validation_loader is None:
            raise ValueError('Given the val_result_package but not supplied the validation_loader. '
                             'If you want to calculate the val_result_package the validation_loader has to be provided.')

        if self.test_result_package is not None and self.test_loader is None:
            raise ValueError('Given the test_result_package but not supplied the test_loader. '
                             'If you want to calculate the test_result_package the test_loader has to be provided.')

        if self.val_result_package is None and self.test_result_package is None:
            raise ValueError("Both val_result_package and test_result_package are None. "
                             "At least one of these should be not None but actual result package.")


class TrainLoopModelCheckpointEndSave(TrainLoopModelEndSave):
    def __init__(self, model,
                 train_loader, validation_loader, test_loader,
                 optimizer, criterion, metrics,
                 project_name, experiment_name, local_model_result_folder_path,
                 args, val_result_package=None, test_result_package=None, cloud_save_mode='s3'):
        """

        Args:
            model (keras.engine.training.Model):
            train_loader:
            validation_loader:
            test_loader:
            optimizer:
            criterion:
            metrics:
            project_name (str):
            experiment_name (str):
            local_model_result_folder_path (str):
            args (dict):
            val_result_package (AIToolbox.experiment.result_package.abstract_result_packages.AbstractResultPackage or None):
            test_result_package (AIToolbox.experiment.result_package.abstract_result_packages.AbstractResultPackage or None):
            cloud_save_mode (str or None): Storage destination selector.
                For AWS S3: 's3' / 'aws_s3' / 'aws'
                For Google Cloud Storage: 'gcs' / 'google_storage' / 'google storage'
                Everything else results just in local storage to disk
        """
        TrainLoopModelEndSave.__init__(self, model, train_loader, validation_loader, test_loader,
                                       optimizer, criterion, metrics,
                                       project_name, experiment_name, local_model_result_folder_path,
                                       args, val_result_package, test_result_package, cloud_save_mode)

        self.callbacks_handler.register_callbacks([
            ModelCheckpoint(self.project_name, self.experiment_name, self.local_model_result_folder_path,
                            cloud_save_mode=self.cloud_save_mode)
        ])
