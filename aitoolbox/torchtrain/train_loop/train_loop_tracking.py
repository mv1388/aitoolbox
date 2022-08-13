import os
import inspect

from aitoolbox.torchtrain.train_loop.train_loop import TrainLoop
from aitoolbox.experiment.result_package.abstract_result_packages import AbstractResultPackage
from aitoolbox.torchtrain.callbacks.model_save import ModelCheckpoint, ModelIterationCheckpoint, ModelTrainEndSave
from aitoolbox.torchtrain.train_loop.components.pred_collate_fns import append_predictions, torch_cat_transf


class TrainLoopCheckpoint(TrainLoop):
    def __init__(self, model,
                 train_loader, validation_loader, test_loader,
                 optimizer, criterion,
                 project_name, experiment_name, local_model_result_folder_path,
                 hyperparams,
                 cloud_save_mode='s3', bucket_name='model-result', cloud_dir_prefix='', source_dirs=(),
                 rm_subopt_local_models=False, num_best_checkpoints_kept=2,
                 iteration_save_freq=0,
                 collate_batch_pred_fn=append_predictions, pred_transform_fn=torch_cat_transf,
                 end_auto_eval=True, lazy_experiment_save=False, print_callbacks=False,
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
            iteration_save_freq (int): frequency of saving the model checkpoint every specified number of
                training iterations
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
            print_callbacks (bool): at the start of training print the list of registered callbacks
                which will be executed during the run of the train loop
            gpu_mode (str): GPU training mode selection. TrainLoop supports different GPU training modes by
                specifying one of the following:

                * ``'single'``: single GPU training
                * ``'dp'``: multi-GPU training via DataParallel
                * ``'ddp'``: multi-GPU training via DistributedDataParallel

            cuda_device_idx (int or None): CUDA device index used when training on multiple GPUs
            use_amp (bool or dict): use 16-bit Automatic Mixed Precision (AMP)

                To switch to AMP mode either:

                * set this parameter to ``True`` to use default AMP ``torch.cuda.amp.GradScaler`` initialization params
                * provide custom AMP ``torch.cuda.amp.GradScaler`` initialization parameters as a dict as this parameter
        """
        TrainLoop.__init__(self, model, train_loader, validation_loader, test_loader, optimizer, criterion,
                           collate_batch_pred_fn, pred_transform_fn,
                           end_auto_eval, lazy_experiment_save, print_callbacks,
                           gpu_mode, cuda_device_idx, use_amp)
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.local_model_result_folder_path = os.path.expanduser(local_model_result_folder_path)
        self.hyperparams = hyperparams
        self.cloud_save_mode = cloud_save_mode
        self.bucket_name = bucket_name
        self.cloud_dir_prefix = cloud_dir_prefix
        self.source_dirs = source_dirs
        self.rm_subopt_local_models = rm_subopt_local_models
        self.iteration_save_freq = iteration_save_freq

        if 'experiment_file_path' not in self.hyperparams:
            self.hyperparams['experiment_file_path'] = inspect.getframeinfo(inspect.currentframe().f_back).filename
        if 'source_dirs_paths' not in self.hyperparams:
            self.hyperparams['source_dirs_paths'] = source_dirs

        if iteration_save_freq == 0:
            model_checkpoint_cb = ModelCheckpoint(
                self.project_name, self.experiment_name, self.local_model_result_folder_path,
                self.hyperparams,
                cloud_save_mode=self.cloud_save_mode,
                bucket_name=bucket_name, cloud_dir_prefix=cloud_dir_prefix,
                rm_subopt_local_models=self.rm_subopt_local_models,
                num_best_checkpoints_kept=num_best_checkpoints_kept
            )
        elif iteration_save_freq > 0:
            model_checkpoint_cb = ModelIterationCheckpoint(
                iteration_save_freq,
                self.project_name, self.experiment_name, self.local_model_result_folder_path,
                self.hyperparams,
                cloud_save_mode=self.cloud_save_mode,
                bucket_name=bucket_name, cloud_dir_prefix=cloud_dir_prefix,
                rm_subopt_local_models=self.rm_subopt_local_models,
                num_best_checkpoints_kept=num_best_checkpoints_kept
            )
        else:
            raise ValueError('iteration_save_freq can have values only >= 0. '
                             f'But received value {iteration_save_freq}.')

        self.callbacks_handler.register_callbacks([model_checkpoint_cb], cache_callbacks=True)


class TrainLoopEndSave(TrainLoop):
    def __init__(self, model,
                 train_loader, validation_loader, test_loader,
                 optimizer, criterion,
                 project_name, experiment_name, local_model_result_folder_path,
                 hyperparams, val_result_package=None, test_result_package=None,
                 cloud_save_mode='s3', bucket_name='model-result', cloud_dir_prefix='', source_dirs=(),
                 collate_batch_pred_fn=append_predictions, pred_transform_fn=torch_cat_transf,
                 end_auto_eval=True, lazy_experiment_save=False, print_callbacks=False,
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
            print_callbacks (bool): at the start of training print the list of registered callbacks
                which will be executed during the run of the train loop
            gpu_mode (str): GPU training mode selection. TrainLoop supports different GPU training modes by
                specifying one of the following:

                * ``'single'``: single GPU training
                * ``'dp'``: multi-GPU training via DataParallel
                * ``'ddp'``: multi-GPU training via DistributedDataParallel

            cuda_device_idx (int or None): CUDA device index used when training on multiple GPUs
            use_amp (bool or dict): use 16-bit Automatic Mixed Precision (AMP)

                To switch to AMP mode either:

                * set this parameter to ``True`` to use default AMP ``torch.cuda.amp.GradScaler`` initialization params
                * provide custom AMP ``torch.cuda.amp.GradScaler`` initialization parameters as a dict as this parameter
        """
        TrainLoop.__init__(self, model, train_loader, validation_loader, test_loader, optimizer, criterion,
                           collate_batch_pred_fn, pred_transform_fn,
                           end_auto_eval, lazy_experiment_save, print_callbacks,
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
        self.source_dirs = source_dirs

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
                 iteration_save_freq=0,
                 collate_batch_pred_fn=append_predictions, pred_transform_fn=torch_cat_transf,
                 end_auto_eval=True, lazy_experiment_save=False, print_callbacks=False,
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
            iteration_save_freq (int): frequency of saving the model checkpoint every specified number of
                training iterations
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
            print_callbacks (bool): at the start of training print the list of registered callbacks
                which will be executed during the run of the train loop
            gpu_mode (str): GPU training mode selection. TrainLoop supports different GPU training modes by
                specifying one of the following:

                * ``'single'``: single GPU training
                * ``'dp'``: multi-GPU training via DataParallel
                * ``'ddp'``: multi-GPU training via DistributedDataParallel

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
                                  end_auto_eval, lazy_experiment_save, print_callbacks,
                                  gpu_mode, cuda_device_idx, use_amp)
        self.rm_subopt_local_models = rm_subopt_local_models
        self.iteration_save_freq = iteration_save_freq

        if iteration_save_freq == 0:
            model_checkpoint_cb = ModelCheckpoint(
                self.project_name, self.experiment_name, self.local_model_result_folder_path,
                self.hyperparams,
                cloud_save_mode=self.cloud_save_mode,
                bucket_name=bucket_name, cloud_dir_prefix=cloud_dir_prefix,
                rm_subopt_local_models=self.rm_subopt_local_models,
                num_best_checkpoints_kept=num_best_checkpoints_kept
            )
        elif iteration_save_freq > 0:
            model_checkpoint_cb = ModelIterationCheckpoint(
                iteration_save_freq,
                self.project_name, self.experiment_name, self.local_model_result_folder_path,
                self.hyperparams,
                cloud_save_mode=self.cloud_save_mode,
                bucket_name=bucket_name, cloud_dir_prefix=cloud_dir_prefix,
                rm_subopt_local_models=self.rm_subopt_local_models,
                num_best_checkpoints_kept=num_best_checkpoints_kept
            )
        else:
            raise ValueError('iteration_save_freq can have values only >= 0. '
                             f'But received value {iteration_save_freq}.')

        self.callbacks_handler.register_callbacks([model_checkpoint_cb], cache_callbacks=True)
