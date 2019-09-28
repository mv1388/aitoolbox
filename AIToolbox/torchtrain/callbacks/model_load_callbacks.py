import os

from AIToolbox.torchtrain.callbacks.callbacks import AbstractCallback
from AIToolbox.cloud.AWS.model_load import PyTorchS3ModelLoader
from AIToolbox.experiment.local_load.local_model_load import PyTorchLocalModelLoader


class ModelLoadTrainingCont(AbstractCallback):
    def __init__(self,
                 saved_experiment_timestamp, saved_model_dir='checkpoint_model', epoch_num=None, used_data_parallel=False,
                 project_name=None, experiment_name=None, local_model_result_folder_path=None,
                 cloud_save_mode='s3', bucket_name='model-result', cloud_dir_prefix='', **kwargs):
        """

        Args:
            saved_experiment_timestamp (str):
            saved_model_dir (str):
            epoch_num (int or None):
            used_data_parallel (bool):
            project_name (str or None):
            experiment_name (str or None):
            local_model_result_folder_path (str or None):
            cloud_save_mode (str):
            bucket_name (str):
            cloud_dir_prefix (str):
            **kwargs:
        """
        AbstractCallback.__init__(self, 'Model loading and initialization from checkpoint before training',
                                  execution_order=0)
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.local_model_result_folder_path = os.path.expanduser(local_model_result_folder_path) \
            if local_model_result_folder_path is not None \
            else None
        self.cloud_save_mode = cloud_save_mode
        self.bucket_name = bucket_name
        self.cloud_dir_prefix = cloud_dir_prefix

        self.saved_experiment_timestamp = saved_experiment_timestamp
        self.saved_model_dir = saved_model_dir
        self.epoch_num = epoch_num
        self.used_data_parallel = used_data_parallel
        self.local_model_loader_kwargs = kwargs

        self.model_loader = None

    def on_train_loop_registration(self):
        self.try_infer_experiment_details()

        if self.cloud_save_mode == 's3' or self.cloud_save_mode == 'aws_s3' or self.cloud_save_mode == 'aws':
            self.model_loader = PyTorchS3ModelLoader(self.local_model_result_folder_path,
                                                     self.bucket_name, self.cloud_dir_prefix)
        elif self.cloud_save_mode == 'gcs' or self.cloud_save_mode == 'google_storage' or self.cloud_save_mode == 'google storage':
            raise NotImplementedError
        else:
            self.model_loader = PyTorchLocalModelLoader(self.local_model_result_folder_path)

        self.model_loader.load_model(self.project_name, self.experiment_name,
                                     self.saved_experiment_timestamp, self.saved_model_dir,
                                     self.epoch_num, **self.local_model_loader_kwargs)

        self.train_loop_obj.model = self.model_loader.init_model(self.train_loop_obj.model, self.used_data_parallel)
        self.train_loop_obj.optimizer = self.model_loader.init_optimizer(self.train_loop_obj.optimizer)

    def try_infer_experiment_details(self):
        """

        Returns:
            None

        Raises:
            AttributeError
        """
        try:
            if self.project_name is None:
                self.project_name = self.train_loop_obj.project_name
            if self.experiment_name is None:
                self.experiment_name = self.train_loop_obj.experiment_name
            if self.local_model_result_folder_path is None:
                self.local_model_result_folder_path = self.train_loop_obj.local_model_result_folder_path

            if self.cloud_save_mode == 's3' and \
                    hasattr(self.train_loop_obj,
                            'cloud_save_mode') and self.cloud_save_mode != self.train_loop_obj.cloud_save_mode:
                self.cloud_save_mode = self.train_loop_obj.cloud_save_mode
            if self.bucket_name == 'model-result' and \
                    hasattr(self.train_loop_obj, 'bucket_name') and self.bucket_name != self.train_loop_obj.bucket_name:
                self.bucket_name = self.train_loop_obj.bucket_name
            if self.cloud_dir_prefix == '' and \
                    hasattr(self.train_loop_obj,
                            'cloud_dir_prefix') and self.cloud_dir_prefix != self.train_loop_obj.cloud_dir_prefix:
                self.cloud_dir_prefix = self.train_loop_obj.cloud_dir_prefix
        except AttributeError:
            raise AttributeError('Currently used TrainLoop does not support automatic project folder structure '
                                 'creation. Project name, etc. thus can not be automatically deduced. Please provide'
                                 'it in the callback parameters instead of currently used None values.')
