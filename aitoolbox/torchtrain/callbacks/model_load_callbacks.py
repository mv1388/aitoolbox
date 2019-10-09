import os

from aitoolbox.torchtrain.callbacks.callbacks import AbstractCallback
from aitoolbox.cloud.AWS.model_load import PyTorchS3ModelLoader
from aitoolbox.cloud.GoogleCloud.model_load import PyTorchGoogleStorageModelLoader
from aitoolbox.experiment.local_load.local_model_load import PyTorchLocalModelLoader


class ModelLoadContinueTraining(AbstractCallback):
    def __init__(self,
                 saved_experiment_timestamp, saved_model_dir='checkpoint_model', epoch_num=None,
                 used_data_parallel=False, custom_local_loader_class=None,
                 project_name=None, experiment_name=None, local_model_result_folder_path=None,
                 cloud_save_mode='s3', bucket_name='model-result', cloud_dir_prefix='', **kwargs):
        """(Down)load previously trained and saved model and continue training from this snapshot instead from beginning

        Args:
            saved_experiment_timestamp (str): timestamp of the saved model experiment
            saved_model_dir (str): folder where saved model file is inside main experiment folder
            epoch_num (int or None): if loading checkpoint model instead of final model this parameter indicates
                from which epoch of training the model will be loaded
            used_data_parallel (bool): if the saved model was nn.DataParallel or normal model
            custom_local_loader_class (AbstractLocalModelLoader class or None): provide a custom local PyTorch model
                loader definition in case the default one is not suitable for particular use case. For example,
                in the case of complex custom optimizer initialization.
            project_name (str or None): root name of the project
            experiment_name (str or None): name of the particular experiment
            local_model_result_folder_path (str or None): root local path where project folder will be created
            cloud_save_mode (str or None): Storage destination selector.
                For AWS S3: 's3' / 'aws_s3' / 'aws'
                For Google Cloud Storage: 'gcs' / 'google_storage' / 'google storage'
                Everything else results just in local storage to disk
            bucket_name (str): name of the bucket in the cloud storage
            cloud_dir_prefix (str): path to the folder inside the bucket where the experiments are going to be saved
            **kwargs: additional parameters for the local model loader load_model() function
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
        self.custom_local_loader_class = custom_local_loader_class
        self.local_loader_kwargs = kwargs

        self.model_loader = None

    def on_train_loop_registration(self):
        self.try_infer_experiment_details()

        if self.cloud_save_mode in ['s3', 'aws_s3', 'aws']:
            self.model_loader = PyTorchS3ModelLoader(self.local_model_result_folder_path,
                                                     self.bucket_name, self.cloud_dir_prefix)
            if self.custom_local_loader_class is not None:
                self.model_loader.local_model_loader = self.custom_local_loader_class(self.local_model_result_folder_path)

        elif self.cloud_save_mode in ['gcs', 'google_storage', 'google storage']:
            self.model_loader = PyTorchGoogleStorageModelLoader(self.local_model_result_folder_path,
                                                                self.bucket_name, self.cloud_dir_prefix)
            if self.custom_local_loader_class is not None:
                self.model_loader.local_model_loader = self.custom_local_loader_class(self.local_model_result_folder_path)

        else:
            if self.custom_local_loader_class is None:
                self.model_loader = PyTorchLocalModelLoader(self.local_model_result_folder_path)
            else:
                self.model_loader = self.custom_local_loader_class(self.local_model_result_folder_path)

        model_representation = self.model_loader.load_model(self.project_name, self.experiment_name,
                                                            self.saved_experiment_timestamp, self.saved_model_dir,
                                                            self.epoch_num, **self.local_loader_kwargs)

        self.train_loop_obj.model = self.model_loader.init_model(self.train_loop_obj.model,
                                                                 self.used_data_parallel)
        self.train_loop_obj.optimizer = self.model_loader.init_optimizer(self.train_loop_obj.optimizer)
        if self.train_loop_obj.use_amp:
            self.model_loader.init_amp()

        self.train_loop_obj.epoch = model_representation['epoch'] + 1

    def try_infer_experiment_details(self):
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
