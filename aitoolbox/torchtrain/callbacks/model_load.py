from aitoolbox.torchtrain.callbacks.abstract import AbstractExperimentCallback
from aitoolbox.cloud.AWS.model_load import PyTorchS3ModelLoader
from aitoolbox.cloud.GoogleCloud.model_load import PyTorchGoogleStorageModelLoader
from aitoolbox.experiment.local_load.local_model_load import PyTorchLocalModelLoader
from aitoolbox.cloud import s3_available_options, gcs_available_options


class ModelLoadContinueTraining(AbstractExperimentCallback):
    def __init__(self,
                 saved_experiment_timestamp, saved_model_dir='checkpoint_model', epoch_num=None,
                 ignore_saved_schedulers=False, ignore_missing_saved_schedulers=False,
                 used_data_parallel=False, custom_local_loader_class=None,
                 project_name=None, experiment_name=None, local_model_result_folder_path=None,
                 cloud_save_mode=None, bucket_name=None, cloud_dir_prefix=None, **kwargs):
        """(Down)load previously trained and saved model and continue training from this snapshot instead from beginning

        Args:
            saved_experiment_timestamp (str): timestamp of the saved model experiment
            saved_model_dir (str): folder where saved model file is inside main experiment folder
            epoch_num (int or None): if loading checkpoint model instead of final model this parameter indicates
                from which epoch of training the model will be loaded
            ignore_saved_schedulers (bool): if exception should be raised in the case there are found scheduler
                snapshots in the checkpoint, but not schedulers are provided to this method
            ignore_missing_saved_schedulers (bool): if exception should be raised in the case schedulers are provided
                to this method but no saved scheduler snapshots can be found in the checkpoint
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
        AbstractExperimentCallback.__init__(self, 'Model loading and initialization from checkpoint before training',
                                            project_name, experiment_name, local_model_result_folder_path,
                                            cloud_save_mode, bucket_name, cloud_dir_prefix,
                                            execution_order=-10)
        self.saved_experiment_timestamp = saved_experiment_timestamp
        self.saved_model_dir = saved_model_dir
        self.epoch_num = epoch_num
        self.ignore_saved_schedulers = ignore_saved_schedulers
        self.ignore_missing_saved_schedulers = ignore_missing_saved_schedulers
        self.used_data_parallel = used_data_parallel
        self.custom_local_loader_class = custom_local_loader_class
        self.local_loader_kwargs = kwargs

        self.model_loader = None

    def on_train_loop_registration(self):
        self.try_infer_experiment_details(infer_cloud_details=True)
        self.init_model_loader()

        model_representation = self.model_loader.load_model(self.project_name, self.experiment_name,
                                                            self.saved_experiment_timestamp, self.saved_model_dir,
                                                            self.epoch_num, **self.local_loader_kwargs)

        self.train_loop_obj.model = self.model_loader.init_model(self.train_loop_obj.model,
                                                                 self.used_data_parallel)
        self.train_loop_obj.optimizer = self.model_loader.init_optimizer(self.train_loop_obj.optimizer)
        if self.train_loop_obj.use_amp:
            self.train_loop_obj.amp_scaler = self.model_loader.init_amp(self.train_loop_obj.amp_scaler)

        self.train_loop_obj.epoch = model_representation['epoch'] + 1

    def on_train_begin(self):
        # Not doing in on_train_loop_registration() in order to ensure
        # schedulers are initialised inside the scheduler callbacks
        schedulers = self.train_loop_obj.get_schedulers()
        self.model_loader.init_scheduler(
            schedulers,
            ignore_saved=self.ignore_saved_schedulers,
            ignore_missing_saved=self.ignore_missing_saved_schedulers
        )

    def init_model_loader(self):
        """Initialize model loader object based on provided arguments to the callback object

        Returns:
            None
        """
        if self.cloud_save_mode in s3_available_options:
            self.model_loader = PyTorchS3ModelLoader(self.local_model_result_folder_path,
                                                     self.bucket_name, self.cloud_dir_prefix)
            if self.custom_local_loader_class is not None:
                self.model_loader.local_model_loader = self.custom_local_loader_class(self.local_model_result_folder_path)

        elif self.cloud_save_mode in gcs_available_options:
            self.model_loader = PyTorchGoogleStorageModelLoader(self.local_model_result_folder_path,
                                                                self.bucket_name, self.cloud_dir_prefix)
            if self.custom_local_loader_class is not None:
                self.model_loader.local_model_loader = self.custom_local_loader_class(self.local_model_result_folder_path)

        else:
            if self.custom_local_loader_class is None:
                self.model_loader = PyTorchLocalModelLoader(self.local_model_result_folder_path)
            else:
                self.model_loader = self.custom_local_loader_class(self.local_model_result_folder_path)
