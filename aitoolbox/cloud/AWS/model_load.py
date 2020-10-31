import os

from aitoolbox.cloud.AWS.data_access import BaseDataLoader
from aitoolbox.experiment.local_load.local_model_load import AbstractLocalModelLoader, PyTorchLocalModelLoader
from aitoolbox.experiment.local_save.folder_create import ExperimentFolder


class BaseModelLoader(BaseDataLoader):
    def __init__(self, local_model_loader, local_model_result_folder_path='~/project/model_result',
                 bucket_name='model-result', cloud_dir_prefix=''):
        """Base saved model loading from S3 storage

        Args:
            local_model_loader (AbstractLocalModelLoader): model loader implementing the loading of the saved model for
                the selected deep learning framework
            local_model_result_folder_path (str): root local path where project folder will be created
            bucket_name (str): name of the bucket in the cloud storage from which the model will be downloaded
            cloud_dir_prefix (str): path to the folder inside the bucket where the experiments are going to be saved
        """
        BaseDataLoader.__init__(self, bucket_name, local_model_result_folder_path)
        self.local_model_result_folder_path = self.local_base_data_folder_path

        self.cloud_dir_prefix = cloud_dir_prefix
        self.local_model_loader = local_model_loader

        if not isinstance(local_model_loader, AbstractLocalModelLoader):
            raise TypeError('Provided local_model_loader is not inherited from AbstractLocalModelLoader as required.')

    def load_model(self, project_name, experiment_name, experiment_timestamp,
                   model_save_dir='checkpoint_model', epoch_num=None,
                   **kwargs):
        """Download and read/load the model

        Args:
            project_name (str): root name of the project
            experiment_name (str): name of the particular experiment
            experiment_timestamp (str): time stamp at the start of training
            model_save_dir (str): name of the folder inside experiment folder where the model is saved
            epoch_num (int or None): epoch number of the model checkpoint or none if loading final model
            **kwargs: additional local_model_loader parameters

        Returns:
            dict: model representation. (currently only returning dicts as only PyTorch model loading is supported)
        """
        cloud_model_folder_path = os.path.join(self.cloud_dir_prefix,
                                               project_name,
                                               experiment_name + '_' + experiment_timestamp,
                                               model_save_dir)
        experiment_dir_path = ExperimentFolder.create_base_folder(project_name, experiment_name, experiment_timestamp,
                                                                  self.local_model_result_folder_path)
        local_model_folder_path = os.path.join(experiment_dir_path, model_save_dir)
        if not os.path.exists(local_model_folder_path):
            os.mkdir(local_model_folder_path)

        if epoch_num is None:
            model_name = f'model_{experiment_name}_{experiment_timestamp}.pth'
        else:
            model_name = f'model_{experiment_name}_{experiment_timestamp}_E{epoch_num}.pth'

        # Loads the model save file from S3 to the local folder
        cloud_model_file_path = os.path.join(cloud_model_folder_path, model_name)
        local_model_file_path = os.path.join(local_model_folder_path, model_name)

        # Will only download from S3 if file not present on local drive
        self.load_file(cloud_model_file_path, local_model_file_path)

        return self.local_model_loader.load_model(project_name, experiment_name, experiment_timestamp,
                                                  model_save_dir, epoch_num, **kwargs)


class PyTorchS3ModelLoader(BaseModelLoader):
    def __init__(self, local_model_result_folder_path='~/project/model_result',
                 bucket_name='model-result', cloud_dir_prefix=''):
        """PyTorch S3 model downloader & loader

        Args:
            local_model_result_folder_path (str): root local path where project folder will be created
            bucket_name (str): name of the bucket in the cloud storage from which the model will be downloaded
            cloud_dir_prefix (str): path to the folder inside the bucket where the experiments are going to be saved
        """
        local_model_loader = PyTorchLocalModelLoader(local_model_result_folder_path)

        BaseModelLoader.__init__(self, local_model_loader, local_model_result_folder_path,
                                 bucket_name, cloud_dir_prefix)

    def init_model(self, model, used_data_parallel=False):
        """Initialize provided PyTorch model with the loaded model weights

        For this function to work, load_model() must be first called to read the model representation into memory.

        Args:
            model: PyTorch model
            used_data_parallel (bool): if the saved model was nn.DataParallel or normal model

        Returns:
            initialized model
        """
        return self.local_model_loader.init_model(model, used_data_parallel)

    def init_optimizer(self, optimizer, device='cuda'):
        """Initialize PyTorch optimizer

        Args:
            optimizer:
            device (str):

        Returns:
            initialized optimizer
        """
        return self.local_model_loader.init_optimizer(optimizer, device)

    def init_scheduler(self, scheduler_callbacks_list, ignore_saved=False, ignore_missing_saved=False):
        """Initialize the list of schedulers based on saved model/optimizer/scheduler checkpoint

        Args:
            scheduler_callbacks_list (list): list of scheduler (callbacks)
            ignore_saved (bool): if exception should be raised in the case there are found scheduler snapshots
                in the checkpoint, but not schedulers are provided to this method
            ignore_missing_saved (bool): if exception should be raised in the case schedulers are provided to
                this method but no saved scheduler snapshots can be found in the checkpoint

        Returns:
            list: list of initialized scheduler (callbacks)
        """
        return self.local_model_loader.init_scheduler(scheduler_callbacks_list, ignore_saved, ignore_missing_saved)

    def init_amp(self, amp_scaler):
        """Initialize AMP GradScaler

        Args:
            amp_scaler (torch.cuda.amp.GradScaler): AMP GradScaler

        Returns:
            torch.cuda.amp.GradScaler: initialized AMP GradScaler
        """
        return self.local_model_loader.init_amp(amp_scaler)
