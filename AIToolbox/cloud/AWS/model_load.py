import os

from AIToolbox.cloud.AWS.data_access import BaseDataLoader
from AIToolbox.experiment.local_load.local_model_load import AbstractLocalModelLoader, PyTorchLocalModelLoader


class BaseModelLoader(BaseDataLoader):
    def __init__(self, local_model_loader, project_name, experiment_folder, model_save_dir,
                 local_model_result_folder_path, pretrained_model_dir='pretrained_models',
                 bucket_name='model-result'):
        """Base saved model loading from S3 storage

        Args:
            local_model_loader (AbstractLocalModelLoader): model loader implementing the loading of the saved model for
                the selected deep learning framework
            project_name (str): root name of the project
            experiment_folder (str): name of the particular experiment. Not to be mistaken for model saver's
                experiment_name parameter. As the framework currently operates, it appends the timestamp at the end of
                the experiment name. Consequently, the experiment_folder = experiment_name + _timestamp.
            model_save_dir (str): name of the folder inside experiment folder where the model is saved
            local_model_result_folder_path (str): root local path where project folder will be created
            pretrained_model_dir (str): if dealing with pre-trained models download them locally into a separate folder
                in order not to mix the downloaded pre-trained models with those which will be subsequently trained
            bucket_name (str): name of the bucket in the cloud storage from which the model will be downloaded
        """
        BaseDataLoader.__init__(self, bucket_name, local_model_result_folder_path)

        self.project_name = project_name
        self.experiment_folder = experiment_folder

        self.pretrained_model_dir = pretrained_model_dir
        self.cloud_model_folder_path = os.path.join(project_name, experiment_folder, model_save_dir)
        self.local_model_folder_path = os.path.join(self.local_base_data_folder_path, self.pretrained_model_dir)

        self.local_model_loader = local_model_loader

        if not isinstance(local_model_loader, AbstractLocalModelLoader):
            raise TypeError('Provided local_model_loader is not inherited from AbstractLocalModelLoader as required.')

    def load_model(self, model_name, **kwargs):
        """Download and read/load the model

        Args:
            model_name (str): model file name
            **kwargs:

        Returns:
            dict: model representation. (currently only returning dicts as only PyTorch model loading is supported)
        """
        self.exists_local_data_folder(self.pretrained_model_dir)

        # Loads the model save file from S3 to the local folder
        cloud_model_file_path = os.path.join(self.cloud_model_folder_path, model_name)
        local_model_file_path = os.path.join(self.local_model_folder_path, model_name)

        self.load_file(cloud_model_file_path, local_model_file_path)

        return self.local_model_loader.load_model(model_name, self.project_name, self.experiment_folder, **kwargs)


class PyTorchS3ModelLoader(BaseModelLoader):
    def __init__(self, project_name, experiment_folder, model_save_dir,
                 local_model_result_folder_path, pretrained_model_dir='pretrained_models',
                 bucket_name='model-result'):
        """PyTorch S3 model downloader & loader

        Args:
            project_name (str): root name of the project
            experiment_folder (str): name of the particular experiment Not to be mistaken for model saver's
                experiment_name parameter. As the framework currently operates, it appends the timestamp at the end of
                the experiment name. Consequently, the experiment_folder = experiment_name + _timestamp.
            model_save_dir (str): name of the folder inside experiment folder where the model is saved
            local_model_result_folder_path (str): root local path where project folder will be created
            pretrained_model_dir (str): if dealing with pre-trained models download them locally into a separate folder
                in order not to mix the downloaded pre-trained models with those which will be subsequently trained
            bucket_name (str): name of the bucket in the cloud storage from which the model will be downloaded
        """
        local_model_folder_path = os.path.expanduser(os.path.join(local_model_result_folder_path, pretrained_model_dir))
        local_model_loader = PyTorchLocalModelLoader(local_model_folder_path)

        BaseModelLoader.__init__(self, local_model_loader,
                                 project_name, experiment_folder, model_save_dir,
                                 local_model_result_folder_path, pretrained_model_dir, bucket_name)
