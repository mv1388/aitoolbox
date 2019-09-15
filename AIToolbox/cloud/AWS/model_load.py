import os

from AIToolbox.cloud.AWS.data_access import BaseDataLoader
from AIToolbox.experiment.local_load.local_model_load import AbstractLocalModelLoader, PyTorchLocalModelLoader


class BaseModelLoader(BaseDataLoader):
    def __init__(self, local_model_loader, project_name, experiment_folder, model_save_dir,
                 local_model_result_folder_path, pretrained_model_dir='pretrained_models',
                 bucket_name='model-result'):
        """

        Args:
            local_model_loader (AbstractLocalModelLoader):
            project_name (str):
            experiment_folder (str):
            model_save_dir (str):
            local_model_result_folder_path (str):
            pretrained_model_dir (str):
            bucket_name (str):
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
        """

        Args:
            model_name (str):
            **kwargs:

        Returns:
            dict:
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
        """

        Args:
            project_name (str):
            experiment_folder (str):
            model_save_dir (str):
            local_model_result_folder_path (str):
            pretrained_model_dir (str):
            bucket_name (str):
        """
        local_model_folder_path = os.path.expanduser(os.path.join(local_model_result_folder_path, pretrained_model_dir))
        local_model_loader = PyTorchLocalModelLoader(local_model_folder_path)

        BaseModelLoader.__init__(self, local_model_loader,
                                 project_name, experiment_folder, model_save_dir,
                                 local_model_result_folder_path, pretrained_model_dir, bucket_name)
