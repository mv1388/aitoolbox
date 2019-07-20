import os

from AIToolbox.cloud.AWS.data_access import BaseDataFetcher
from AIToolbox.experiment_save.local_load.local_model_load import PyTorchLocalModelLoader


class BaseModelLoader(BaseDataFetcher):
    def __init__(self, project_name, experiment_folder, model_save_dir,
                 local_model_result_folder_path, pretrained_model_dir='pretrained_models',
                 bucket_name='model-result'):
        """

        Args:
            project_name:
            experiment_folder:
            model_save_dir:
            local_model_result_folder_path:
            pretrained_model_dir:
            bucket_name:
        """
        BaseDataFetcher.__init__(self, bucket_name, local_model_result_folder_path)

        self.pretrained_model_dir = pretrained_model_dir
        self.cloud_model_folder_path = os.path.join(project_name, experiment_folder, model_save_dir)


class PyTorchS3ModelLoader(BaseModelLoader):
    def __init__(self, project_name, experiment_folder, model_save_dir,
                 local_model_result_folder_path, pretrained_model_dir='pretrained_models',
                 bucket_name='model-result'):
        """

        Args:
            project_name:
            experiment_folder:
            model_save_dir:
            local_model_result_folder_path:
            pretrained_model_dir:
            bucket_name:
        """
        BaseModelLoader.__init__(self, project_name, experiment_folder, model_save_dir,
                                 local_model_result_folder_path, pretrained_model_dir, bucket_name)
        self.project_name = project_name
        self.experiment_name = experiment_folder

        self.local_model_folder_path = os.path.join(self.local_base_data_folder_path, self.pretrained_model_dir)
        self.local_model_loader = PyTorchLocalModelLoader(self.local_model_folder_path)

    def load_model(self, model_name, map_location=None):
        self.exists_local_data_folder(self.pretrained_model_dir)

        # Loads the model save file from S3 to the local folder
        cloud_model_file_path = os.path.join(self.cloud_model_folder_path, model_name)
        local_model_file_path = os.path.join(self.local_model_folder_path, model_name)

        self.fetch_file(cloud_model_file_path, local_model_file_path)

        return self.local_model_loader.load_model(model_name, self.project_name, self.experiment_name, map_location)
