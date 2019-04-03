import os
from google.cloud import storage

from AIToolbox.cloud.AWS.model_save import KerasS3ModelSaver, TensorFlowS3ModelSaver, PyTorchS3ModelSaver
from AIToolbox.experiment_save.local_model_save import KerasLocalModelSaver, TensorFlowLocalModelSaver, PyTorchLocalModelSaver


class BaseModelGoogleStorageSaver:
    def __init__(self, bucket_name='model-result', local_model_result_folder_path='~/project/model_result',
                 checkpoint_model=False):
        """

        Args:
            bucket_name (str):
            local_model_result_folder_path (str):
            checkpoint_model (bool):
        """
        self.bucket_name = bucket_name
        self.gcs_client = storage.Client()
        self.gcs_bucket = self.gcs_client.get_bucket(bucket_name)

        self.local_model_result_folder_path = os.path.expanduser(local_model_result_folder_path)
        self.checkpoint_model = checkpoint_model

    def save_file(self, local_file_path, cloud_file_path):
        """

        Args:
            local_file_path (str):
            cloud_file_path (str):

        Returns:
            None
        """
        blob = self.gcs_bucket.blob(cloud_file_path)
        blob.upload_from_filename(local_file_path)


class KerasGoogleStorageModelSaver(BaseModelGoogleStorageSaver, KerasS3ModelSaver):
    def __init__(self, bucket_name='model-result', local_model_result_folder_path='~/project/model_result',
                 checkpoint_model=False):
        """

        Args:
            bucket_name (str):
            local_model_result_folder_path (str):
            checkpoint_model (bool):
        """
        BaseModelGoogleStorageSaver.__init__(self, bucket_name, local_model_result_folder_path, checkpoint_model)
        self.keras_local_saver = KerasLocalModelSaver(local_model_result_folder_path, checkpoint_model)
        

class TensorFlowGoogleStorageModelSaver(BaseModelGoogleStorageSaver, TensorFlowS3ModelSaver):
    def __init__(self, bucket_name='model-result', local_model_result_folder_path='~/project/model_result',
                 checkpoint_model=False):
        """

        Args:
            bucket_name (str):
            local_model_result_folder_path (str):
            checkpoint_model (bool):
        """
        BaseModelGoogleStorageSaver.__init__(self, bucket_name, local_model_result_folder_path, checkpoint_model)
        self.tf_local_saver = TensorFlowLocalModelSaver(local_model_result_folder_path, checkpoint_model)
        
        raise NotImplementedError


class PyTorchGoogleStorageModelSaver(BaseModelGoogleStorageSaver, PyTorchS3ModelSaver):
    def __init__(self, bucket_name='model-result', local_model_result_folder_path='~/project/model_result',
                 checkpoint_model=False):
        """

        Args:
            bucket_name (str):
            local_model_result_folder_path (str):
            checkpoint_model (bool):
        """
        BaseModelGoogleStorageSaver.__init__(self, bucket_name, local_model_result_folder_path, checkpoint_model)
        self.pytorch_local_saver = PyTorchLocalModelSaver(local_model_result_folder_path, checkpoint_model)
