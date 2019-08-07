from AIToolbox.cloud.GoogleCloud.data_access import BaseGoogleStorageDataSaver
from AIToolbox.cloud.AWS.model_save import KerasS3ModelSaver, TensorFlowS3ModelSaver, PyTorchS3ModelSaver
from AIToolbox.experiment.local_save.local_model_save import KerasLocalModelSaver, TensorFlowLocalModelSaver, PyTorchLocalModelSaver


class BaseModelGoogleStorageSaver(BaseGoogleStorageDataSaver):
    def __init__(self, bucket_name='model-result', cloud_dir_prefix='', checkpoint_model=False):
        """

        Args:
            bucket_name (str):
            cloud_dir_prefix (str):
            checkpoint_model (bool):
        """
        BaseGoogleStorageDataSaver.__init__(self, bucket_name)
        self.cloud_dir_prefix = cloud_dir_prefix
        self.checkpoint_model = checkpoint_model


class KerasGoogleStorageModelSaver(BaseModelGoogleStorageSaver, KerasS3ModelSaver):
    def __init__(self, bucket_name='model-result', cloud_dir_prefix='',
                 local_model_result_folder_path='~/project/model_result', checkpoint_model=False):
        """

        Args:
            bucket_name (str):
            cloud_dir_prefix (str):
            local_model_result_folder_path (str):
            checkpoint_model (bool):
        """
        BaseModelGoogleStorageSaver.__init__(self, bucket_name, cloud_dir_prefix, checkpoint_model)
        self.keras_local_saver = KerasLocalModelSaver(local_model_result_folder_path, checkpoint_model)
        

class TensorFlowGoogleStorageModelSaver(BaseModelGoogleStorageSaver, TensorFlowS3ModelSaver):
    def __init__(self, bucket_name='model-result', cloud_dir_prefix='',
                 local_model_result_folder_path='~/project/model_result', checkpoint_model=False):
        """

        Args:
            bucket_name (str):
            cloud_dir_prefix (str):
            local_model_result_folder_path (str):
            checkpoint_model (bool):
        """
        BaseModelGoogleStorageSaver.__init__(self, bucket_name, cloud_dir_prefix, checkpoint_model)
        self.tf_local_saver = TensorFlowLocalModelSaver(local_model_result_folder_path, checkpoint_model)
        
        raise NotImplementedError


class PyTorchGoogleStorageModelSaver(BaseModelGoogleStorageSaver, PyTorchS3ModelSaver):
    def __init__(self, bucket_name='model-result', cloud_dir_prefix='',
                 local_model_result_folder_path='~/project/model_result', checkpoint_model=False):
        """

        Args:
            bucket_name (str):
            cloud_dir_prefix (str):
            local_model_result_folder_path (str):
            checkpoint_model (bool):
        """
        BaseModelGoogleStorageSaver.__init__(self, bucket_name, cloud_dir_prefix, checkpoint_model)
        self.pytorch_local_saver = PyTorchLocalModelSaver(local_model_result_folder_path, checkpoint_model)
