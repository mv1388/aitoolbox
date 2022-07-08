from aitoolbox.cloud.GoogleCloud.data_access import BaseGoogleStorageDataSaver
from aitoolbox.cloud.AWS.model_save import PyTorchS3ModelSaver, KerasS3ModelSaver
from aitoolbox.experiment.local_save.local_model_save import PyTorchLocalModelSaver, KerasLocalModelSaver


class BaseModelGoogleStorageSaver(BaseGoogleStorageDataSaver):
    def __init__(self, bucket_name='model-result', cloud_dir_prefix='', checkpoint_model=False):
        """Base model saving to Google Cloud Storage functionality

        Args:
            bucket_name (str): Google Cloud Storage bucket into which the files will be saved
            cloud_dir_prefix (str): destination folder path inside selected bucket
            checkpoint_model (bool): if the model that is going to be saved is final model or mid-training checkpoint
        """
        BaseGoogleStorageDataSaver.__init__(self, bucket_name)
        self.cloud_dir_prefix = cloud_dir_prefix
        self.checkpoint_model = checkpoint_model


class PyTorchGoogleStorageModelSaver(BaseModelGoogleStorageSaver, PyTorchS3ModelSaver):
    def __init__(self, bucket_name='model-result', cloud_dir_prefix='',
                 local_model_result_folder_path='~/project/model_result', checkpoint_model=False):
        """PyTorch Google Cloud Storage model saving

        Args:
            bucket_name (str): name of the bucket in the Google Cloud Storage to which the models will be saved
            cloud_dir_prefix (str): destination folder path inside selected bucket
            local_model_result_folder_path (str): root local path where project folder will be created
            checkpoint_model (bool): if the model being saved is checkpoint model or final end of training model
        """
        BaseModelGoogleStorageSaver.__init__(self, bucket_name, cloud_dir_prefix, checkpoint_model)
        self.pytorch_local_saver = PyTorchLocalModelSaver(local_model_result_folder_path, checkpoint_model)


class KerasGoogleStorageModelSaver(BaseModelGoogleStorageSaver, KerasS3ModelSaver):
    def __init__(self, bucket_name='model-result', cloud_dir_prefix='',
                 local_model_result_folder_path='~/project/model_result', checkpoint_model=False):
        """Keras Google Storage model saving

        Args:
            bucket_name (str): name of the bucket in the Google Cloud Storage to which the models will be saved
            cloud_dir_prefix (str): destination folder path inside selected bucket
            local_model_result_folder_path (str): root local path where project folder will be created
            checkpoint_model (bool): if the model being saved is checkpoint model or final end of training model
        """
        BaseModelGoogleStorageSaver.__init__(self, bucket_name, cloud_dir_prefix, checkpoint_model)
        self.keras_local_saver = KerasLocalModelSaver(local_model_result_folder_path, checkpoint_model)
