from aitoolbox.cloud.GoogleCloud.data_access import BaseGoogleStorageDataLoader
from aitoolbox.cloud.AWS.model_load import PyTorchS3ModelLoader
from aitoolbox.experiment.local_load.local_model_load import AbstractLocalModelLoader, PyTorchLocalModelLoader


class BaseModelGoogleStorageLoader(BaseGoogleStorageDataLoader):
    def __init__(self, local_model_loader, local_model_result_folder_path='~/project/model_result',
                 bucket_name='model-result', cloud_dir_prefix=''):
        """Base saved model loading from Google Cloud Storage

        Args:
            local_model_loader (AbstractLocalModelLoader): model loader implementing the loading of the saved model for
                the selected deep learning framework
            local_model_result_folder_path (str): root local path where project folder will be created
            bucket_name (str): name of the bucket in the cloud storage from which the model will be downloaded
            cloud_dir_prefix (str): path to the folder inside the bucket where the experiments are going to be saved
        """
        BaseGoogleStorageDataLoader.__init__(self, bucket_name, local_model_result_folder_path)
        self.local_model_result_folder_path = self.local_dataset_folder_path

        self.cloud_dir_prefix = cloud_dir_prefix
        self.local_model_loader = local_model_loader

        if not isinstance(local_model_loader, AbstractLocalModelLoader):
            raise TypeError('Provided local_model_loader is not inherited from AbstractLocalModelLoader as required.')


class PyTorchGoogleStorageModelLoader(BaseModelGoogleStorageLoader, PyTorchS3ModelLoader):
    def __init__(self, local_model_result_folder_path='~/project/model_result',
                 bucket_name='model-result', cloud_dir_prefix=''):
        """PyTorch Google Cloud Storage model downloader & loader

        Args:
            local_model_result_folder_path (str): root local path where project folder will be created
            bucket_name (str): name of the bucket in the cloud storage from which the model will be downloaded
            cloud_dir_prefix (str): path to the folder inside the bucket where the experiments are going to be saved
        """
        local_model_loader = PyTorchLocalModelLoader(local_model_result_folder_path)

        BaseModelGoogleStorageLoader.__init__(self, local_model_loader,
                                              local_model_result_folder_path, bucket_name, cloud_dir_prefix)
