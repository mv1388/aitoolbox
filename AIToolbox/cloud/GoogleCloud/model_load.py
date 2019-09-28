from AIToolbox.cloud.GoogleCloud.data_access import BaseGoogleStorageDataLoader
from AIToolbox.cloud.AWS.model_load import PyTorchS3ModelLoader
from AIToolbox.experiment.local_load.local_model_load import AbstractLocalModelLoader, PyTorchLocalModelLoader


class BaseModelGoogleStorageLoader(BaseGoogleStorageDataLoader):
    def __init__(self, local_model_loader, local_model_result_folder_path='~/project/model_result',
                 bucket_name='model-result', cloud_dir_prefix=''):
        BaseGoogleStorageDataLoader.__init__(self, bucket_name, local_model_result_folder_path)
        self.local_model_result_folder_path = self.local_dataset_folder_path

        self.cloud_dir_prefix = cloud_dir_prefix
        self.local_model_loader = local_model_loader

        if not isinstance(local_model_loader, AbstractLocalModelLoader):
            raise TypeError('Provided local_model_loader is not inherited from AbstractLocalModelLoader as required.')


class PyTorchGoogleStorageModelLoader(BaseModelGoogleStorageLoader, PyTorchS3ModelLoader):
    def __init__(self, local_model_result_folder_path='~/project/model_result',
                 bucket_name='model-result', cloud_dir_prefix=''):
        local_model_loader = PyTorchLocalModelLoader(local_model_result_folder_path)

        BaseModelGoogleStorageLoader.__init__(self, local_model_loader,
                                              local_model_result_folder_path, bucket_name, cloud_dir_prefix)
