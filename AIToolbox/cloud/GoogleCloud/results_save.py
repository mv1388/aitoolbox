from AIToolbox.cloud.GoogleCloud.data_access import BaseGoogleStorageDataSaver
from AIToolbox.cloud.AWS.results_save import S3ResultsSaver
from AIToolbox.experiment.local_save.local_results_save import LocalResultsSaver


class BaseResultsGoogleStorageSaver(BaseGoogleStorageDataSaver):
    def __init__(self, bucket_name='model-result', local_results_folder_path='~/project/model_result'):
        """

        Args:
            bucket_name (str):
            local_results_folder_path (str):
        """
        BaseGoogleStorageDataSaver.__init__(self, bucket_name, local_results_folder_path)


class GoogleStorageResultsSaver(BaseResultsGoogleStorageSaver, S3ResultsSaver):
    def __init__(self, bucket_name='model-result', local_model_result_folder_path='~/project/model_result'):
        """

        Args:
            bucket_name (str):
            local_model_result_folder_path (str):
        """
        BaseResultsGoogleStorageSaver.__init__(self, bucket_name, local_model_result_folder_path)
        self.local_results_saver = LocalResultsSaver(local_model_result_folder_path)
