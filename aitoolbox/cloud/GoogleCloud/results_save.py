from aitoolbox.cloud.GoogleCloud.data_access import BaseGoogleStorageDataSaver
from aitoolbox.cloud.AWS.results_save import S3ResultsSaver
from aitoolbox.experiment.local_save.local_results_save import LocalResultsSaver


class BaseResultsGoogleStorageSaver(BaseGoogleStorageDataSaver):
    def __init__(self, bucket_name='model-result', cloud_dir_prefix=''):
        """Base experiment results saving to Google Cloud Storage functionality

        Args:
            bucket_name (str): Google Cloud Storage bucket into which the files will be saved
            cloud_dir_prefix (str): destination folder path inside selected bucket
        """
        BaseGoogleStorageDataSaver.__init__(self, bucket_name)
        self.cloud_dir_prefix = cloud_dir_prefix


class GoogleStorageResultsSaver(BaseResultsGoogleStorageSaver, S3ResultsSaver):
    def __init__(self, bucket_name='model-result', cloud_dir_prefix='',
                 local_model_result_folder_path='~/project/model_result'):
        """Google Cloud Storage results saver

        It first saves the results files to local drive and then uploads them to GCS.

        Args:
            bucket_name (str): name of the bucket in the Google Cloud Storage to which the results files will be saved
            cloud_dir_prefix (str): destination folder path inside selected bucket
            local_model_result_folder_path (str): root local path where project folder will be created
        """
        BaseResultsGoogleStorageSaver.__init__(self, bucket_name, cloud_dir_prefix)
        self.local_results_saver = LocalResultsSaver(local_model_result_folder_path)
