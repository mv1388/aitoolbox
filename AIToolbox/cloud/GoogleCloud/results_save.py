import os
from google.cloud import storage

from AIToolbox.cloud.AWS.results_save import S3ResultsSaver
from AIToolbox.experiment_save.local_save.local_results_save import LocalResultsSaver


class BaseResultsGoogleStorageSaver:
    def __init__(self, bucket_name='model-result', local_results_folder_path='~/project/model_result'):
        """

        Args:
            bucket_name (str):
            local_results_folder_path (str):

        """
        self.bucket_name = bucket_name
        self.gcs_client = storage.Client()
        self.gcs_bucket = self.gcs_client.get_bucket(bucket_name)

        self.local_model_result_folder_path = os.path.expanduser(local_results_folder_path)

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


class GoogleStorageResultsSaver(BaseResultsGoogleStorageSaver, S3ResultsSaver):
    def __init__(self, bucket_name='model-result', local_model_result_folder_path='~/project/model_result'):
        """

        Args:
            bucket_name (str):
            local_model_result_folder_path (str):

        """
        BaseResultsGoogleStorageSaver.__init__(self, bucket_name, local_model_result_folder_path)
        self.local_results_saver = LocalResultsSaver(local_model_result_folder_path)
