import os
from google.cloud import storage


class BaseGoogleStorageDataSaver:
    def __init__(self, bucket_name='model-result'):
        """

        Args:
            bucket_name (str):
        """
        self.bucket_name = bucket_name
        self.gcs_client = storage.Client()
        self.gcs_bucket = self.gcs_client.get_bucket(bucket_name)

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


class BaseGoogleStorageDataLoader:
    def __init__(self, bucket_name='dataset-store', local_dataset_folder_path='~/project/data'):
        """

        Args:
            bucket_name (str):
            local_dataset_folder_path (str):
        """
        self.bucket_name = bucket_name
        self.gcs_client = storage.Client()
        self.gcs_bucket = self.gcs_client.get_bucket(bucket_name)

        self.local_dataset_folder_path = os.path.expanduser(local_dataset_folder_path)
        self.available_prepocessed_datasets = []

    def load_file(self, cloud_file_path, local_file_path):
        """

        Args:
            cloud_file_path (str):
            local_file_path (str):

        Returns:
            None
        """
        local_file_path = os.path.expanduser(local_file_path)

        if os.path.isfile(local_file_path):
            print('File already exists on local disk. Not downloading from Google Cloud Storage')
        else:
            print('Local file does not exist on the local disk. Downloading from Google Cloud Storage.')
            blob = self.gcs_bucket.blob(cloud_file_path)
            blob.download_to_filename(local_file_path)
