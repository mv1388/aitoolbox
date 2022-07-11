from abc import ABC, abstractmethod
import boto3
import botocore
import os
import shutil


class BaseDataSaver:
    def __init__(self, bucket_name='model-result'):
        """Base class implementing S3 file saving logic

        Args:
            bucket_name (str): S3 bucket into which the files will be saved
        """
        self.bucket_name = bucket_name
        self.s3_client = boto3.client('s3')

    def save_file(self, local_file_path, cloud_file_path):
        """Save / upload file on local drive to the AWS S3

        Args:
            local_file_path (str): path to the file on the local drive
            cloud_file_path (str): destination where the file will be saved on S3 inside the specified bucket

        Returns:
            None
        """
        self.s3_client.upload_file(os.path.expanduser(local_file_path),
                                   self.bucket_name, cloud_file_path)

    def save_folder(self, local_folder_path, cloud_folder_path):
        """Save / upload the contents of the local folder on the local drive to AWS S3

        This function uploads the *contents inside* the provided local folder. If the encapsulating folder should
        also be created on the S3, specify the folder name at the end of the ``cloud_folder_path``.

        For example if:

            ``local_folder_path = '~/bla/my_folder'``

            and we want to have the content of *my_folder* also placed into the folder *my_folder on S3* then append
            *my_folder* at the end of the cloud_folder_path:

            ``cloud_folder_path = 'cloud_bla/my_folder'``

        Args:
            local_folder_path (str): local path to the folder which should be uploaded
            cloud_folder_path (str): destination path on S3 where the folder and its content should be uploaded

        Returns:
            None
        """
        for root, dirs, files in os.walk(local_folder_path):
            for filename in files:
                local_file_path = os.path.join(root, filename)
                file_path_inside_folder = os.path.relpath(local_file_path, local_folder_path)
                s3_file_path = os.path.join(cloud_folder_path, file_path_inside_folder)

                self.s3_client.upload_file(local_file_path, self.bucket_name, s3_file_path)


class BaseDataLoader:
    def __init__(self, bucket_name='dataset-store', local_base_data_folder_path='~/project/data'):
        """Base class implementing S3 file downloading logic

        Args:
            bucket_name (str): S3 bucket from which the files will be downloaded
            local_base_data_folder_path (str): local main experiment saving folder
        """
        self.bucket_name = bucket_name
        self.s3 = boto3.resource('s3')

        self.local_base_data_folder_path = os.path.expanduser(local_base_data_folder_path)
        self.available_prepocessed_datasets = []

    def load_file(self, cloud_file_path, local_file_path):
        """Download the file AWS S3 to the local drive

        Args:
            cloud_file_path (str): location where the file is saved on S3 inside the specified bucket
            local_file_path (str): destination path where the file will be downloaded to the local drive

        Returns:
            None
        """
        local_file_path = os.path.expanduser(local_file_path)

        if os.path.isfile(local_file_path):
            print('File already exists on local disk. Not downloading from S3')
        else:
            print('Local file does not exist on the local disk. Downloading from S3')
            try:
                self.s3.Bucket(self.bucket_name).download_file(cloud_file_path, local_file_path)
            except botocore.exceptions.ClientError as e:
                if e.response['Error']['Code'] == "404":
                    print("The object does not exist on S3.")
                else:
                    raise

    def exists_local_data_folder(self, data_folder_name, protect_local_folder=True):
        """Check if a specific folder exists in the base data folder

        For example, Squad dataset folder inside /data folder, or pretrained_models folder inside /model_results folder

        Args:
            data_folder_name (str):
            protect_local_folder (bool):

        Returns:
            bool:
        """
        if os.path.exists(os.path.join(self.local_base_data_folder_path, data_folder_name)) and protect_local_folder:
            print('Local folder exists and protect_local_folder in True: leaving local folder as it is.')
            return True

        if os.path.exists(os.path.join(self.local_base_data_folder_path, data_folder_name)) and not protect_local_folder:
            print('Local folder exists and protect_local_folder in False: deleting local folder copy')
            shutil.rmtree(os.path.join(self.local_base_data_folder_path, data_folder_name))

        os.mkdir(os.path.join(self.local_base_data_folder_path, data_folder_name))
        return False

    def preproc_dataset_available(self, preproc_dataset_name):
        return preproc_dataset_name in self.available_prepocessed_datasets


class AbstractDatasetFetcher(ABC):
    @abstractmethod
    def fetch_dataset(self, dataset_name=None, protect_local_folder=True):
        """

        Args:
            dataset_name (str or None):
            protect_local_folder (bool):

        Returns:
            None
        """
        pass
