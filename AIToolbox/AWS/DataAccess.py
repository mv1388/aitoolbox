from abc import ABC, abstractmethod
import boto3
import botocore
import os
import shutil


class AbstractDatasetFetcher(ABC):
    @abstractmethod
    def fetch_dataset(self, local_dataset_folder_path, protect_local_folder=True):
        pass


class SmartDatasetFetcher:
    def __init__(self, bucket_name='dataset-store'):
        """

        Args:
            bucket_name (str):
        """
        self.bucket_name = bucket_name
        self.s3 = boto3.resource('s3')

    def fetch_file(self, s3_file_path, local_file_path):
        """

        Args:
            s3_file_path (str):
            local_file_path (str):

        Returns:
            None

        """
        if os.path.isfile(local_file_path):
            print('File already exists on local disk. Not downloading from S3')
        else:
            print('Local file does not exist on the local disk. Downloading from S3')
            try:
                self.s3.Bucket(self.bucket_name).download_file(s3_file_path, local_file_path)
            except botocore.exceptions.ClientError as e:
                if e.response['Error']['Code'] == "404":
                    print("The object does not exist on S3.")
                else:
                    raise

    def exists_local_dataset_folder(self, local_dataset_folder_path, dataset_name, protect_local_folder=True):
        """

        Args:
            local_dataset_folder_path (str):
            dataset_name (str):
            protect_local_folder (bool):

        Returns:
            bool

        """
        if os.path.exists(os.path.join(local_dataset_folder_path, dataset_name)) and protect_local_folder:
            print('Local folder exists and protect_local_folder in True: leaving local folder as it is.')
            return True

        if os.path.exists(os.path.join(local_dataset_folder_path, dataset_name)):
            print('Local folder exists and protect_local_folder in False: deleting local folder copy')
            shutil.rmtree(os.path.join(local_dataset_folder_path, dataset_name))

        os.mkdir(os.path.join(local_dataset_folder_path, dataset_name))
        return False


class SQuAD2DatasetFetcher(AbstractDatasetFetcher, SmartDatasetFetcher):
    def __init__(self, bucket_name='dataset-store'):
        """

        Args:
            bucket_name (str):
        """
        SmartDatasetFetcher.__init__(self, bucket_name)

    def fetch_dataset(self, local_dataset_folder_path, protect_local_folder=True):
        """

        Args:
            local_dataset_folder_path (str):
            protect_local_folder (bool):

        Returns:
            None

        """
        if not self.exists_local_dataset_folder(local_dataset_folder_path, 'SQuAD2', protect_local_folder):
            self.fetch_file(s3_file_path='SQuAD2/train-v2.0.json',
                            local_file_path=os.path.join(local_dataset_folder_path, 'SQuAD2', 'train-v2.0.json'))
            self.fetch_file(s3_file_path='SQuAD2/dev-v2.0.json',
                            local_file_path=os.path.join(local_dataset_folder_path, 'SQuAD2', 'dev-v2.0.json'))
