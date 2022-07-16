import unittest

import os
import boto3
from moto import mock_s3

from tests.setup_moto_env import setup_aws_for_test
from aitoolbox.cloud.AWS.data_access import BaseDataSaver, BaseDataLoader

setup_aws_for_test()
BUCKET_NAME = 'test-bucket'
THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class TestBaseDataSaver(unittest.TestCase):
    @mock_s3
    def test_data_upload(self):
        s3 = boto3.resource('s3', region_name='us-east-1')
        s3.create_bucket(Bucket=BUCKET_NAME)
        s3_client = boto3.client('s3')

        data_saver = BaseDataSaver(bucket_name=BUCKET_NAME)
        data_saver.save_file(os.path.join(THIS_DIR, 'resources/file.txt'), 'file.txt')

        bucket_content_1 = [el['Key'] for el in s3_client.list_objects(Bucket=BUCKET_NAME)['Contents']]
        self.assertEqual(bucket_content_1, ['file.txt'])

        data_saver.save_file(os.path.join(THIS_DIR, 'resources/file.txt'), 'some_folder/file.txt')
        bucket_content_2 = [el['Key'] for el in s3_client.list_objects(Bucket=BUCKET_NAME)['Contents']]
        self.assertEqual(bucket_content_2, ['file.txt', 'some_folder/file.txt'])

    @mock_s3
    def test_data_upload_content(self):
        s3 = boto3.resource('s3', region_name='us-east-1')
        s3.create_bucket(Bucket=BUCKET_NAME)

        data_saver = BaseDataSaver(bucket_name=BUCKET_NAME)
        data_saver.save_file(os.path.join(THIS_DIR, 'resources/file.txt'), 'file.txt')

        downloaded_file_path = os.path.join(THIS_DIR, 'downloaded_file.txt')
        s3.Bucket(BUCKET_NAME).download_file('file.txt', downloaded_file_path)

        with open(os.path.join(THIS_DIR, 'resources/file.txt')) as f:
            original_content = f.readlines()

        with open(downloaded_file_path) as f:
            uploaded_content = f.readlines()

        self.assertEqual(original_content, uploaded_content)

        if os.path.exists(downloaded_file_path):
            os.remove(downloaded_file_path)

    @mock_s3
    def test_folder_upload(self):
        s3 = boto3.resource('s3', region_name='us-east-1')
        s3.create_bucket(Bucket=BUCKET_NAME)
        s3_client = boto3.client('s3')

        data_saver = BaseDataSaver(bucket_name=BUCKET_NAME)

        data_saver.save_folder(os.path.join(THIS_DIR, 'resources'), 'resources')
        bucket_content_1 = [el['Key'] for el in s3_client.list_objects(Bucket=BUCKET_NAME)['Contents']]
        self.assertEqual(
            bucket_content_1,
            ['resources/file.txt', 'resources/upload_folder/file_2.txt', 'resources/upload_folder/some_file.txt']
        )

        data_saver.save_folder(os.path.join(THIS_DIR, 'resources'), '')
        bucket_content_2 = [el['Key'] for el in s3_client.list_objects(Bucket=BUCKET_NAME)['Contents']]
        self.assertEqual(
            bucket_content_2,
            ['file.txt', 'resources/file.txt', 'resources/upload_folder/file_2.txt',
             'resources/upload_folder/some_file.txt', 'upload_folder/file_2.txt', 'upload_folder/some_file.txt']
        )


class TestBaseDataLoader(unittest.TestCase):
    @mock_s3
    def test_data_download(self):
        s3 = boto3.resource('s3', region_name='us-east-1')
        s3.create_bucket(Bucket=BUCKET_NAME)
        s3_client = boto3.client('s3')
        s3_client.upload_file(os.path.join(THIS_DIR, 'resources/file.txt'), BUCKET_NAME, 'file.txt')
        s3_client.upload_file(os.path.join(THIS_DIR, 'resources/file.txt'), BUCKET_NAME, 'some_folder/file.txt')

        data_loader = BaseDataLoader(bucket_name=BUCKET_NAME, local_base_data_folder_path=THIS_DIR)

        dl_file_path = os.path.join(THIS_DIR, 'downloaded_file.txt')
        data_loader.load_file('file.txt', dl_file_path)
        dl_some_folder_file_path = os.path.join(THIS_DIR, 'downloaded_some_folder_file.txt')
        data_loader.load_file('some_folder/file.txt', dl_some_folder_file_path)

        with open(os.path.join(THIS_DIR, 'resources/file.txt')) as f:
            original_content = f.readlines()

        with open(dl_file_path) as f:
            downloaded_file = f.readlines()

        with open(dl_some_folder_file_path) as f:
            downloaded_some_folder_file = f.readlines()

        self.assertEqual(original_content, downloaded_file)
        self.assertEqual(downloaded_some_folder_file, downloaded_file)

        if os.path.exists(dl_file_path):
            os.remove(dl_file_path)
        if os.path.exists(dl_some_folder_file_path):
            os.remove(dl_some_folder_file_path)
