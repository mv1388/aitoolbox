import unittest

import os
import boto3
from moto import mock_s3

from aitoolbox.cloud.AWS.data_access import BaseDataSaver

os.environ['AWS_ACCESS_KEY_ID'] = "testing"
os.environ['AWS_SECRET_ACCESS_KEY'] = "testing"
os.environ['AWS_SECURITY_TOKEN'] = "testing"
os.environ['AWS_SESSION_TOKEN'] = "testing"

BUCKET_NAME = 'test-bucket'
THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class TestBaseDataSaver(unittest.TestCase):
    @mock_s3
    def test_data_upload(self):
        s3 = boto3.resource('s3')
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
        s3 = boto3.resource('s3')
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
