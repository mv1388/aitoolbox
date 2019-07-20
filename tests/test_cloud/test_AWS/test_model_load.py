import unittest

from AIToolbox.cloud.AWS.model_load import PyTorchS3ModelLoader
from AIToolbox.experiment_save.local_load.local_model_load import AbstractLocalModelLoader, PyTorchLocalModelLoader


class TestPyTorchS3ModelLoader(unittest.TestCase):
    def test_init(self):
        s3_model_loader = PyTorchS3ModelLoader('', '', '', '')

        self.assertEqual(type(s3_model_loader.local_model_loader), PyTorchLocalModelLoader)
        self.assertIsInstance(s3_model_loader.local_model_loader, AbstractLocalModelLoader)
