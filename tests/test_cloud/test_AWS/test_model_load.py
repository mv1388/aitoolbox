import unittest

from aitoolbox.cloud.AWS.model_load import PyTorchS3ModelLoader
from aitoolbox.experiment.local_load.local_model_load import AbstractLocalModelLoader, PyTorchLocalModelLoader


class TestPyTorchS3ModelLoader(unittest.TestCase):
    def test_init(self):
        s3_model_loader = PyTorchS3ModelLoader('', '', '')

        self.assertEqual(type(s3_model_loader.local_model_loader), PyTorchLocalModelLoader)
        self.assertIsInstance(s3_model_loader.local_model_loader, AbstractLocalModelLoader)
