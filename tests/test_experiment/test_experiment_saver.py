import unittest
import os

from tests.utils import *

from aitoolbox.experiment.experiment_saver import *
from aitoolbox.cloud.AWS.model_save import PyTorchS3ModelSaver, KerasS3ModelSaver
from aitoolbox.cloud.GoogleCloud.model_save import KerasGoogleStorageModelSaver, PyTorchGoogleStorageModelSaver
from aitoolbox.cloud.AWS.results_save import S3ResultsSaver
from aitoolbox.cloud.GoogleCloud.results_save import GoogleStorageResultsSaver


THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class TestFullKerasExperimentS3Saver(unittest.TestCase):
    def test_init(self):
        project_dir_name = 'projectKerasLocalModelSaver'
        exp_dir_name = 'experimentSubDirPT'

        saver = FullKerasExperimentS3Saver(project_name=project_dir_name, experiment_name=exp_dir_name,
                                           bucket_name='model-result', local_model_result_folder_path=THIS_DIR)

        self.assertIsInstance(saver, BaseFullExperimentSaver)
        self.assertEqual(type(saver.model_saver), KerasS3ModelSaver)
        self.assertEqual(type(saver.results_saver), S3ResultsSaver)
        self.assertEqual(saver.project_name, project_dir_name)
        self.assertEqual(saver.experiment_name, exp_dir_name)


class TestFullPyTorchExperimentS3Saver(unittest.TestCase):
    def test_init(self):
        project_dir_name = 'projectPyTorchLocalModelSaver'
        exp_dir_name = 'experimentSubDirPT'

        saver = FullPyTorchExperimentS3Saver(project_name=project_dir_name, experiment_name=exp_dir_name,
                                             bucket_name='model-result', local_model_result_folder_path=THIS_DIR)

        self.assertIsInstance(saver, BaseFullExperimentSaver)
        self.assertEqual(type(saver.model_saver), PyTorchS3ModelSaver)
        self.assertEqual(type(saver.results_saver), S3ResultsSaver)
        self.assertEqual(saver.project_name, project_dir_name)
        self.assertEqual(saver.experiment_name, exp_dir_name)


# class TestFullKerasExperimentGoogleStorageSaver(unittest.TestCase):
#     def test_init(self):
#         project_dir_name = 'projectKerasLocalModelSaver'
#         exp_dir_name = 'experimentSubDirPT'
#
#         saver = FullKerasExperimentGoogleStorageSaver(project_name=project_dir_name, experiment_name=exp_dir_name,
#                                                       bucket_name='model-result', local_model_result_folder_path=THIS_DIR)
#
#         self.assertIsInstance(saver, BaseFullExperimentSaver)
#         self.assertEqual(type(saver.model_saver), KerasGoogleStorageModelSaver)
#         self.assertEqual(type(saver.results_saver), GoogleStorageResultsSaver)
#         self.assertEqual(saver.project_name, project_dir_name)
#         self.assertEqual(saver.experiment_name, exp_dir_name)
#
#
# class TestFullPyTorchExperimentGoogleStorageSaver(unittest.TestCase):
#     def test_init(self):
#         project_dir_name = 'projectPyTorchLocalModelSaver'
#         exp_dir_name = 'experimentSubDirPT'
#
#         saver = FullPyTorchExperimentGoogleStorageSaver(project_name=project_dir_name, experiment_name=exp_dir_name,
#                                                         bucket_name='model-result', local_model_result_folder_path=THIS_DIR)
#
#         self.assertIsInstance(saver, BaseFullExperimentSaver)
#         self.assertEqual(type(saver.model_saver), PyTorchGoogleStorageModelSaver)
#         self.assertEqual(type(saver.results_saver), GoogleStorageResultsSaver)
#         self.assertEqual(saver.project_name, project_dir_name)
#         self.assertEqual(saver.experiment_name, exp_dir_name)
