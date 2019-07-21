import os
import unittest
import shutil

from tests.utils import *

from AIToolbox.experiment.local_load.local_model_load import PyTorchLocalModelLoader
from AIToolbox.experiment.local_save.local_model_save import PyTorchLocalModelSaver

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class TestPyTorchLocalModelLoader(unittest.TestCase):
    def test_init(self):
        model_loader = PyTorchLocalModelLoader(THIS_DIR)
        self.assertIsNone(model_loader.model_load)

    def test_load_model(self):
        model_checkpoint = self.save_dummy_model()

        model_loader = PyTorchLocalModelLoader(os.path.join(THIS_DIR, 'project', 'exp_12', 'model'))
        model_return = model_loader.load_model('model_exp_12_E3.pth', 'project', 'exp')

        self.assertEqual(sorted(model_checkpoint.keys()), sorted(model_loader.model_load.keys()))
        self.assertEqual(sorted(model_checkpoint.keys()), sorted(model_return.keys()))

        if os.path.exists(os.path.join(THIS_DIR, 'project')):
            shutil.rmtree(os.path.join(THIS_DIR, 'project'))

    def save_dummy_model(self):
        model = Net()

        model_checkpoint = {'model_state_dict': model.state_dict(), 'optimizer_state_dict': None,
                            'epoch': 10, 'args': {}}
        saver = PyTorchLocalModelSaver(local_model_result_folder_path=THIS_DIR)
        paths = saver.save_model(model_checkpoint, 'project', 'exp', '12', 3)

        return model_checkpoint
