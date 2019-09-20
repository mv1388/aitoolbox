import os
import unittest
import shutil
from collections import OrderedDict
import torch.nn as nn

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
                            'epoch': 10, 'hyperparams': {}}
        saver = PyTorchLocalModelSaver(local_model_result_folder_path=THIS_DIR)
        paths = saver.save_model(model_checkpoint, 'project', 'exp', '12', 3)

        return model_checkpoint

    def test_init_data_parallel_model(self):
        state_dict_fixed = self.save_dummy_data_parallel_model()

        model = Net()
        model_loader = PyTorchLocalModelLoader(os.path.join(THIS_DIR, 'project', 'exp_12', 'model'))
        model_snapshot = model_loader.load_model('model_exp_12_E3.pth', 'project', 'exp')
        model_init = model_loader.init_model(model, used_data_parallel=True)

        self.assertEqual(sorted(state_dict_fixed.keys()), sorted(model_init.state_dict().keys()))

        if os.path.exists(os.path.join(THIS_DIR, 'project')):
            shutil.rmtree(os.path.join(THIS_DIR, 'project'))

    def save_dummy_data_parallel_model(self):
        model = Net()
        model = nn.DataParallel(model)

        model_checkpoint = {'model_state_dict': model.state_dict(), 'optimizer_state_dict': None,
                            'epoch': 10, 'hyperparams': {}}
        saver = PyTorchLocalModelSaver(local_model_result_folder_path=THIS_DIR)
        paths = saver.save_model(model_checkpoint, 'project', 'exp', '12', 3)

        state_dict_fixed = OrderedDict()
        for k, v in model_checkpoint['model_state_dict'].items():
            name = k[7:]  # remove `module.`
            state_dict_fixed[name] = v

        return state_dict_fixed
