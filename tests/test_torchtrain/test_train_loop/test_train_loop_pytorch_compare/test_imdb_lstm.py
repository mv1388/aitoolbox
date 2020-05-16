import unittest

import os
import shutil
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from aitoolbox import TrainLoop, TTModel

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class TestImbdLSTM(unittest.TestCase):
    def test_trainloop_core_pytorch_compare(self):
        pass









    @staticmethod
    def set_seeds():
        manual_seed = 0
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        np.random.seed(manual_seed)
        random.seed(manual_seed)
        torch.manual_seed(manual_seed)
        # if you are suing GPU
        torch.cuda.manual_seed(manual_seed)
        torch.cuda.manual_seed_all(manual_seed)
