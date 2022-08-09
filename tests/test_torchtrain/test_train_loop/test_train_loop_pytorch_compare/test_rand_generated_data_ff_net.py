import unittest

import os
import random
import shutil
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import TensorDataset

import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from aitoolbox.torchtrain.train_loop import TrainLoop
from aitoolbox.torchtrain.model import TTModel
from aitoolbox.torchtrain.schedulers.warmup import LinearWithWarmupScheduler

from tests.ddp_cpu_prediction_saver import DDPCPUPredictionSave

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class FFNetAIToolbox(TTModel):
    def __init__(self):
        super().__init__()
        self.ff_1 = nn.Linear(50, 100)
        self.ff_2 = nn.Linear(100, 100)
        self.ff_3 = nn.Linear(100, 10)

    def forward(self, batch_data):
        ff_out = F.relu(self.ff_1(batch_data))
        ff_out = F.relu(self.ff_2(ff_out))
        ff_out = self.ff_3(ff_out)
        out_softmax = F.log_softmax(ff_out, dim=1)
        return out_softmax

    def get_loss(self, batch_data, criterion, device):
        input_data, target = batch_data
        input_data = input_data.to(device)
        target = target.to(device)

        predicted = self(input_data)
        loss = criterion(predicted, target)

        return loss

    def get_predictions(self, batch_data, device):
        input_data, target = batch_data
        input_data = input_data.to(device)

        predicted = self(input_data).argmax(dim=1, keepdim=False)

        return predicted.cpu(), target, {}


class FFNetPyTorch(nn.Module):
    def __init__(self):
        super().__init__()
        self.ff_1 = nn.Linear(50, 100)
        self.ff_2 = nn.Linear(100, 100)
        self.ff_3 = nn.Linear(100, 10)

    def forward(self, batch_data):
        ff_out = F.relu(self.ff_1(batch_data))
        ff_out = F.relu(self.ff_2(ff_out))
        ff_out = self.ff_3(ff_out)
        out_softmax = F.log_softmax(ff_out, dim=1)
        return out_softmax


class TestTrainLoopVSCorePyTorch(unittest.TestCase):
    def test_trainloop_core_pytorch_prediction_loss_compare(self):
        batch_size = 50
        num_epochs = 10

        self.set_seeds()
        model_aitb = FFNetAIToolbox()
        optimizer_aitb = optim.Adam(model_aitb.parameters(), lr=0.001, betas=(0.9, 0.999))
        criterion_aitb = nn.NLLLoss()

        self.set_seeds()
        model_pt = FFNetPyTorch()
        optimizer_pt = optim.Adam(model_pt.parameters(), lr=0.001, betas=(0.9, 0.999))
        criterion_pt = nn.NLLLoss()

        train_dataset = TensorDataset(torch.randn(1000, 50), torch.randint(low=0, high=10, size=(1000,)))
        val_dataset = TensorDataset(torch.randn(300, 50), torch.randint(low=0, high=10, size=(300,)))
        test_dataset = TensorDataset(torch.randn(300, 50), torch.randint(low=0, high=10, size=(300,)))
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

        train_loop = TrainLoop(
            model_aitb,
            train_dataloader, val_dataloader, test_dataloader,
            optimizer_aitb, criterion_aitb
        )
        train_loop.fit(num_epochs=num_epochs)

        train_pred_aitb, _, _ = train_loop.predict_on_train_set()
        val_pred_aitb, _, _ = train_loop.predict_on_validation_set()
        test_pred_aitb, _, _ = train_loop.predict_on_test_set()
        train_loss_aitb = train_loop.evaluate_loss_on_train_set()
        val_loss_aitb = train_loop.evaluate_loss_on_validation_set()
        test_loss_aitb = train_loop.evaluate_loss_on_test_set()

        model_pt.train()
        for epoch in range(num_epochs):
            for input_data, target in train_dataloader:
                predicted = model_pt(input_data)
                loss = criterion_pt(predicted, target)
                optimizer_pt.zero_grad()
                loss.backward()
                optimizer_pt.step()

        train_pred_pt, train_loss_pt = [], []
        model_pt.eval()
        with torch.no_grad():
            for input_data, target in train_dataloader:
                predicted = model_pt(input_data)
                loss_batch = criterion_pt(predicted, target).item()
                train_pred_pt += predicted.argmax(dim=1, keepdim=False).tolist()
                train_loss_pt.append(loss_batch)
            train_loss_pt = np.mean(train_loss_pt)

        val_pred_pt, val_loss_pt = [], []
        model_pt.eval()
        with torch.no_grad():
            for input_data, target in val_dataloader:
                predicted = model_pt(input_data)
                loss_batch = criterion_pt(predicted, target).item()
                val_pred_pt += predicted.argmax(dim=1, keepdim=False).tolist()
                val_loss_pt.append(loss_batch)
            val_loss_pt = np.mean(val_loss_pt)

        test_pred_pt, test_loss_pt = [], []
        model_pt.eval()
        with torch.no_grad():
            for input_data, target in test_dataloader:
                predicted = model_pt(input_data)
                loss_batch = criterion_pt(predicted, target).item()
                test_pred_pt += predicted.argmax(dim=1, keepdim=False).tolist()
                test_loss_pt.append(loss_batch)
            test_loss_pt = np.mean(test_loss_pt)

        self.assertEqual(train_pred_aitb.tolist(), train_pred_pt)
        self.assertEqual(val_pred_aitb.tolist(), val_pred_pt)
        self.assertEqual(test_pred_aitb.tolist(), test_pred_pt)

        self.assertEqual(train_loss_aitb, train_loss_pt)
        self.assertEqual(val_loss_aitb, val_loss_pt)
        self.assertEqual(test_loss_aitb, test_loss_pt)

    def test_training_iteration_trainloop_core_pytorch_prediction_loss_compare(self):
        num_epochs = 10
        batch_size = 10

        self.set_seeds()
        model_aitb = FFNetAIToolbox()
        optimizer_aitb = optim.Adam(model_aitb.parameters(), lr=0.001, betas=(0.9, 0.999))
        criterion_aitb = nn.NLLLoss()

        self.set_seeds()
        model_pt = FFNetPyTorch()
        optimizer_pt = optim.Adam(model_pt.parameters(), lr=0.001, betas=(0.9, 0.999))
        criterion_pt = nn.NLLLoss()

        train_dataset = TensorDataset(torch.randn(1000, 50), torch.randint(low=0, high=10, size=(1000,)))
        val_dataset = TensorDataset(torch.randn(300, 50), torch.randint(low=0, high=10, size=(300,)))
        test_dataset = TensorDataset(torch.randn(300, 50), torch.randint(low=0, high=10, size=(300,)))
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

        train_loop = TrainLoop(
            model_aitb,
            train_dataloader, val_dataloader, test_dataloader,
            optimizer_aitb, criterion_aitb
        )
        train_loop.fit(num_iterations=num_epochs * len(train_dataloader))

        self.assertEqual(train_loop.total_iteration_idx, (num_epochs * len(train_dataloader)) - 1)

        train_pred_aitb, _, _ = train_loop.predict_on_train_set()
        val_pred_aitb, _, _ = train_loop.predict_on_validation_set()
        test_pred_aitb, _, _ = train_loop.predict_on_test_set()
        train_loss_aitb = train_loop.evaluate_loss_on_train_set()
        val_loss_aitb = train_loop.evaluate_loss_on_validation_set()
        test_loss_aitb = train_loop.evaluate_loss_on_test_set()

        model_pt.train()
        iteration_ctr = 0
        while iteration_ctr < num_epochs * len(train_dataloader):
            print('Epoch')
            for i, (input_data, target) in enumerate(train_dataloader):
                predicted = model_pt(input_data)
                loss = criterion_pt(predicted, target)
                loss.backward()
                optimizer_pt.step()
                optimizer_pt.zero_grad()

                iteration_ctr += 1

        self.assertEqual(iteration_ctr, 1000)
        self.assertEqual(iteration_ctr, num_epochs * len(train_dataloader))

        self.assertEqual(train_loop.total_iteration_idx, iteration_ctr - 1)

        train_pred_pt, train_loss_pt = [], []
        model_pt.eval()
        with torch.no_grad():
            for input_data, target in train_dataloader:
                predicted = model_pt(input_data)
                loss_batch = criterion_pt(predicted, target).item()
                train_pred_pt += predicted.argmax(dim=1, keepdim=False).tolist()
                train_loss_pt.append(loss_batch)
            train_loss_pt = np.mean(train_loss_pt)

        val_pred_pt, val_loss_pt = [], []
        model_pt.eval()
        with torch.no_grad():
            for input_data, target in val_dataloader:
                predicted = model_pt(input_data)
                loss_batch = criterion_pt(predicted, target).item()
                val_pred_pt += predicted.argmax(dim=1, keepdim=False).tolist()
                val_loss_pt.append(loss_batch)
            val_loss_pt = np.mean(val_loss_pt)

        test_pred_pt, test_loss_pt = [], []
        model_pt.eval()
        with torch.no_grad():
            for input_data, target in test_dataloader:
                predicted = model_pt(input_data)
                loss_batch = criterion_pt(predicted, target).item()
                test_pred_pt += predicted.argmax(dim=1, keepdim=False).tolist()
                test_loss_pt.append(loss_batch)
            test_loss_pt = np.mean(test_loss_pt)

        self.assertEqual(train_pred_aitb.tolist(), train_pred_pt)
        self.assertEqual(val_pred_aitb.tolist(), val_pred_pt)
        self.assertEqual(test_pred_aitb.tolist(), test_pred_pt)

        self.assertEqual(train_loss_aitb, train_loss_pt)
        self.assertEqual(val_loss_aitb, val_loss_pt)
        self.assertEqual(test_loss_aitb, test_loss_pt)

    def test_grad_accumulate_trainloop_core_pytorch_prediction_loss_compare(self):
        num_epochs = 10

        batch_size = 10
        grad_accumulation = 5

        self.set_seeds()
        model_aitb = FFNetAIToolbox()
        optimizer_aitb = optim.Adam(model_aitb.parameters(), lr=0.001, betas=(0.9, 0.999))
        criterion_aitb = nn.NLLLoss()

        self.set_seeds()
        model_pt = FFNetPyTorch()
        optimizer_pt = optim.Adam(model_pt.parameters(), lr=0.001, betas=(0.9, 0.999))
        criterion_pt = nn.NLLLoss()

        train_dataset = TensorDataset(torch.randn(1000, 50), torch.randint(low=0, high=10, size=(1000,)))
        val_dataset = TensorDataset(torch.randn(300, 50), torch.randint(low=0, high=10, size=(300,)))
        test_dataset = TensorDataset(torch.randn(300, 50), torch.randint(low=0, high=10, size=(300,)))
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

        train_loop = TrainLoop(
            model_aitb,
            train_dataloader, val_dataloader, test_dataloader,
            optimizer_aitb, criterion_aitb
        )
        train_loop.fit(num_epochs=num_epochs, grad_accumulation=grad_accumulation)

        train_pred_aitb, _, _ = train_loop.predict_on_train_set()
        val_pred_aitb, _, _ = train_loop.predict_on_validation_set()
        test_pred_aitb, _, _ = train_loop.predict_on_test_set()
        train_loss_aitb = train_loop.evaluate_loss_on_train_set()
        val_loss_aitb = train_loop.evaluate_loss_on_validation_set()
        test_loss_aitb = train_loop.evaluate_loss_on_test_set()

        model_pt.train()
        for epoch in range(num_epochs):
            for i, (input_data, target) in enumerate(train_dataloader):
                predicted = model_pt(input_data)
                loss = criterion_pt(predicted, target)
                loss = loss / grad_accumulation
                loss.backward()

                if (i+1) % grad_accumulation == 0:
                    optimizer_pt.step()
                    optimizer_pt.zero_grad()

        train_pred_pt, train_loss_pt = [], []
        model_pt.eval()
        with torch.no_grad():
            for input_data, target in train_dataloader:
                predicted = model_pt(input_data)
                loss_batch = criterion_pt(predicted, target).item()
                train_pred_pt += predicted.argmax(dim=1, keepdim=False).tolist()
                train_loss_pt.append(loss_batch)
            train_loss_pt = np.mean(train_loss_pt)

        val_pred_pt, val_loss_pt = [], []
        model_pt.eval()
        with torch.no_grad():
            for input_data, target in val_dataloader:
                predicted = model_pt(input_data)
                loss_batch = criterion_pt(predicted, target).item()
                val_pred_pt += predicted.argmax(dim=1, keepdim=False).tolist()
                val_loss_pt.append(loss_batch)
            val_loss_pt = np.mean(val_loss_pt)

        test_pred_pt, test_loss_pt = [], []
        model_pt.eval()
        with torch.no_grad():
            for input_data, target in test_dataloader:
                predicted = model_pt(input_data)
                loss_batch = criterion_pt(predicted, target).item()
                test_pred_pt += predicted.argmax(dim=1, keepdim=False).tolist()
                test_loss_pt.append(loss_batch)
            test_loss_pt = np.mean(test_loss_pt)

        self.assertEqual(train_pred_aitb.tolist(), train_pred_pt)
        self.assertEqual(val_pred_aitb.tolist(), val_pred_pt)
        self.assertEqual(test_pred_aitb.tolist(), test_pred_pt)

        self.assertEqual(train_loss_aitb, train_loss_pt)
        self.assertEqual(val_loss_aitb, val_loss_pt)
        self.assertEqual(test_loss_aitb, test_loss_pt)

    def test_scheduler_trainloop_core_pytorch_compare(self):
        batch_size = 50
        num_epochs = 10

        self.set_seeds()
        model_aitb = FFNetAIToolbox()
        optimizer_aitb = optim.Adam(model_aitb.parameters(), lr=0.001, betas=(0.9, 0.999))
        criterion_aitb = nn.NLLLoss()

        self.set_seeds()
        model_pt = FFNetPyTorch()
        optimizer_pt = optim.Adam(model_pt.parameters(), lr=0.001, betas=(0.9, 0.999))
        criterion_pt = nn.NLLLoss()

        train_dataset = TensorDataset(torch.randn(1000, 50), torch.randint(low=0, high=10, size=(1000,)))
        val_dataset = TensorDataset(torch.randn(300, 50), torch.randint(low=0, high=10, size=(300,)))
        test_dataset = TensorDataset(torch.randn(300, 50), torch.randint(low=0, high=10, size=(300,)))
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

        num_warmup_steps = 5
        num_training_steps = len(train_dataloader) * num_epochs

        scheduler_aitb = LinearWithWarmupScheduler(num_warmup_steps=num_warmup_steps,
                                                   num_training_steps=num_training_steps)

        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(
                0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
            )
        scheduler_pt = optim.lr_scheduler.LambdaLR(optimizer_pt, lr_lambda)

        train_loop = TrainLoop(
            model_aitb,
            train_dataloader, val_dataloader, test_dataloader,
            optimizer_aitb, criterion_aitb
        )
        train_loop.fit(num_epochs=num_epochs, callbacks=[scheduler_aitb])

        train_pred_aitb, _, _ = train_loop.predict_on_train_set()
        val_pred_aitb, _, _ = train_loop.predict_on_validation_set()
        test_pred_aitb, _, _ = train_loop.predict_on_test_set()
        train_loss_aitb = train_loop.evaluate_loss_on_train_set()
        val_loss_aitb = train_loop.evaluate_loss_on_validation_set()
        test_loss_aitb = train_loop.evaluate_loss_on_test_set()

        model_pt.train()
        for epoch in range(num_epochs):
            for input_data, target in train_dataloader:
                predicted = model_pt(input_data)
                loss = criterion_pt(predicted, target)
                optimizer_pt.zero_grad()
                loss.backward()
                optimizer_pt.step()
                scheduler_pt.step()

        train_pred_pt, train_loss_pt = [], []
        model_pt.eval()
        with torch.no_grad():
            for input_data, target in train_dataloader:
                predicted = model_pt(input_data)
                loss_batch = criterion_pt(predicted, target).item()
                train_pred_pt += predicted.argmax(dim=1, keepdim=False).tolist()
                train_loss_pt.append(loss_batch)
            train_loss_pt = np.mean(train_loss_pt)

        val_pred_pt, val_loss_pt = [], []
        model_pt.eval()
        with torch.no_grad():
            for input_data, target in val_dataloader:
                predicted = model_pt(input_data)
                loss_batch = criterion_pt(predicted, target).item()
                val_pred_pt += predicted.argmax(dim=1, keepdim=False).tolist()
                val_loss_pt.append(loss_batch)
            val_loss_pt = np.mean(val_loss_pt)

        test_pred_pt, test_loss_pt = [], []
        model_pt.eval()
        with torch.no_grad():
            for input_data, target in test_dataloader:
                predicted = model_pt(input_data)
                loss_batch = criterion_pt(predicted, target).item()
                test_pred_pt += predicted.argmax(dim=1, keepdim=False).tolist()
                test_loss_pt.append(loss_batch)
            test_loss_pt = np.mean(test_loss_pt)

        self.assertEqual(train_pred_aitb.tolist(), train_pred_pt)
        self.assertEqual(val_pred_aitb.tolist(), val_pred_pt)
        self.assertEqual(test_pred_aitb.tolist(), test_pred_pt)

        self.assertEqual(train_loss_aitb, train_loss_pt)
        self.assertEqual(val_loss_aitb, val_loss_pt)
        self.assertEqual(test_loss_aitb, test_loss_pt)

    @staticmethod
    def set_seeds():
        manual_seed = 0
        np.random.seed(manual_seed)
        random.seed(manual_seed)
        torch.manual_seed(manual_seed)
        # if you are suing GPU
        torch.cuda.manual_seed(manual_seed)
        torch.cuda.manual_seed_all(manual_seed)

        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


class TestDDPCPUTrainLoopVSCorePyTorch(unittest.TestCase):
    def test_ddp_cpu_trainloop_core_pytorch_prediction_loss_compare(self):
        os.mkdir(f'{THIS_DIR}/ddp_temp_save')

        batch_size = 50
        num_epochs = 10

        train_pred_aitb, val_pred_aitb, test_pred_aitb, \
            train_true_aitb, val_true_aitb, test_true_aitb, \
            train_loss_aitb, val_loss_aitb, test_loss_aitb = \
            self.train_eval_trainloop(batch_size, num_epochs)

        train_pred_pt, val_pred_pt, test_pred_pt, \
            train_true_pt, val_true_pt, test_true_pt, \
            train_loss_pt, val_loss_pt, test_loss_pt = \
            self.train_eval_core_pytorch(batch_size, num_epochs)

        print('Doing comparison test')

        self.assertEqual(train_true_aitb, train_true_pt)
        self.assertEqual(val_true_aitb, val_true_pt)
        self.assertEqual(test_true_aitb, test_true_pt)

        self.assertEqual(train_pred_aitb, train_pred_pt)
        self.assertEqual(val_pred_aitb, val_pred_pt)
        self.assertEqual(test_pred_aitb, test_pred_pt)

        self.assertEqual(train_loss_aitb, train_loss_pt)
        self.assertEqual(val_loss_aitb, val_loss_pt)
        self.assertEqual(test_loss_aitb, test_loss_pt)

        project_path = os.path.join(THIS_DIR, 'ddp_temp_save')
        if os.path.exists(project_path):
            shutil.rmtree(project_path)

    def train_eval_trainloop(self, batch_size, num_epochs=10):
        self.set_seeds()

        train_dataset = TensorDataset(torch.randn(1000, 50), torch.randint(low=0, high=10, size=(1000,)))
        val_dataset = TensorDataset(torch.randn(300, 50), torch.randint(low=0, high=10, size=(300,)))
        test_dataset = TensorDataset(torch.randn(300, 50), torch.randint(low=0, high=10, size=(300,)))
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

        model_aitb = FFNetAIToolbox()
        optimizer_aitb = optim.Adam(model_aitb.parameters(), lr=0.001, betas=(0.9, 0.999))
        criterion_aitb = nn.NLLLoss()

        train_loop = TrainLoop(
            model_aitb,
            train_dataloader, val_dataloader, test_dataloader,
            optimizer_aitb, criterion_aitb,
            gpu_mode='ddp'
        )
        train_loop.fit(
            num_epochs=num_epochs,
            backend='gloo', num_gpus=2, on_gpu=False,
            callbacks=[
                DDPCPUPredictionSave(dir_path=f'{THIS_DIR}/ddp_temp_save',
                                     file_name='tl_ddp_predictions.p')
            ]
        )

        with open(f'{THIS_DIR}/ddp_temp_save/tl_ddp_predictions.p', 'rb') as f:
            train_pred_aitb, val_pred_aitb, test_pred_aitb, \
                train_true_aitb, val_true_aitb, test_true_aitb, \
                train_loss_aitb, val_loss_aitb, test_loss_aitb = \
                pickle.load(f)

        return train_pred_aitb, val_pred_aitb, test_pred_aitb, \
            train_true_aitb, val_true_aitb, test_true_aitb, \
            train_loss_aitb, val_loss_aitb, test_loss_aitb

    def train_eval_core_pytorch(self, batch_size, num_epochs=10):
        self.set_seeds()

        train_dataset = TensorDataset(torch.randn(1000, 50), torch.randint(low=0, high=10, size=(1000,)))
        val_dataset = TensorDataset(torch.randn(300, 50), torch.randint(low=0, high=10, size=(300,)))
        test_dataset = TensorDataset(torch.randn(300, 50), torch.randint(low=0, high=10, size=(300,)))
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

        nprocs = 2

        model_pt = FFNetPyTorch()
        optimizer_pt = optim.Adam(model_pt.parameters(), lr=0.001, betas=(0.9, 0.999))
        criterion_pt = nn.NLLLoss()

        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '8888'
        # Based on:
        # https://blog.exxactcorp.com/pytorch-1-5-1-bug-fix-release/
        # https://github.com/pytorch/pytorch/issues/37377
        os.environ['MKL_THREADING_LAYER'] = 'GNU'

        print('Starting the manual DDP training')

        mp.spawn(
            self.manual_ddp_training,
            args=(nprocs, num_epochs, model_pt, optimizer_pt, criterion_pt, train_dataloader, val_dataloader, test_dataloader),
            nprocs=nprocs
        )

        train_pred_pt, val_pred_pt, test_pred_pt, \
            train_true_pt, val_true_pt, test_true_pt, \
            train_loss_pt, val_loss_pt, test_loss_pt = [], [], [], [], [], [], [], [], []

        for idx in range(nprocs):
            with open(f'{THIS_DIR}/ddp_temp_save/pt_ddp_predictions_{idx}.p', 'rb') as f:
                train_pred_f, val_pred_f, test_pred_f, \
                    train_true_f, val_true_f, test_true_f, \
                    train_loss_f, val_loss_f, test_loss_f = pickle.load(f)
                train_pred_pt += train_pred_f
                val_pred_pt += val_pred_f
                test_pred_pt += test_pred_f
                train_true_pt += train_true_f
                val_true_pt += val_true_f
                test_true_pt += test_true_f
                train_loss_pt += train_loss_f
                val_loss_pt += val_loss_f
                test_loss_pt += test_loss_f

        train_loss_pt = np.mean(train_loss_pt)
        val_loss_pt = np.mean(val_loss_pt)
        test_loss_pt = np.mean(test_loss_pt)

        return train_pred_pt, val_pred_pt, test_pred_pt, \
            train_true_pt, val_true_pt, test_true_pt, \
            train_loss_pt, val_loss_pt, test_loss_pt

    @staticmethod
    def manual_ddp_training(
            gpu, nprocs,
            num_epochs,
            model_pt, optimizer_pt, criterion_pt,
            train_loader, val_loader, test_loader
    ):
        rank = gpu
        print(rank)
        print(f'nprocs: {nprocs}')
        dist.init_process_group(backend='gloo', init_method='env://', world_size=nprocs, rank=rank)
        torch.manual_seed(0)
        device = torch.device('cpu')

        train_sampler = DistributedSampler(dataset=train_loader.dataset, shuffle=False,
                                           num_replicas=nprocs, rank=rank)
        val_sampler = DistributedSampler(dataset=val_loader.dataset, shuffle=False,
                                         num_replicas=nprocs, rank=rank)
        test_sampler = DistributedSampler(dataset=test_loader.dataset, shuffle=False,
                                          num_replicas=nprocs, rank=rank)
        train_loader_ddp = DataLoader(train_loader.dataset, batch_size=50, sampler=train_sampler)
        val_loader_ddp = DataLoader(val_loader.dataset, batch_size=50, sampler=val_sampler)
        test_loader_ddp = DataLoader(test_loader.dataset, batch_size=50, sampler=test_sampler)

        model_pt = model_pt.to(device)
        criterion_pt = criterion_pt.to(device)

        model_pt = DistributedDataParallel(model_pt)

        print('Training')
        model_pt.train()
        for epoch in range(num_epochs):
            print(f'Epoch: {epoch}')
            train_sampler.set_epoch(epoch)

            for i, (input_data, target) in enumerate(train_loader_ddp):
                input_data = input_data.to(device)
                target = target.to(device)

                predicted = model_pt(input_data)
                loss = criterion_pt(predicted, target)
                loss.backward()
                optimizer_pt.step()
                optimizer_pt.zero_grad()

            # Imitate what happens in auto_execute_end_of_epoch() in TrainLoop
            for _ in train_loader_ddp:
                pass
            for _ in val_loader_ddp:
                pass

        train_pred_pt, val_pred_pt, test_pred_pt, \
            train_true_pt, val_true_pt, test_true_pt, \
            train_loss_pt, val_loss_pt, test_loss_pt = [], [], [], [], [], [], [], [], []

        print('Evaluating - train')
        model_pt.eval()
        with torch.no_grad():
            for input_data, target in train_loader_ddp:
                input_data = input_data.to(device)
                target = target.to(device)

                predicted = model_pt(input_data)
                loss_batch = criterion_pt(predicted, target).cpu().item()

                train_pred_pt += predicted.argmax(dim=1, keepdim=False).cpu().tolist()
                train_true_pt += target.cpu().tolist()
                train_loss_pt.append(loss_batch)

        print('Evaluating - validation')
        model_pt.eval()
        with torch.no_grad():
            for input_data, target in val_loader_ddp:
                input_data = input_data.to(device)
                target = target.to(device)

                predicted = model_pt(input_data)
                loss_batch = criterion_pt(predicted, target).cpu().item()

                val_pred_pt += predicted.argmax(dim=1, keepdim=False).cpu().tolist()
                val_true_pt += target.cpu().tolist()
                val_loss_pt.append(loss_batch)

        print('Evaluating - test')
        model_pt.eval()
        with torch.no_grad():
            for input_data, target in test_loader_ddp:
                input_data = input_data.to(device)
                target = target.to(device)

                predicted = model_pt(input_data)
                loss_batch = criterion_pt(predicted, target).cpu().item()

                test_pred_pt += predicted.argmax(dim=1, keepdim=False).cpu().tolist()
                test_true_pt += target.cpu().tolist()
                test_loss_pt.append(loss_batch)

        with open(f'{THIS_DIR}/ddp_temp_save/pt_ddp_predictions_{gpu}.p', 'wb') as f:
            pickle.dump([
                train_pred_pt, val_pred_pt, test_pred_pt,
                train_true_pt, val_true_pt, test_true_pt,
                train_loss_pt, val_loss_pt, test_loss_pt
            ], f)

    @staticmethod
    def set_seeds():
        manual_seed = 0
        np.random.seed(manual_seed)
        random.seed(manual_seed)
        torch.manual_seed(manual_seed)
        # if you are suing GPU
        torch.cuda.manual_seed(manual_seed)
        torch.cuda.manual_seed_all(manual_seed)

        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        torch.backends.enabled = False
        torch.backends.benchmark = False
        torch.backends.deterministic = True
