import unittest

import os
import shutil
import random
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from aitoolbox import TrainLoopCheckpointEndSave, TTModel, ModelPerformanceEvaluation, ModelPerformancePrintReport, \
    ModelTrainHistoryPlot, ModelTrainHistoryFileWriter, ClassificationResultPackage
from tests_gpu.test_multi_gpu.ddp_utils import DDPPredictionSave, SetSeedInTrainLoop

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class CNNNet(TTModel):
    def __init__(self):
        super(CNNNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

    def get_loss(self, batch_data, criterion, device):
        data, target = batch_data
        data, target = data.to(device), target.to(device)

        output = self(data)
        loss = criterion(output, target)

        return loss

    def get_predictions(self, batch_data, device):
        data, y_test = batch_data
        data = data.to(device)

        output = self(data)
        y_pred = output.argmax(dim=1, keepdim=False)

        return y_pred.cpu(), y_test, {}


class TestMNISTCNN(unittest.TestCase):
    def test_trainloop_core_pytorch_compare(self):
        os.mkdir(f'{THIS_DIR}/ddp_cnn_save')

        val_loss_tl, y_pred_tl, y_true_tl = self.train_eval_trainloop(num_epochs=5, use_real_train_data=True)
        val_loss_pt, y_pred_pt, y_true_pt = self.train_eval_core_pytorch(num_epochs=5, use_real_train_data=True)

        self.assertEqual(val_loss_tl, val_loss_pt)
        self.assertEqual(y_pred_tl, y_pred_pt)
        self.assertEqual(y_true_tl, y_true_pt)

        project_path = os.path.join(THIS_DIR, 'ddp_cnn_save')
        if os.path.exists(project_path):
            shutil.rmtree(project_path)
        project_path = os.path.join(THIS_DIR, 'data')
        if os.path.exists(project_path):
            shutil.rmtree(project_path)
        project_path = os.path.join(THIS_DIR, 'tl_full_experiment_tracking')
        if os.path.exists(project_path):
            shutil.rmtree(project_path)

    def train_eval_trainloop(self, num_epochs, use_real_train_data=False):
        self.set_seeds()
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(os.path.join(THIS_DIR, 'data'), train=use_real_train_data, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=100, shuffle=True)
        val_loader = torch.utils.data.DataLoader(
            datasets.MNIST(os.path.join(THIS_DIR, 'data'), train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])),
            batch_size=100)

        model = CNNNet()
        # TODO: There is currently a bug in PyTorch 1.12 Adam... replacing temporarily
        # optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
        optimizer = optim.Adadelta(model.parameters(), lr=0.001)
        criterion = nn.NLLLoss()

        callbacks = [
            ModelPerformanceEvaluation(ClassificationResultPackage(), {},
                                       on_train_data=True, on_val_data=True),
            ModelPerformancePrintReport(['train_Accuracy', 'val_Accuracy']),
            ModelTrainHistoryPlot(),
            ModelTrainHistoryFileWriter(),
            DDPPredictionSave(dir_path=f'{THIS_DIR}/ddp_cnn_save',
                              file_name='tl_ddp_predictions.p'),
            SetSeedInTrainLoop()
        ]

        print('Starting train loop')
        tl = TrainLoopCheckpointEndSave(
            model,
            train_loader, val_loader, None,
            optimizer, criterion,
            project_name='tl_full_experiment_tracking', experiment_name='tutorial_example',
            local_model_result_folder_path=THIS_DIR,
            hyperparams={},
            val_result_package=ClassificationResultPackage(),
            cloud_save_mode=None,
            gpu_mode='ddp'
        )
        self.assertEqual(tl.device.type, "cuda")

        tl.fit(num_epochs=num_epochs, callbacks=callbacks)

        with open(f'{THIS_DIR}/ddp_cnn_save/tl_ddp_predictions.p', 'rb') as f:
            val_loss, y_pred, y_true = pickle.load(f)

        return val_loss, y_pred, y_true

    def train_eval_core_pytorch(self, num_epochs, use_real_train_data=False):
        self.set_seeds()
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(os.path.join(THIS_DIR, 'data'), train=use_real_train_data, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=100)
        val_loader = torch.utils.data.DataLoader(
            datasets.MNIST(os.path.join(THIS_DIR, 'data'), train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])),
            batch_size=100)

        model_pt = CNNNet()
        # TODO: There is currently a bug in PyTorch 1.12 Adam... replacing temporarily
        # optimizer_pt = optim.Adam(model_pt.parameters(), lr=0.001, betas=(0.9, 0.999))
        optimizer_pt = optim.Adadelta(model_pt.parameters(), lr=0.001)
        criterion_pt = nn.NLLLoss()

        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '8888'

        print('Starting the manual DDP training')

        mp.spawn(
            self.manual_ddp_training,
            args=(num_epochs, model_pt, optimizer_pt, criterion_pt, train_loader, val_loader),
            nprocs=torch.cuda.device_count()
        )

        val_loss, y_pred, y_true = [], [], []
        for idx in range(torch.cuda.device_count()):
            with open(f'{THIS_DIR}/ddp_cnn_save/pt_ddp_predictions_{idx}.p', 'rb') as f:
                val_loss_f, y_pred_f, y_true_f = pickle.load(f)
                val_loss += val_loss_f
                y_pred += y_pred_f
                y_true += y_true_f

        val_loss = np.mean(val_loss)
        return val_loss, y_pred, y_true

    @staticmethod
    def manual_ddp_training(gpu, num_epochs, model_pt, optimizer_pt, criterion_pt, train_loader, val_loader):
        rank = gpu
        dist.init_process_group(backend='nccl', init_method='env://', world_size=torch.cuda.device_count(), rank=rank)
        torch.manual_seed(0)
        torch.cuda.set_device(gpu)
        device = torch.device(f"cuda:{gpu}")

        train_sampler = DistributedSampler(dataset=train_loader.dataset, shuffle=True,
                                           num_replicas=torch.cuda.device_count(), rank=rank)
        val_sampler = DistributedSampler(dataset=val_loader.dataset, shuffle=False,
                                         num_replicas=torch.cuda.device_count(), rank=rank)
        train_loader = DataLoader(train_loader.dataset, batch_size=100, sampler=train_sampler)
        val_loader = DataLoader(val_loader.dataset, batch_size=100, sampler=val_sampler)

        model_pt = model_pt.to(device)
        criterion_pt = criterion_pt.to(device)

        model_pt = DistributedDataParallel(model_pt, device_ids=[gpu])

        TestMNISTCNN.set_seeds()

        model_pt.train()
        for epoch in range(num_epochs):
            print(f'Epoch: {epoch}')
            train_sampler.set_epoch(epoch)

            for i, (input_data, target) in enumerate(train_loader):
                input_data = input_data.to(device)
                target = target.to(device)

                predicted = model_pt(input_data)
                loss = criterion_pt(predicted, target)
                loss.backward()
                optimizer_pt.step()
                optimizer_pt.zero_grad()

            # Imitate what happens in auto_execute_end_of_epoch() in TrainLoop
            for _ in train_loader:
                pass
            for _ in val_loader:
                pass

            for _ in train_loader:
                pass
            for _ in val_loader:
                pass

        for _ in val_loader:
            pass

        print('Evaluating')
        val_loss, val_pred, val_true = [], [], []
        model_pt.eval()
        with torch.no_grad():
            for input_data, target in val_loader:
                input_data = input_data.to(device)
                target = target.to(device)

                predicted = model_pt(input_data)
                loss_batch = criterion_pt(predicted, target).cpu().item()
                val_pred += predicted.argmax(dim=1, keepdim=False).cpu().tolist()
                val_true += target.cpu().tolist()
                val_loss.append(loss_batch)

        with open(f'{THIS_DIR}/ddp_cnn_save/pt_ddp_predictions_{gpu}.p', 'wb') as f:
            pickle.dump([val_loss, val_pred, val_true], f)

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
