import unittest

import os
import shutil
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from aitoolbox import TrainLoopCheckpointEndSave, TTModel, ModelPerformanceEvaluation, ModelPerformancePrintReport, \
    ModelTrainHistoryPlot, ModelTrainHistoryFileWriter, ClassificationResultPackage

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


class TestMNISTCNNExperimentTrack(unittest.TestCase):
    def test_trainloop_core_pytorch_compare(self):
        val_loss_tl, y_pred_tl, y_true_tl = self.train_eval_trainloop(num_epochs=5, use_real_train_data=True)
        val_loss_pt, y_pred_pt, y_true_pt = self.train_eval_core_pytorch(num_epochs=5, use_real_train_data=True)

        self.assertEqual(val_loss_tl, val_loss_pt)
        self.assertEqual(y_pred_tl, y_pred_pt)
        self.assertEqual(y_true_tl, y_true_pt)

        val_dataset = datasets.MNIST(
            os.path.join(THIS_DIR, 'data'), train=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        )
        self.assertEqual(val_dataset.targets.tolist(), y_true_tl)
        self.assertEqual(val_dataset.targets.tolist(), y_true_pt)

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
        optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
        criterion = nn.NLLLoss()

        callbacks = [
            ModelPerformanceEvaluation(ClassificationResultPackage(), {},
                                       on_train_data=True, on_val_data=True),
            ModelPerformancePrintReport(['train_Accuracy', 'val_Accuracy']),
            ModelTrainHistoryPlot(),
            ModelTrainHistoryFileWriter()
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
            cloud_save_mode=None
        )

        self.assertEqual(tl.device.type, "cuda")

        tl.fit(num_epochs=num_epochs, callbacks=callbacks)

        val_loss = tl.evaluate_loss_on_validation_set(force_prediction=True)
        y_pred, y_true, _ = tl.predict_on_validation_set(force_prediction=True)

        return val_loss, y_pred.tolist(), y_true.tolist()

    def train_eval_core_pytorch(self, num_epochs, use_real_train_data=False):
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

        USE_CUDA = torch.cuda.is_available()
        device = torch.device("cuda" if USE_CUDA else "cpu")
        self.assertEqual(device.type, "cuda")

        model_pt = CNNNet().to(device)
        optimizer_pt = optim.Adam(model_pt.parameters(), lr=0.001, betas=(0.9, 0.999))
        criterion_pt = nn.NLLLoss()

        model_pt.train()
        for epoch in range(num_epochs):
            print(f'Epoch: {epoch}')
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
            val_loss = np.mean(val_loss)

        return val_loss, val_pred, val_true

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
