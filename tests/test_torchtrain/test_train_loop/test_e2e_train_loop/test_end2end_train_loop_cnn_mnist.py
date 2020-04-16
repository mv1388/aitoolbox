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
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets, transforms

from aitoolbox.torchtrain.train_loop import TrainLoopCheckpointEndSave
from aitoolbox.torchtrain.model import TTModel
from aitoolbox.experiment.result_package.basic_packages import ClassificationResultPackage

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
THIS_FILE = os.path.basename(__file__)


class CnnNet(TTModel):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

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

        return y_pred.cpu(), y_test, {'example_target': y_test.tolist()}


class TestCNNMnistEnd2EndTrainLoopCheckpointEndSave(unittest.TestCase):
    def test_e2e_ff_net_train_loop(self):
        self.set_seeds()

        train_loader = DataLoader(
            datasets.MNIST(
                f'{THIS_DIR}/mnist_data', train=False, download=True,  # set to train=False so that test is done faster
                transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
            ),
            batch_size=100)
        val_loader = DataLoader(
            datasets.MNIST(
                f'{THIS_DIR}/mnist_data', train=False,
                transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
            ),
            batch_size=100)
        test_loader = DataLoader(
            datasets.MNIST(
                f'{THIS_DIR}/mnist_data', train=False,
                transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
            ),
            batch_size=100)

        model = CnnNet()
        optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
        criterion = nn.NLLLoss()

        train_loop = TrainLoopCheckpointEndSave(
            model,
            train_loader, val_loader, test_loader,
            optimizer, criterion,
            project_name='e2e_train_loop_example', experiment_name='TrainLoopCheckpointEndSave_cnn_mnist_example',
            local_model_result_folder_path=THIS_DIR,
            hyperparams={'batch_size': 100, 'num_epochs': 3},
            val_result_package=ClassificationResultPackage(), test_result_package=ClassificationResultPackage(),
            cloud_save_mode=None
        )
        train_loop.fit(num_epochs=3)
        tl_history = train_loop.train_history.train_history

        self.assertEqual(train_loop.epoch, 2)

        result_approx = {
            'loss': [0.2760361768677831, 0.10083887679851614, 0.04546945019857958],
            'accumulated_loss': [0.44150516266934575, 0.1245885466481559, 0.07134055544040166],
            'val_loss': [0.2760361768677831, 0.10083887679851614, 0.04546945019857958],
            'train_end_test_loss': [0.04546945019857958]
        }

        self.assertEqual(sorted(tl_history.keys()), sorted(result_approx.keys()))

        for metric, results_list in result_approx.items():
            self.assertEqual(len(results_list), len(tl_history[metric]))

            for correct_result, tl_result in zip(results_list, tl_history[metric]):
                self.assertAlmostEqual(correct_result, tl_result, places=6)

        self.assertAlmostEqual(train_loop.evaluate_loss_on_train_set(), 0.04546945019857958, places=6)
        self.assertAlmostEqual(train_loop.evaluate_loss_on_validation_set(), 0.04546945019857958, places=6)
        self.assertAlmostEqual(train_loop.evaluate_loss_on_test_set(), 0.04546945019857958, places=6)

        train_pred, train_target, train_meta = train_loop.predict_on_train_set()
        with open(f'{THIS_DIR}/resources/mnist_pred_train_3e.p', 'rb') as f:
            correct_train_pred = pickle.load(f)
            self.assertEqual(train_pred.tolist(), correct_train_pred)
        self.assertEqual(train_target.tolist(), train_loader.dataset.targets.tolist())
        self.assertEqual(train_loader.dataset.targets.tolist(), train_meta['example_target'])

        val_pred, val_target, val_meta = train_loop.predict_on_validation_set()
        with open(f'{THIS_DIR}/resources/mnist_pred_val_3e.p', 'rb') as f:
            correct_val_pred = pickle.load(f)
            self.assertEqual(val_pred.tolist(), correct_val_pred)
        self.assertEqual(val_target.tolist(), val_loader.dataset.targets.tolist())
        self.assertEqual(val_loader.dataset.targets.tolist(), val_meta['example_target'])

        test_pred, test_target, test_meta = train_loop.predict_on_test_set()
        with open(f'{THIS_DIR}/resources/mnist_pred_test_3e.p', 'rb') as f:
            correct_test_pred = pickle.load(f)
            self.assertEqual(test_pred.tolist(), correct_test_pred)
        self.assertEqual(test_target.tolist(), test_loader.dataset.targets.tolist())
        self.assertEqual(test_loader.dataset.targets.tolist(), test_meta['example_target'])

        project_path = os.path.join(THIS_DIR, 'e2e_train_loop_example')
        if os.path.exists(project_path):
            shutil.rmtree(project_path)
        data_path = os.path.join(THIS_DIR, 'mnist_data')
        if os.path.exists(data_path):
            shutil.rmtree(data_path)

    def test_e2e_ff_net_train_loop_tracking_saved_files_check(self):
        self.set_seeds()

        train_loader = DataLoader(
            datasets.MNIST(
                f'{THIS_DIR}/mnist_data', train=False, download=True,  # set to train=False so that test is done faster
                transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
            ),
            batch_size=100)
        val_loader = DataLoader(
            datasets.MNIST(
                f'{THIS_DIR}/mnist_data', train=False,
                transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
            ),
            batch_size=100)
        test_loader = DataLoader(
            datasets.MNIST(
                f'{THIS_DIR}/mnist_data', train=False,
                transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
            ),
            batch_size=100)

        model = CnnNet()
        optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
        criterion = nn.NLLLoss()

        train_loop = TrainLoopCheckpointEndSave(
            model,
            train_loader, val_loader, test_loader,
            optimizer, criterion,
            project_name='e2e_train_loop_example', experiment_name='TrainLoopCheckpointEndSave_cnn_mnist_example',
            local_model_result_folder_path=THIS_DIR,
            hyperparams={'batch_size': 100, 'num_epochs': 3},
            val_result_package=ClassificationResultPackage(), test_result_package=ClassificationResultPackage(),
            cloud_save_mode=None
        )
        train_loop.fit(num_epochs=3)

        experiment_dir_path = os.path.join(THIS_DIR, train_loop.project_name,
                                           f'{train_loop.experiment_name}_{train_loop.experiment_timestamp}')

        self.assertTrue(os.path.exists(os.path.join(experiment_dir_path, 'checkpoint_model')))
        self.assertTrue(os.path.isdir(os.path.join(experiment_dir_path, 'checkpoint_model')))
        self.assertTrue(os.path.exists(os.path.join(experiment_dir_path, 'model')))
        self.assertTrue(os.path.isdir(os.path.join(experiment_dir_path, 'model')))
        self.assertTrue(os.path.exists(os.path.join(experiment_dir_path, 'results')))
        self.assertTrue(os.path.isdir(os.path.join(experiment_dir_path, 'results')))

        self.assertTrue(os.path.exists(os.path.join(experiment_dir_path, 'hyperparams_list.txt')))
        self.assertTrue(os.path.isfile(os.path.join(experiment_dir_path, 'hyperparams_list.txt')))
        self.assertTrue(os.path.exists(os.path.join(experiment_dir_path, THIS_FILE)))
        self.assertTrue(os.path.isfile(os.path.join(experiment_dir_path, THIS_FILE)))

        self.assertEqual(
            sorted(os.listdir(os.path.join(experiment_dir_path, 'checkpoint_model'))),
            [f'model_{train_loop.experiment_name}_{train_loop.experiment_timestamp}_E{ep}.pth'
             for ep in range(train_loop.epoch + 1)]
        )
        self.assertEqual(os.listdir(os.path.join(experiment_dir_path, 'model')),
                         [f'model_{train_loop.experiment_name}_{train_loop.experiment_timestamp}.pth'])

        results_dir_path = os.path.join(experiment_dir_path, 'results')

        self.assertEqual(sorted(os.listdir(os.path.join(results_dir_path, 'plots'))),
                         ['accumulated_loss.png', 'loss.png', 'val_loss.png'])

        results_pickle_path = os.path.join(
            results_dir_path,
            f'results_hyperParams_hist_{train_loop.experiment_name}_{train_loop.experiment_timestamp}.p'
        )
        with open(results_pickle_path, 'rb') as f:
            results_dict = pickle.load(f)

        self.assertEqual(list(results_dict.keys()),
                         ['y_true', 'y_predicted', 'experiment_name', 'experiment_results_local_path', 'results',
                          'hyperparameters', 'training_history'])

        self.assertEqual(results_dict['experiment_name'], train_loop.experiment_name)
        self.assertEqual(results_dict['experiment_results_local_path'], results_dir_path)

        self.assertEqual(results_dict['y_predicted']['ClassificationResult_TEST'].tolist(),
                         train_loop.predict_on_test_set()[0].tolist())
        self.assertEqual(results_dict['y_predicted']['ClassificationResult_VAL'].tolist(),
                         train_loop.predict_on_validation_set()[0].tolist())

        self.assertEqual(results_dict['y_true']['ClassificationResult_TEST'].tolist(),
                         train_loop.predict_on_test_set()[1].tolist())
        self.assertEqual(results_dict['y_true']['ClassificationResult_VAL'].tolist(),
                         train_loop.predict_on_validation_set()[1].tolist())

        self.assertEqual(
            results_dict['results'],
            {'ClassificationResult_TEST': {'Accuracy': 0.9847}, 'ClassificationResult_VAL': {'Accuracy': 0.9847}}
        )

        self.assertEqual(len(results_dict['hyperparameters']), 4)
        self.assertEqual(results_dict['hyperparameters']['num_epochs'], 3)
        self.assertEqual(results_dict['hyperparameters']['batch_size'], 100)
        self.assertEqual(results_dict['hyperparameters']['experiment_file_path'], __file__)
        self.assertEqual(results_dict['hyperparameters']['source_dirs_paths'], ())

        self.assertEqual(results_dict['training_history'], train_loop.train_history.train_history)

        project_path = os.path.join(THIS_DIR, 'e2e_train_loop_example')
        if os.path.exists(project_path):
            shutil.rmtree(project_path)
        data_path = os.path.join(THIS_DIR, 'mnist_data')
        if os.path.exists(data_path):
            shutil.rmtree(data_path)

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
