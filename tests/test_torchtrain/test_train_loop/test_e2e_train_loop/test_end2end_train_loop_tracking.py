import unittest

import os
import shutil
import random
import pickle
import numpy as np
import boto3
from moto import mock_s3

from tests.setup_moto_env import setup_aws_for_test

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import TensorDataset

from aitoolbox.torchtrain.train_loop import TrainLoopCheckpointEndSave, TrainLoopCheckpoint, TrainLoopEndSave
from aitoolbox.torchtrain.model import TTModel
from aitoolbox.experiment.result_package.basic_packages import ClassificationResultPackage
from aitoolbox.torchtrain.schedulers.basic import StepLRScheduler
from aitoolbox.torchtrain.schedulers.warmup import LinearWithWarmupScheduler

setup_aws_for_test()
BUCKET_NAME = 'test-bucket'
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
THIS_FILE = os.path.basename(__file__)


class FFNet(TTModel):
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

        return predicted.cpu(), target, {'example_feat_sum': input_data.sum(dim=1).tolist()}


class TestEnd2EndTrainLoopCheckpointEndSave(unittest.TestCase):
    def test_e2e_ff_net_train_loop(self):
        self.set_seeds()
        batch_size = 10

        train_dataset = TensorDataset(torch.randn(100, 50), torch.randint(low=0, high=10, size=(100,)))
        val_dataset = TensorDataset(torch.randn(30, 50), torch.randint(low=0, high=10, size=(30,)))
        test_dataset = TensorDataset(torch.randn(30, 50), torch.randint(low=0, high=10, size=(30,)))

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

        model = FFNet()
        optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
        criterion = nn.NLLLoss()

        train_loop = TrainLoopCheckpointEndSave(
            model,
            train_dataloader, val_dataloader, test_dataloader,
            optimizer, criterion,
            project_name='e2e_train_loop_example', experiment_name='TrainLoopCheckpointEndSave_example',
            local_model_result_folder_path=THIS_DIR,
            hyperparams={'batch_size': batch_size},
            val_result_package=ClassificationResultPackage(), test_result_package=ClassificationResultPackage(),
            cloud_save_mode=None
        )
        train_loop.fit(num_epochs=5)
        tl_history = train_loop.train_history.train_history

        self.assertEqual(train_loop.epoch, 4)

        result_approx = {
            'loss': [2.224587655067444, 2.1440203189849854, 2.0584306001663206, 1.962017869949341, 1.8507084131240845],
            'accumulated_loss': [2.3059947967529295, 2.1976317405700683, 2.114974856376648, 2.0259472250938417, 1.9252637863159179],
            'val_loss': [2.330514828364054, 2.345397472381592, 2.363233725229899, 2.3853348096211753, 2.4111196994781494],
            'train_end_test_loss': [2.31626296043396]
        }
        self.assertEqual(sorted(tl_history.keys()), sorted(result_approx.keys()))

        for metric, results_list in result_approx.items():
            self.assertEqual(len(results_list), len(tl_history[metric]))

            for correct_result, tl_result in zip(results_list, tl_history[metric]):
                self.assertAlmostEqual(correct_result, tl_result, places=6)

        # linux_result = {
        #     'loss': [2.224587655067444, 2.1440203189849854, 2.0584306001663206,
        #              1.962017869949341, 1.8507084846496582],
        #     'accumulated_loss': [2.3059947967529295, 2.1976317405700683, 2.114974856376648,
        #                          2.0259472012519835, 1.9252637863159179],
        #     'val_loss': [2.330514828364054, 2.345397472381592, 2.363233804702759,
        #                  2.3853348096211753, 2.4111196994781494],
        #     'train_end_test_loss': [2.31626296043396]
        # }
        # self.assertEqual(train_loop.train_history.train_history, linux_result)

        project_path = os.path.join(THIS_DIR, 'e2e_train_loop_example')
        if os.path.exists(project_path):
            shutil.rmtree(project_path)

    def test_e2e_ff_net_train_loop_loss(self):
        self.set_seeds()
        batch_size = 10

        train_dataset = TensorDataset(torch.randn(100, 50), torch.randint(low=0, high=10, size=(100,)))
        val_dataset = TensorDataset(torch.randn(30, 50), torch.randint(low=0, high=10, size=(30,)))
        test_dataset = TensorDataset(torch.randn(30, 50), torch.randint(low=0, high=10, size=(30,)))

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

        model = FFNet()
        optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
        criterion = nn.NLLLoss()

        train_loop = TrainLoopCheckpointEndSave(
            model,
            train_dataloader, val_dataloader, test_dataloader,
            optimizer, criterion,
            project_name='e2e_train_loop_example', experiment_name='TrainLoopCheckpointEndSave_example',
            local_model_result_folder_path=THIS_DIR,
            hyperparams={'batch_size': batch_size},
            val_result_package=ClassificationResultPackage(), test_result_package=ClassificationResultPackage(),
            cloud_save_mode=None
        )
        train_loop.fit(num_epochs=5)

        self.assertAlmostEqual(train_loop.evaluate_loss_on_train_set(), 1.8507084131240845, places=6)
        self.assertAlmostEqual(train_loop.evaluate_loss_on_validation_set(), 2.4111196994781494, places=6)
        self.assertAlmostEqual(train_loop.evaluate_loss_on_test_set(), 2.31626296043396, places=6)

        project_path = os.path.join(THIS_DIR, 'e2e_train_loop_example')
        if os.path.exists(project_path):
            shutil.rmtree(project_path)

    def test_e2e_ff_net_train_loop_predictions(self):
        self.set_seeds()
        batch_size = 10

        train_dataset = TensorDataset(torch.randn(100, 50), torch.randint(low=0, high=10, size=(100,)))
        val_dataset = TensorDataset(torch.randn(30, 50), torch.randint(low=0, high=10, size=(30,)))
        test_dataset = TensorDataset(torch.randn(30, 50), torch.randint(low=0, high=10, size=(30,)))

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

        model = FFNet()
        optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
        criterion = nn.NLLLoss()

        train_loop = TrainLoopCheckpointEndSave(
            model,
            train_dataloader, val_dataloader, test_dataloader,
            optimizer, criterion,
            project_name='e2e_train_loop_example', experiment_name='TrainLoopCheckpointEndSave_example',
            local_model_result_folder_path=THIS_DIR,
            hyperparams={'batch_size': batch_size},
            val_result_package=ClassificationResultPackage(), test_result_package=ClassificationResultPackage(),
            cloud_save_mode=None
        )
        train_loop.fit(num_epochs=5)

        train_pred, train_target, train_meta = train_loop.predict_on_train_set()
        self.assertEqual(
            train_pred.tolist(),
            [0, 4, 8, 0, 1, 1, 6, 1, 6, 1, 8, 3, 0, 0, 8, 8, 1, 8, 8, 1, 8, 0, 8, 0, 0, 1, 3, 4, 8, 8, 0, 8, 8, 1, 8, 8,
             5, 5, 1, 8, 8, 8, 8, 8, 9, 0, 8, 8, 5, 8, 8, 1, 1, 8, 5, 1, 8, 8, 5, 5, 1, 8, 8, 8, 8, 0, 0, 1, 1, 0, 8, 8,
             3, 0, 5, 0, 8, 9, 1, 8, 8, 8, 5, 8, 5, 8, 0, 1, 8, 5, 8, 6, 5, 1, 8, 1, 0, 8, 1, 8]
        )
        self.assertEqual(train_target.tolist(), train_dataset.tensors[1].tolist())
        self.assertEqual(train_dataset.tensors[0].sum(dim=1).tolist(), train_meta['example_feat_sum'])

        val_pred, val_target, val_meta = train_loop.predict_on_validation_set()
        self.assertEqual(
            val_pred.tolist(),
            [1, 1, 1, 1, 5, 8, 0, 8, 1, 1, 5, 8, 8, 1, 8, 8, 1, 8, 8, 8, 1, 0, 0, 8, 0, 1, 1, 0, 1, 8]
        )
        self.assertEqual(val_target.tolist(), val_dataset.tensors[1].tolist())
        self.assertEqual(val_dataset.tensors[0].sum(dim=1).tolist(), val_meta['example_feat_sum'])

        test_pred, test_target, test_meta = train_loop.predict_on_test_set()
        self.assertEqual(
            test_pred.tolist(),
            [4, 8, 0, 8, 1, 8, 1, 1, 8, 8, 8, 8, 8, 0, 8, 8, 5, 8, 8, 5, 8, 1, 0, 5, 1, 8, 8, 8, 8, 1]
        )
        self.assertEqual(test_target.tolist(), test_dataset.tensors[1].tolist())
        self.assertEqual(test_dataset.tensors[0].sum(dim=1).tolist(), test_meta['example_feat_sum'])

        project_path = os.path.join(THIS_DIR, 'e2e_train_loop_example')
        if os.path.exists(project_path):
            shutil.rmtree(project_path)

    def test_e2e_ff_net_train_loop_tracking_saved_files_check(self):
        self.set_seeds()
        batch_size = 10

        train_dataset = TensorDataset(torch.randn(100, 50), torch.randint(low=0, high=10, size=(100,)))
        val_dataset = TensorDataset(torch.randn(40, 50), torch.randint(low=0, high=10, size=(40,)))
        test_dataset = TensorDataset(torch.randn(30, 50), torch.randint(low=0, high=10, size=(30,)))

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

        model = FFNet()
        optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
        criterion = nn.NLLLoss()

        train_loop = TrainLoopCheckpointEndSave(
            model,
            train_dataloader, val_dataloader, test_dataloader,
            optimizer, criterion,
            project_name='e2e_train_loop_example', experiment_name='TrainLoopCheckpointEndSave_example',
            local_model_result_folder_path=THIS_DIR,
            hyperparams={'batch_size': batch_size},
            val_result_package=ClassificationResultPackage(), test_result_package=ClassificationResultPackage(),
            cloud_save_mode=None
        )
        train_loop.fit(num_epochs=5)

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
            {'ClassificationResult_TEST': {'Accuracy': 0.1}, 'ClassificationResult_VAL': {'Accuracy': 0.125}}
        )

        self.assertEqual(len(results_dict['hyperparameters']), 3)
        self.assertEqual(results_dict['hyperparameters']['batch_size'], 10)
        self.assertEqual(results_dict['hyperparameters']['experiment_file_path'], __file__)
        self.assertEqual(results_dict['hyperparameters']['source_dirs_paths'], ())

        self.assertEqual(results_dict['training_history'], train_loop.train_history.train_history)

        project_path = os.path.join(THIS_DIR, 'e2e_train_loop_example')
        if os.path.exists(project_path):
            shutil.rmtree(project_path)

    def test_e2e_ff_net_train_loop_tracking_saved_model_snapshot(self):
        self.set_seeds()
        batch_size = 10

        train_dataset = TensorDataset(torch.randn(100, 50), torch.randint(low=0, high=10, size=(100,)))
        val_dataset = TensorDataset(torch.randn(40, 50), torch.randint(low=0, high=10, size=(40,)))
        test_dataset = TensorDataset(torch.randn(30, 50), torch.randint(low=0, high=10, size=(30,)))

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

        scheduler_cb = [
            StepLRScheduler(step_size=2),
            LinearWithWarmupScheduler(num_warmup_steps=1, num_training_steps=len(train_dataloader) * 5)
        ]

        model = FFNet()
        optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
        criterion = nn.NLLLoss()

        train_loop = TrainLoopCheckpointEndSave(
            model,
            train_dataloader, val_dataloader, test_dataloader,
            optimizer, criterion,
            project_name='e2e_train_loop_example', experiment_name='TrainLoopCheckpointEndSave_example',
            local_model_result_folder_path=THIS_DIR,
            hyperparams={'batch_size': batch_size},
            val_result_package=ClassificationResultPackage(), test_result_package=ClassificationResultPackage(),
            cloud_save_mode=None
        )
        train_loop.fit(num_epochs=5, callbacks=scheduler_cb)

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

        model_representation = torch.load(
            os.path.join(experiment_dir_path, 'model',
                         f'model_{train_loop.experiment_name}_{train_loop.experiment_timestamp}.pth')
        )
        self.check_loaded_representation(model_representation, optimizer, scheduler_cb)

        checkpoint_representation = torch.load(
            os.path.join(experiment_dir_path, 'checkpoint_model',
                         f'model_{train_loop.experiment_name}_{train_loop.experiment_timestamp}_E4.pth')
        )
        self.check_loaded_representation(checkpoint_representation, optimizer, scheduler_cb)

        project_path = os.path.join(THIS_DIR, 'e2e_train_loop_example')
        if os.path.exists(project_path):
            shutil.rmtree(project_path)

    def check_loaded_representation(self, model_representation, optimizer, scheduler_cb):
        self.assertEqual(model_representation['optimizer_state_dict'].keys(), optimizer.state_dict().keys())

        loaded_optimizer_state = model_representation['optimizer_state_dict']['state']
        for state_idx in range(len(loaded_optimizer_state)):
            opti_state = optimizer.state_dict()['state'][state_idx]
            loaded_state = loaded_optimizer_state[state_idx]

            self.assertEqual(opti_state.keys(), loaded_state.keys())
            self.assertEqual(opti_state['step'], loaded_state['step'])
            self.assertEqual(opti_state['exp_avg'].tolist(), loaded_state['exp_avg'].tolist())
            self.assertEqual(opti_state['exp_avg_sq'].tolist(), loaded_state['exp_avg_sq'].tolist())

        self.assertEqual(
            model_representation['optimizer_state_dict']['param_groups'],
            optimizer.state_dict()['param_groups']
        )

        for scheduler_idx in range(len(scheduler_cb)):
            self.assertEqual(
                model_representation['schedulers_state_dict'][scheduler_idx],
                scheduler_cb[scheduler_idx].state_dict()
            )

    @mock_s3
    def test_e2e_ff_net_train_loop_cloud_save(self):
        s3 = boto3.resource('s3', region_name='us-east-1')
        s3.create_bucket(Bucket=BUCKET_NAME)
        s3_client = boto3.client('s3')

        self.set_seeds()
        batch_size = 10
        num_epochs = 10

        train_dataset = TensorDataset(torch.randn(100, 50), torch.randint(low=0, high=10, size=(100,)))
        val_dataset = TensorDataset(torch.randn(30, 50), torch.randint(low=0, high=10, size=(30,)))
        test_dataset = TensorDataset(torch.randn(30, 50), torch.randint(low=0, high=10, size=(30,)))

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

        model = FFNet()
        optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
        criterion = nn.NLLLoss()

        train_loop = TrainLoopCheckpointEndSave(
            model,
            train_dataloader, val_dataloader, test_dataloader,
            optimizer, criterion,
            project_name='e2e_train_loop_example', experiment_name='TrainLoopEndSave_example',
            local_model_result_folder_path=THIS_DIR,
            hyperparams={'batch_size': batch_size, 'num_epochs': num_epochs},
            val_result_package=ClassificationResultPackage(), test_result_package=ClassificationResultPackage(),
            cloud_save_mode='s3', bucket_name=BUCKET_NAME, cloud_dir_prefix='experiment_results'
        )
        train_loop.fit(num_epochs=num_epochs)

        bucket_content = [el['Key'] for el in s3_client.list_objects(Bucket=BUCKET_NAME)['Contents']]

        cloud_experiment_folder = f'{train_loop.cloud_dir_prefix}/{train_loop.project_name}/' \
                                  f'{train_loop.experiment_name}_{train_loop.experiment_timestamp}'
        model_checkpoints = [
            f'{cloud_experiment_folder}/'
            f'checkpoint_model/model_{train_loop.experiment_name}_{train_loop.experiment_timestamp}_E{i}.pth'
            for i in range(num_epochs)
        ]
        additional_files = [
            f'{cloud_experiment_folder}/hyperparams_list.txt',
            f'{cloud_experiment_folder}/model/model_{train_loop.experiment_name}_{train_loop.experiment_timestamp}.pth',
            f'{cloud_experiment_folder}/results/plots/accumulated_loss.png',
            f'{cloud_experiment_folder}/results/plots/loss.png',
            f'{cloud_experiment_folder}/results/plots/val_loss.png',
            f'{cloud_experiment_folder}/results/results_hyperParams_hist_{train_loop.experiment_name}_{train_loop.experiment_timestamp}.p',
            f'{cloud_experiment_folder}/{THIS_FILE}'
        ]
        expected_files = model_checkpoints + additional_files

        self.assertEqual(bucket_content, expected_files)

        project_path = os.path.join(THIS_DIR, 'e2e_train_loop_example')
        if os.path.exists(project_path):
            shutil.rmtree(project_path)

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


class TestEnd2EndTrainLoopCheckpoint(unittest.TestCase):
    def test_e2e_ff_net_train_loop(self):
        self.set_seeds()
        batch_size = 10

        train_dataset = TensorDataset(torch.randn(100, 50), torch.randint(low=0, high=10, size=(100,)))
        val_dataset = TensorDataset(torch.randn(30, 50), torch.randint(low=0, high=10, size=(30,)))
        test_dataset = TensorDataset(torch.randn(30, 50), torch.randint(low=0, high=10, size=(30,)))

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

        model = FFNet()
        optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
        criterion = nn.NLLLoss()

        train_loop = TrainLoopCheckpoint(
            model,
            train_dataloader, val_dataloader, test_dataloader,
            optimizer, criterion,
            project_name='e2e_train_loop_example', experiment_name='TrainLoopCheckpoint_example',
            local_model_result_folder_path=THIS_DIR,
            hyperparams={'batch_size': batch_size},
            cloud_save_mode=None
        )
        train_loop.fit(num_epochs=5)
        tl_history = train_loop.train_history.train_history

        self.assertEqual(train_loop.epoch, 4)

        result_approx = {
            'loss': [2.224587655067444, 2.1440203189849854, 2.0584306001663206, 1.962017869949341, 1.8507084131240845],
            'accumulated_loss': [2.3059947967529295, 2.1976317405700683, 2.114974856376648, 2.0259472250938417, 1.9252637863159179],
            'val_loss': [2.330514828364054, 2.345397472381592, 2.363233725229899, 2.3853348096211753, 2.4111196994781494],
            'train_end_test_loss': [2.31626296043396]
        }
        self.assertEqual(sorted(tl_history.keys()), sorted(result_approx.keys()))

        for metric, results_list in result_approx.items():
            self.assertEqual(len(results_list), len(tl_history[metric]))

            for correct_result, tl_result in zip(results_list, tl_history[metric]):
                self.assertAlmostEqual(correct_result, tl_result, places=6)

        # linux_result = {
        #     'loss': [2.224587655067444, 2.1440203189849854, 2.0584306001663206,
        #              1.962017869949341, 1.8507084846496582],
        #     'accumulated_loss': [2.3059947967529295, 2.1976317405700683, 2.114974856376648,
        #                          2.0259472012519835, 1.9252637863159179],
        #     'val_loss': [2.330514828364054, 2.345397472381592, 2.363233804702759,
        #                  2.3853348096211753, 2.4111196994781494],
        #     'train_end_test_loss': [2.31626296043396]
        # }
        # self.assertEqual(train_loop.train_history.train_history, linux_result)

        project_path = os.path.join(THIS_DIR, 'e2e_train_loop_example')
        if os.path.exists(project_path):
            shutil.rmtree(project_path)

    def test_e2e_ff_net_train_loop_loss(self):
        self.set_seeds()
        batch_size = 10

        train_dataset = TensorDataset(torch.randn(100, 50), torch.randint(low=0, high=10, size=(100,)))
        val_dataset = TensorDataset(torch.randn(30, 50), torch.randint(low=0, high=10, size=(30,)))
        test_dataset = TensorDataset(torch.randn(30, 50), torch.randint(low=0, high=10, size=(30,)))

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

        model = FFNet()
        optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
        criterion = nn.NLLLoss()

        train_loop = TrainLoopCheckpoint(
            model,
            train_dataloader, val_dataloader, test_dataloader,
            optimizer, criterion,
            project_name='e2e_train_loop_example', experiment_name='TrainLoopCheckpoint_example',
            local_model_result_folder_path=THIS_DIR,
            hyperparams={'batch_size': batch_size},
            cloud_save_mode=None
        )
        train_loop.fit(num_epochs=5)

        self.assertAlmostEqual(train_loop.evaluate_loss_on_train_set(), 1.8507084131240845, places=6)
        self.assertAlmostEqual(train_loop.evaluate_loss_on_validation_set(), 2.4111196994781494, places=6)
        self.assertAlmostEqual(train_loop.evaluate_loss_on_test_set(), 2.31626296043396, places=6)

        project_path = os.path.join(THIS_DIR, 'e2e_train_loop_example')
        if os.path.exists(project_path):
            shutil.rmtree(project_path)

    def test_e2e_ff_net_train_loop_predictions(self):
        self.set_seeds()
        batch_size = 10

        train_dataset = TensorDataset(torch.randn(100, 50), torch.randint(low=0, high=10, size=(100,)))
        val_dataset = TensorDataset(torch.randn(30, 50), torch.randint(low=0, high=10, size=(30,)))
        test_dataset = TensorDataset(torch.randn(30, 50), torch.randint(low=0, high=10, size=(30,)))

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

        model = FFNet()
        optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
        criterion = nn.NLLLoss()

        train_loop = TrainLoopCheckpoint(
            model,
            train_dataloader, val_dataloader, test_dataloader,
            optimizer, criterion,
            project_name='e2e_train_loop_example', experiment_name='TrainLoopCheckpoint_example',
            local_model_result_folder_path=THIS_DIR,
            hyperparams={'batch_size': batch_size},
            cloud_save_mode=None
        )
        train_loop.fit(num_epochs=5)

        train_pred, train_target, train_meta = train_loop.predict_on_train_set()
        self.assertEqual(
            train_pred.tolist(),
            [0, 4, 8, 0, 1, 1, 6, 1, 6, 1, 8, 3, 0, 0, 8, 8, 1, 8, 8, 1, 8, 0, 8, 0, 0, 1, 3, 4, 8, 8, 0, 8, 8, 1, 8, 8,
             5, 5, 1, 8, 8, 8, 8, 8, 9, 0, 8, 8, 5, 8, 8, 1, 1, 8, 5, 1, 8, 8, 5, 5, 1, 8, 8, 8, 8, 0, 0, 1, 1, 0, 8, 8,
             3, 0, 5, 0, 8, 9, 1, 8, 8, 8, 5, 8, 5, 8, 0, 1, 8, 5, 8, 6, 5, 1, 8, 1, 0, 8, 1, 8]
        )
        self.assertEqual(train_target.tolist(), train_dataset.tensors[1].tolist())
        self.assertEqual(train_dataset.tensors[0].sum(dim=1).tolist(), train_meta['example_feat_sum'])

        val_pred, val_target, val_meta = train_loop.predict_on_validation_set()
        self.assertEqual(
            val_pred.tolist(),
            [1, 1, 1, 1, 5, 8, 0, 8, 1, 1, 5, 8, 8, 1, 8, 8, 1, 8, 8, 8, 1, 0, 0, 8, 0, 1, 1, 0, 1, 8]
        )
        self.assertEqual(val_target.tolist(), val_dataset.tensors[1].tolist())
        self.assertEqual(val_dataset.tensors[0].sum(dim=1).tolist(), val_meta['example_feat_sum'])

        test_pred, test_target, test_meta = train_loop.predict_on_test_set()
        self.assertEqual(
            test_pred.tolist(),
            [4, 8, 0, 8, 1, 8, 1, 1, 8, 8, 8, 8, 8, 0, 8, 8, 5, 8, 8, 5, 8, 1, 0, 5, 1, 8, 8, 8, 8, 1]
        )
        self.assertEqual(test_target.tolist(), test_dataset.tensors[1].tolist())
        self.assertEqual(test_dataset.tensors[0].sum(dim=1).tolist(), test_meta['example_feat_sum'])

        project_path = os.path.join(THIS_DIR, 'e2e_train_loop_example')
        if os.path.exists(project_path):
            shutil.rmtree(project_path)

    def test_e2e_ff_net_train_loop_tracking_saved_files_check(self):
        self.set_seeds()
        batch_size = 10

        train_dataset = TensorDataset(torch.randn(100, 50), torch.randint(low=0, high=10, size=(100,)))
        val_dataset = TensorDataset(torch.randn(40, 50), torch.randint(low=0, high=10, size=(40,)))
        test_dataset = TensorDataset(torch.randn(30, 50), torch.randint(low=0, high=10, size=(30,)))

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

        model = FFNet()
        optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
        criterion = nn.NLLLoss()

        train_loop = TrainLoopCheckpoint(
            model,
            train_dataloader, val_dataloader, test_dataloader,
            optimizer, criterion,
            project_name='e2e_train_loop_example', experiment_name='TrainLoopCheckpoint_example',
            local_model_result_folder_path=THIS_DIR,
            hyperparams={'batch_size': batch_size},
            cloud_save_mode=None
        )
        train_loop.fit(num_epochs=5)

        experiment_dir_path = os.path.join(THIS_DIR, train_loop.project_name,
                                           f'{train_loop.experiment_name}_{train_loop.experiment_timestamp}')

        self.assertTrue(os.path.exists(os.path.join(experiment_dir_path, 'checkpoint_model')))
        self.assertTrue(os.path.isdir(os.path.join(experiment_dir_path, 'checkpoint_model')))

        self.assertTrue(os.path.exists(os.path.join(experiment_dir_path, 'hyperparams_list.txt')))
        self.assertTrue(os.path.isfile(os.path.join(experiment_dir_path, 'hyperparams_list.txt')))
        self.assertTrue(os.path.exists(os.path.join(experiment_dir_path, THIS_FILE)))
        self.assertTrue(os.path.isfile(os.path.join(experiment_dir_path, THIS_FILE)))

        self.assertEqual(
            sorted(os.listdir(os.path.join(experiment_dir_path, 'checkpoint_model'))),
            [f'model_{train_loop.experiment_name}_{train_loop.experiment_timestamp}_E{ep}.pth'
             for ep in range(train_loop.epoch + 1)]
        )

        project_path = os.path.join(THIS_DIR, 'e2e_train_loop_example')
        if os.path.exists(project_path):
            shutil.rmtree(project_path)

    @mock_s3
    def test_e2e_ff_net_train_loop_cloud_save(self):
        s3 = boto3.resource('s3', region_name='us-east-1')
        s3.create_bucket(Bucket=BUCKET_NAME)
        s3_client = boto3.client('s3')

        self.set_seeds()
        batch_size = 10
        num_epochs = 10

        train_dataset = TensorDataset(torch.randn(100, 50), torch.randint(low=0, high=10, size=(100,)))
        val_dataset = TensorDataset(torch.randn(30, 50), torch.randint(low=0, high=10, size=(30,)))
        test_dataset = TensorDataset(torch.randn(30, 50), torch.randint(low=0, high=10, size=(30,)))

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

        model = FFNet()
        optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
        criterion = nn.NLLLoss()

        train_loop = TrainLoopCheckpoint(
            model,
            train_dataloader, val_dataloader, test_dataloader,
            optimizer, criterion,
            project_name='e2e_train_loop_example', experiment_name='TrainLoopCheckpoint_example',
            local_model_result_folder_path=THIS_DIR,
            hyperparams={'batch_size': batch_size, 'num_epochs': num_epochs},
            cloud_save_mode='s3', bucket_name=BUCKET_NAME, cloud_dir_prefix='experiment_results'
        )
        train_loop.fit(num_epochs=num_epochs)

        bucket_content = [el['Key'] for el in s3_client.list_objects(Bucket=BUCKET_NAME)['Contents']]

        cloud_experiment_folder = f'{train_loop.cloud_dir_prefix}/{train_loop.project_name}/' \
                                  f'{train_loop.experiment_name}_{train_loop.experiment_timestamp}'
        model_checkpoints = [
            f'{cloud_experiment_folder}/'
            f'checkpoint_model/model_{train_loop.experiment_name}_{train_loop.experiment_timestamp}_E{i}.pth'
            for i in range(num_epochs)
        ]
        additional_files = [
            f'{cloud_experiment_folder}/hyperparams_list.txt',
            f'{cloud_experiment_folder}/{THIS_FILE}'
        ]
        expected_files = model_checkpoints + additional_files

        self.assertEqual(bucket_content, expected_files)

        project_path = os.path.join(THIS_DIR, 'e2e_train_loop_example')
        if os.path.exists(project_path):
            shutil.rmtree(project_path)

    def test_e2e_ff_net_train_loop_iteration_model_checkpointing(self):
        self.set_seeds()
        batch_size = 10

        train_dataset = TensorDataset(torch.randn(100, 50), torch.randint(low=0, high=10, size=(100,)))
        val_dataset = TensorDataset(torch.randn(30, 50), torch.randint(low=0, high=10, size=(30,)))
        test_dataset = TensorDataset(torch.randn(30, 50), torch.randint(low=0, high=10, size=(30,)))

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

        model = FFNet()
        optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
        criterion = nn.NLLLoss()

        train_loop = TrainLoopCheckpoint(
            model,
            train_dataloader, val_dataloader, test_dataloader,
            optimizer, criterion,
            project_name='e2e_train_loop_example', experiment_name='TrainLoopCheckpoint_example',
            local_model_result_folder_path=THIS_DIR,
            hyperparams={'batch_size': batch_size},
            cloud_save_mode=None,
            iteration_save_freq=5
        )
        train_loop.fit(num_iterations=100)

        experiment_dir_path = os.path.join(THIS_DIR, train_loop.project_name,
                                           f'{train_loop.experiment_name}_{train_loop.experiment_timestamp}')

        self.assertEqual(
            sorted(os.listdir(os.path.join(experiment_dir_path, 'checkpoint_model'))),
            sorted(
                [f'model_{train_loop.experiment_name}_{train_loop.experiment_timestamp}_E{ep}.pth'
                 for ep in range(10)] +
                [f'model_{train_loop.experiment_name}_{train_loop.experiment_timestamp}_E{int(iteration / 10)}_ITER{iteration}.pth'
                 for iteration in range(5, 100, 5)]
            )
        )

        project_path = os.path.join(THIS_DIR, 'e2e_train_loop_example')
        if os.path.exists(project_path):
            shutil.rmtree(project_path)

    @mock_s3
    def test_e2e_ff_net_train_loop_iteration_model_checkpointing_cloud_save(self):
        s3 = boto3.resource('s3', region_name='us-east-1')
        s3.create_bucket(Bucket=BUCKET_NAME)
        s3_client = boto3.client('s3')

        self.set_seeds()
        batch_size = 10

        train_dataset = TensorDataset(torch.randn(100, 50), torch.randint(low=0, high=10, size=(100,)))
        val_dataset = TensorDataset(torch.randn(30, 50), torch.randint(low=0, high=10, size=(30,)))
        test_dataset = TensorDataset(torch.randn(30, 50), torch.randint(low=0, high=10, size=(30,)))

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

        model = FFNet()
        optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
        criterion = nn.NLLLoss()

        train_loop = TrainLoopCheckpoint(
            model,
            train_dataloader, val_dataloader, test_dataloader,
            optimizer, criterion,
            project_name='e2e_train_loop_example', experiment_name='TrainLoopCheckpoint_example',
            local_model_result_folder_path=THIS_DIR,
            hyperparams={'batch_size': batch_size},
            cloud_save_mode='s3', bucket_name=BUCKET_NAME, cloud_dir_prefix='experiment_results',
            iteration_save_freq=5
        )
        train_loop.fit(num_iterations=100)

        bucket_content = [el['Key'] for el in s3_client.list_objects(Bucket=BUCKET_NAME)['Contents']]

        cloud_experiment_folder = f'{train_loop.cloud_dir_prefix}/{train_loop.project_name}/' \
                                  f'{train_loop.experiment_name}_{train_loop.experiment_timestamp}'

        model_checkpoint_files = \
            [f'model_{train_loop.experiment_name}_{train_loop.experiment_timestamp}_E{ep}.pth'
             for ep in range(10)] + \
            [f'model_{train_loop.experiment_name}_{train_loop.experiment_timestamp}_E{int(iteration / 10)}_ITER{iteration}.pth'
             for iteration in range(5, 100, 5)]
        model_checkpoints = [
            f'{cloud_experiment_folder}/checkpoint_model/{model_fn}' for model_fn in model_checkpoint_files
        ]
        additional_files = [
            f'{cloud_experiment_folder}/hyperparams_list.txt',
            f'{cloud_experiment_folder}/{THIS_FILE}'
        ]
        expected_files = model_checkpoints + additional_files

        self.assertEqual(sorted(bucket_content), sorted(expected_files))

        project_path = os.path.join(THIS_DIR, 'e2e_train_loop_example')
        if os.path.exists(project_path):
            shutil.rmtree(project_path)

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


class TestEnd2EndTrainLoopEndSave(unittest.TestCase):
    def test_e2e_ff_net_train_loop(self):
        self.set_seeds()
        batch_size = 10

        train_dataset = TensorDataset(torch.randn(100, 50), torch.randint(low=0, high=10, size=(100,)))
        val_dataset = TensorDataset(torch.randn(30, 50), torch.randint(low=0, high=10, size=(30,)))
        test_dataset = TensorDataset(torch.randn(30, 50), torch.randint(low=0, high=10, size=(30,)))

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

        model = FFNet()
        optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
        criterion = nn.NLLLoss()

        train_loop = TrainLoopEndSave(
            model,
            train_dataloader, val_dataloader, test_dataloader,
            optimizer, criterion,
            project_name='e2e_train_loop_example', experiment_name='TrainLoopEndSave_example',
            local_model_result_folder_path=THIS_DIR,
            hyperparams={'batch_size': batch_size},
            val_result_package=ClassificationResultPackage(), test_result_package=ClassificationResultPackage(),
            cloud_save_mode=None
        )
        train_loop.fit(num_epochs=5)
        tl_history = train_loop.train_history.train_history

        self.assertEqual(train_loop.epoch, 4)

        result_approx = {
            'loss': [2.224587655067444, 2.1440203189849854, 2.0584306001663206, 1.962017869949341, 1.8507084131240845],
            'accumulated_loss': [2.3059947967529295, 2.1976317405700683, 2.114974856376648, 2.0259472250938417, 1.9252637863159179],
            'val_loss': [2.330514828364054, 2.345397472381592, 2.363233725229899, 2.3853348096211753, 2.4111196994781494],
            'train_end_test_loss': [2.31626296043396]
        }
        self.assertEqual(sorted(tl_history.keys()), sorted(result_approx.keys()))

        for metric, results_list in result_approx.items():
            self.assertEqual(len(results_list), len(tl_history[metric]))

            for correct_result, tl_result in zip(results_list, tl_history[metric]):
                self.assertAlmostEqual(correct_result, tl_result, places=6)

        # linux_result = {
        #     'loss': [2.224587655067444, 2.1440203189849854, 2.0584306001663206,
        #              1.962017869949341, 1.8507084846496582],
        #     'accumulated_loss': [2.3059947967529295, 2.1976317405700683, 2.114974856376648,
        #                          2.0259472012519835, 1.9252637863159179],
        #     'val_loss': [2.330514828364054, 2.345397472381592, 2.363233804702759,
        #                  2.3853348096211753, 2.4111196994781494],
        #     'train_end_test_loss': [2.31626296043396]
        # }
        # self.assertEqual(train_loop.train_history.train_history, linux_result)

        project_path = os.path.join(THIS_DIR, 'e2e_train_loop_example')
        if os.path.exists(project_path):
            shutil.rmtree(project_path)

    def test_e2e_ff_net_train_loop_loss(self):
        self.set_seeds()
        batch_size = 10

        train_dataset = TensorDataset(torch.randn(100, 50), torch.randint(low=0, high=10, size=(100,)))
        val_dataset = TensorDataset(torch.randn(30, 50), torch.randint(low=0, high=10, size=(30,)))
        test_dataset = TensorDataset(torch.randn(30, 50), torch.randint(low=0, high=10, size=(30,)))

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

        model = FFNet()
        optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
        criterion = nn.NLLLoss()

        train_loop = TrainLoopEndSave(
            model,
            train_dataloader, val_dataloader, test_dataloader,
            optimizer, criterion,
            project_name='e2e_train_loop_example', experiment_name='TrainLoopEndSave_example',
            local_model_result_folder_path=THIS_DIR,
            hyperparams={'batch_size': batch_size},
            val_result_package=ClassificationResultPackage(), test_result_package=ClassificationResultPackage(),
            cloud_save_mode=None
        )
        train_loop.fit(num_epochs=5)

        self.assertAlmostEqual(train_loop.evaluate_loss_on_train_set(), 1.8507084131240845, places=6)
        self.assertAlmostEqual(train_loop.evaluate_loss_on_validation_set(), 2.4111196994781494, places=6)
        self.assertAlmostEqual(train_loop.evaluate_loss_on_test_set(), 2.31626296043396, places=6)

        project_path = os.path.join(THIS_DIR, 'e2e_train_loop_example')
        if os.path.exists(project_path):
            shutil.rmtree(project_path)

    def test_e2e_ff_net_train_loop_predictions(self):
        self.set_seeds()
        batch_size = 10

        train_dataset = TensorDataset(torch.randn(100, 50), torch.randint(low=0, high=10, size=(100,)))
        val_dataset = TensorDataset(torch.randn(30, 50), torch.randint(low=0, high=10, size=(30,)))
        test_dataset = TensorDataset(torch.randn(30, 50), torch.randint(low=0, high=10, size=(30,)))

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

        model = FFNet()
        optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
        criterion = nn.NLLLoss()

        train_loop = TrainLoopEndSave(
            model,
            train_dataloader, val_dataloader, test_dataloader,
            optimizer, criterion,
            project_name='e2e_train_loop_example', experiment_name='TrainLoopEndSave_example',
            local_model_result_folder_path=THIS_DIR,
            hyperparams={'batch_size': batch_size},
            val_result_package=ClassificationResultPackage(), test_result_package=ClassificationResultPackage(),
            cloud_save_mode=None
        )
        train_loop.fit(num_epochs=5)

        train_pred, train_target, train_meta = train_loop.predict_on_train_set()
        self.assertEqual(
            train_pred.tolist(),
            [0, 4, 8, 0, 1, 1, 6, 1, 6, 1, 8, 3, 0, 0, 8, 8, 1, 8, 8, 1, 8, 0, 8, 0, 0, 1, 3, 4, 8, 8, 0, 8, 8, 1, 8, 8,
             5, 5, 1, 8, 8, 8, 8, 8, 9, 0, 8, 8, 5, 8, 8, 1, 1, 8, 5, 1, 8, 8, 5, 5, 1, 8, 8, 8, 8, 0, 0, 1, 1, 0, 8, 8,
             3, 0, 5, 0, 8, 9, 1, 8, 8, 8, 5, 8, 5, 8, 0, 1, 8, 5, 8, 6, 5, 1, 8, 1, 0, 8, 1, 8]
        )
        self.assertEqual(train_target.tolist(), train_dataset.tensors[1].tolist())
        self.assertEqual(train_dataset.tensors[0].sum(dim=1).tolist(), train_meta['example_feat_sum'])

        val_pred, val_target, val_meta = train_loop.predict_on_validation_set()
        self.assertEqual(
            val_pred.tolist(),
            [1, 1, 1, 1, 5, 8, 0, 8, 1, 1, 5, 8, 8, 1, 8, 8, 1, 8, 8, 8, 1, 0, 0, 8, 0, 1, 1, 0, 1, 8]
        )
        self.assertEqual(val_target.tolist(), val_dataset.tensors[1].tolist())
        self.assertEqual(val_dataset.tensors[0].sum(dim=1).tolist(), val_meta['example_feat_sum'])

        test_pred, test_target, test_meta = train_loop.predict_on_test_set()
        self.assertEqual(
            test_pred.tolist(),
            [4, 8, 0, 8, 1, 8, 1, 1, 8, 8, 8, 8, 8, 0, 8, 8, 5, 8, 8, 5, 8, 1, 0, 5, 1, 8, 8, 8, 8, 1]
        )
        self.assertEqual(test_target.tolist(), test_dataset.tensors[1].tolist())
        self.assertEqual(test_dataset.tensors[0].sum(dim=1).tolist(), test_meta['example_feat_sum'])

        project_path = os.path.join(THIS_DIR, 'e2e_train_loop_example')
        if os.path.exists(project_path):
            shutil.rmtree(project_path)

    def test_e2e_ff_net_train_loop_tracking_saved_files_check(self):
        self.set_seeds()
        batch_size = 10

        train_dataset = TensorDataset(torch.randn(100, 50), torch.randint(low=0, high=10, size=(100,)))
        val_dataset = TensorDataset(torch.randn(40, 50), torch.randint(low=0, high=10, size=(40,)))
        test_dataset = TensorDataset(torch.randn(30, 50), torch.randint(low=0, high=10, size=(30,)))

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

        model = FFNet()
        optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
        criterion = nn.NLLLoss()

        train_loop = TrainLoopEndSave(
            model,
            train_dataloader, val_dataloader, test_dataloader,
            optimizer, criterion,
            project_name='e2e_train_loop_example', experiment_name='TrainLoopEndSave_example',
            local_model_result_folder_path=THIS_DIR,
            hyperparams={'batch_size': batch_size},
            val_result_package=ClassificationResultPackage(), test_result_package=ClassificationResultPackage(),
            cloud_save_mode=None
        )
        train_loop.fit(num_epochs=5)

        experiment_dir_path = os.path.join(THIS_DIR, train_loop.project_name,
                                           f'{train_loop.experiment_name}_{train_loop.experiment_timestamp}')

        self.assertTrue(os.path.exists(os.path.join(experiment_dir_path, 'model')))
        self.assertTrue(os.path.isdir(os.path.join(experiment_dir_path, 'model')))
        self.assertTrue(os.path.exists(os.path.join(experiment_dir_path, 'results')))
        self.assertTrue(os.path.isdir(os.path.join(experiment_dir_path, 'results')))

        self.assertTrue(os.path.exists(os.path.join(experiment_dir_path, 'hyperparams_list.txt')))
        self.assertTrue(os.path.isfile(os.path.join(experiment_dir_path, 'hyperparams_list.txt')))
        self.assertTrue(os.path.exists(os.path.join(experiment_dir_path, THIS_FILE)))
        self.assertTrue(os.path.isfile(os.path.join(experiment_dir_path, THIS_FILE)))

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
            {'ClassificationResult_TEST': {'Accuracy': 0.1}, 'ClassificationResult_VAL': {'Accuracy': 0.125}}
        )

        self.assertEqual(len(results_dict['hyperparameters']), 3)
        self.assertEqual(results_dict['hyperparameters']['batch_size'], 10)
        self.assertEqual(results_dict['hyperparameters']['experiment_file_path'], __file__)
        self.assertEqual(results_dict['hyperparameters']['source_dirs_paths'], ())

        self.assertEqual(results_dict['training_history'], train_loop.train_history.train_history)

        project_path = os.path.join(THIS_DIR, 'e2e_train_loop_example')
        if os.path.exists(project_path):
            shutil.rmtree(project_path)

    @mock_s3
    def test_e2e_ff_net_train_loop_cloud_save(self):
        s3 = boto3.resource('s3', region_name='us-east-1')
        s3.create_bucket(Bucket=BUCKET_NAME)
        s3_client = boto3.client('s3')

        self.set_seeds()
        batch_size = 10
        num_epochs = 10

        train_dataset = TensorDataset(torch.randn(100, 50), torch.randint(low=0, high=10, size=(100,)))
        val_dataset = TensorDataset(torch.randn(30, 50), torch.randint(low=0, high=10, size=(30,)))
        test_dataset = TensorDataset(torch.randn(30, 50), torch.randint(low=0, high=10, size=(30,)))

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

        model = FFNet()
        optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
        criterion = nn.NLLLoss()

        train_loop = TrainLoopEndSave(
            model,
            train_dataloader, val_dataloader, test_dataloader,
            optimizer, criterion,
            project_name='e2e_train_loop_example', experiment_name='TrainLoopEndSave_example',
            local_model_result_folder_path=THIS_DIR,
            hyperparams={'batch_size': batch_size, 'num_epochs': num_epochs},
            val_result_package=ClassificationResultPackage(), test_result_package=ClassificationResultPackage(),
            cloud_save_mode='s3', bucket_name=BUCKET_NAME, cloud_dir_prefix='experiment_results'
        )
        train_loop.fit(num_epochs=num_epochs)

        bucket_content = [el['Key'] for el in s3_client.list_objects(Bucket=BUCKET_NAME)['Contents']]

        cloud_experiment_folder = f'{train_loop.cloud_dir_prefix}/{train_loop.project_name}/' \
                                  f'{train_loop.experiment_name}_{train_loop.experiment_timestamp}'
        expected_files = [
            f'{cloud_experiment_folder}/hyperparams_list.txt',
            f'{cloud_experiment_folder}/model/model_{train_loop.experiment_name}_{train_loop.experiment_timestamp}.pth',
            f'{cloud_experiment_folder}/results/plots/accumulated_loss.png',
            f'{cloud_experiment_folder}/results/plots/loss.png',
            f'{cloud_experiment_folder}/results/plots/val_loss.png',
            f'{cloud_experiment_folder}/results/results_hyperParams_hist_{train_loop.experiment_name}_{train_loop.experiment_timestamp}.p',
            f'{cloud_experiment_folder}/{THIS_FILE}'
        ]

        self.assertEqual(bucket_content, expected_files)

        project_path = os.path.join(THIS_DIR, 'e2e_train_loop_example')
        if os.path.exists(project_path):
            shutil.rmtree(project_path)

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
