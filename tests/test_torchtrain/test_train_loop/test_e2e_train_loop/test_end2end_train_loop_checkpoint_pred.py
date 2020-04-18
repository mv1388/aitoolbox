import unittest

import os
import shutil
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import TensorDataset

from aitoolbox.torchtrain.train_loop import TrainLoopCheckpointEndSave, TrainLoop
from aitoolbox.torchtrain.model import TTModel
from aitoolbox.experiment.result_package.basic_packages import ClassificationResultPackage
from aitoolbox.torchtrain.model_predict import PyTorchModelPredictor
from aitoolbox.experiment.local_load.local_model_load import PyTorchLocalModelLoader
from aitoolbox.torchtrain.callbacks.model_load import ModelLoadContinueTraining

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


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


class TestEnd2EndTrainLoopModelSaveReloadPrediction(unittest.TestCase):
    def test_e2e_ff_net_reload_train_loop_prediction(self):
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

        train_pred, _, _ = train_loop.predict_on_train_set()
        val_pred, _, _ = train_loop.predict_on_validation_set()
        test_pred, _, _ = train_loop.predict_on_test_set()

        train_loss = train_loop.evaluate_loss_on_train_set()
        val_loss = train_loop.evaluate_loss_on_validation_set()
        test_loss = train_loop.evaluate_loss_on_test_set()

        model_state = torch.load(
            os.path.join(THIS_DIR, train_loop.project_name,
                         f'{train_loop.experiment_name}_{train_loop.experiment_timestamp}',
                         'model',
                         f'model_{train_loop.experiment_name}_{train_loop.experiment_timestamp}.pth')
        )

        model_reloaded = FFNet()
        model_reloaded.load_state_dict(model_state['model_state_dict'])

        train_loop_reload = TrainLoop(
            model_reloaded,
            train_dataloader, val_dataloader, test_dataloader,
            None, criterion
        )

        train_pred_reload, _, _ = train_loop_reload.predict_on_train_set()
        val_pred_reload, _, _ = train_loop_reload.predict_on_validation_set()
        test_pred_reload, _, _ = train_loop_reload.predict_on_test_set()

        train_loss_reload = train_loop_reload.evaluate_loss_on_train_set()
        val_loss_reload = train_loop_reload.evaluate_loss_on_validation_set()
        test_loss_reload = train_loop_reload.evaluate_loss_on_test_set()

        self.assertEqual(train_pred.tolist(), train_pred_reload.tolist())
        self.assertEqual(val_pred.tolist(), val_pred_reload.tolist())
        self.assertEqual(test_pred.tolist(), test_pred_reload.tolist())

        self.assertEqual(train_loss, train_loss_reload)
        self.assertEqual(val_loss, val_loss_reload)
        self.assertEqual(test_loss, test_loss_reload)

        project_path = os.path.join(THIS_DIR, 'e2e_train_loop_example')
        if os.path.exists(project_path):
            shutil.rmtree(project_path)

    def test_e2e_ff_net_reload_model_predictor_prediction(self):
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

        train_pred, _, _ = train_loop.predict_on_train_set()
        val_pred, _, _ = train_loop.predict_on_validation_set()
        test_pred, _, _ = train_loop.predict_on_test_set()

        train_loss = train_loop.evaluate_loss_on_train_set()
        val_loss = train_loop.evaluate_loss_on_validation_set()
        test_loss = train_loop.evaluate_loss_on_test_set()

        model_state = torch.load(
            os.path.join(THIS_DIR, train_loop.project_name,
                         f'{train_loop.experiment_name}_{train_loop.experiment_timestamp}',
                         'model',
                         f'model_{train_loop.experiment_name}_{train_loop.experiment_timestamp}.pth')
        )

        model_reloaded = FFNet()
        model_reloaded.load_state_dict(model_state['model_state_dict'])

        train_model_predictor = PyTorchModelPredictor(model_reloaded, train_dataloader)
        val_model_predictor = PyTorchModelPredictor(model_reloaded, val_dataloader)
        test_model_predictor = PyTorchModelPredictor(model_reloaded, test_dataloader)

        train_pred_reload, _, _ = train_model_predictor.model_predict()
        val_pred_reload, _, _ = val_model_predictor.model_predict()
        test_pred_reload, _, _ = test_model_predictor.model_predict()

        train_loss_reload = train_model_predictor.model_get_loss(criterion)
        val_loss_reload = val_model_predictor.model_get_loss(criterion)
        test_loss_reload = test_model_predictor.model_get_loss(criterion)

        self.assertEqual(train_pred.tolist(), train_pred_reload.tolist())
        self.assertEqual(val_pred.tolist(), val_pred_reload.tolist())
        self.assertEqual(test_pred.tolist(), test_pred_reload.tolist())

        self.assertEqual(train_loss, train_loss_reload)
        self.assertEqual(val_loss, val_loss_reload)
        self.assertEqual(test_loss, test_loss_reload)

        project_path = os.path.join(THIS_DIR, 'e2e_train_loop_example')
        if os.path.exists(project_path):
            shutil.rmtree(project_path)

    def test_e2e_ff_net_local_model_loader_predict(self):
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

        train_pred, _, _ = train_loop.predict_on_train_set()
        val_pred, _, _ = train_loop.predict_on_validation_set()
        test_pred, _, _ = train_loop.predict_on_test_set()

        train_loss = train_loop.evaluate_loss_on_train_set()
        val_loss = train_loop.evaluate_loss_on_validation_set()
        test_loss = train_loop.evaluate_loss_on_test_set()

        model_loader = PyTorchLocalModelLoader(THIS_DIR)
        model_loader.load_model(train_loop.project_name, train_loop.experiment_name,
                                train_loop.experiment_timestamp, model_save_dir='model')

        model_reloaded = FFNet()
        model_reloaded = model_loader.init_model(model_reloaded)

        train_loop_reload = TrainLoop(
            model_reloaded,
            train_dataloader, val_dataloader, test_dataloader,
            None, criterion
        )

        train_pred_reload, _, _ = train_loop_reload.predict_on_train_set()
        val_pred_reload, _, _ = train_loop_reload.predict_on_validation_set()
        test_pred_reload, _, _ = train_loop_reload.predict_on_test_set()

        train_loss_reload = train_loop_reload.evaluate_loss_on_train_set()
        val_loss_reload = train_loop_reload.evaluate_loss_on_validation_set()
        test_loss_reload = train_loop_reload.evaluate_loss_on_test_set()

        self.assertEqual(train_pred.tolist(), train_pred_reload.tolist())
        self.assertEqual(val_pred.tolist(), val_pred_reload.tolist())
        self.assertEqual(test_pred.tolist(), test_pred_reload.tolist())

        self.assertEqual(train_loss, train_loss_reload)
        self.assertEqual(val_loss, val_loss_reload)
        self.assertEqual(test_loss, test_loss_reload)

        project_path = os.path.join(THIS_DIR, 'e2e_train_loop_example')
        if os.path.exists(project_path):
            shutil.rmtree(project_path)

    def test_e2e_ff_net_local_model_load_callback_predict(self):
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

        train_pred, _, _ = train_loop.predict_on_train_set()
        val_pred, _, _ = train_loop.predict_on_validation_set()
        test_pred, _, _ = train_loop.predict_on_test_set()

        train_loss = train_loop.evaluate_loss_on_train_set()
        val_loss = train_loop.evaluate_loss_on_validation_set()
        test_loss = train_loop.evaluate_loss_on_test_set()

        model_reloaded = FFNet()
        optimizer_reloaded = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

        train_loop_reload = TrainLoop(
            model_reloaded,
            train_dataloader, val_dataloader, test_dataloader,
            optimizer_reloaded, criterion
        )
        train_loop_reload.callbacks_handler.register_callbacks([
            ModelLoadContinueTraining(train_loop.experiment_timestamp, saved_model_dir='model',
                                      project_name=train_loop.project_name, experiment_name=train_loop.experiment_name,
                                      local_model_result_folder_path=THIS_DIR,
                                      cloud_save_mode='local')
        ])

        train_pred_reload, _, _ = train_loop_reload.predict_on_train_set()
        val_pred_reload, _, _ = train_loop_reload.predict_on_validation_set()
        test_pred_reload, _, _ = train_loop_reload.predict_on_test_set()

        train_loss_reload = train_loop_reload.evaluate_loss_on_train_set()
        val_loss_reload = train_loop_reload.evaluate_loss_on_validation_set()
        test_loss_reload = train_loop_reload.evaluate_loss_on_test_set()

        self.assertEqual(train_pred.tolist(), train_pred_reload.tolist())
        self.assertEqual(val_pred.tolist(), val_pred_reload.tolist())
        self.assertEqual(test_pred.tolist(), test_pred_reload.tolist())

        self.assertEqual(train_loss, train_loss_reload)
        self.assertEqual(val_loss, val_loss_reload)
        self.assertEqual(test_loss, test_loss_reload)

        project_path = os.path.join(THIS_DIR, 'e2e_train_loop_example')
        if os.path.exists(project_path):
            shutil.rmtree(project_path)

    def test_e2e_ff_net_local_model_load_callback_experiment_detail_infer_predict(self):
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

        train_pred, _, _ = train_loop.predict_on_train_set()
        val_pred, _, _ = train_loop.predict_on_validation_set()
        test_pred, _, _ = train_loop.predict_on_test_set()

        train_loss = train_loop.evaluate_loss_on_train_set()
        val_loss = train_loop.evaluate_loss_on_validation_set()
        test_loss = train_loop.evaluate_loss_on_test_set()

        model_reloaded = FFNet()
        optimizer_reloaded = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

        train_loop_reload = TrainLoopCheckpointEndSave(
            model_reloaded,
            train_dataloader, val_dataloader, test_dataloader,
            optimizer_reloaded, criterion,
            project_name=train_loop.project_name, experiment_name=train_loop.experiment_name,
            local_model_result_folder_path=THIS_DIR,
            hyperparams={'batch_size': batch_size},
            val_result_package=ClassificationResultPackage(), test_result_package=ClassificationResultPackage(),
            cloud_save_mode=None
        )
        train_loop_reload.callbacks_handler.register_callbacks([
            ModelLoadContinueTraining(train_loop.experiment_timestamp, saved_model_dir='model')
        ])

        train_pred_reload, _, _ = train_loop_reload.predict_on_train_set()
        val_pred_reload, _, _ = train_loop_reload.predict_on_validation_set()
        test_pred_reload, _, _ = train_loop_reload.predict_on_test_set()

        train_loss_reload = train_loop_reload.evaluate_loss_on_train_set()
        val_loss_reload = train_loop_reload.evaluate_loss_on_validation_set()
        test_loss_reload = train_loop_reload.evaluate_loss_on_test_set()

        self.assertEqual(train_pred.tolist(), train_pred_reload.tolist())
        self.assertEqual(val_pred.tolist(), val_pred_reload.tolist())
        self.assertEqual(test_pred.tolist(), test_pred_reload.tolist())

        self.assertEqual(train_loss, train_loss_reload)
        self.assertEqual(val_loss, val_loss_reload)
        self.assertEqual(test_loss, test_loss_reload)

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
