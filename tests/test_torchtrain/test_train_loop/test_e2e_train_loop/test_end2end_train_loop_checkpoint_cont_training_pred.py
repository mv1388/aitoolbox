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

from aitoolbox import TrainLoopCheckpointEndSave, ClassificationResultPackage, TrainLoop
from aitoolbox.torchtrain.model import TTModel
from aitoolbox.experiment.local_load.local_model_load import PyTorchLocalModelLoader
from aitoolbox.torchtrain.callbacks.model_load import ModelLoadContinueTraining
from aitoolbox.torchtrain.schedulers.basic import StepLRScheduler
from aitoolbox.torchtrain.schedulers.warmup import LinearWithWarmupScheduler

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


class TestEnd2EndTrainLoopModelOptimizerSaveReloadContinueTraining(unittest.TestCase):
    def test_e2e_ff_net_continue_training_further_1_epoch(self):
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

        model_loader = PyTorchLocalModelLoader(THIS_DIR)
        model_loader.load_model(train_loop.project_name, train_loop.experiment_name,
                                train_loop.experiment_timestamp,
                                model_save_dir='model', epoch_num=None)

        model_reloaded = FFNet()
        model_reloaded = model_loader.init_model(model_reloaded)
        optimizer_reloaded = optim.Adam(model_reloaded.parameters(), lr=0.001, betas=(0.9, 0.999))
        optimizer_reloaded = model_loader.init_optimizer(optimizer_reloaded)
        criterion_reloaded = nn.NLLLoss()

        for (orig_k, orig_state), (reload_k, reload_state) in zip(model.state_dict().items(), model_reloaded.state_dict().items()):
            self.assertEqual(orig_k, reload_k)
            self.assertEqual(orig_state.tolist(), reload_state.tolist())

        train_loop_cont = TrainLoop(
            model,
            train_dataloader, val_dataloader, test_dataloader,
            optimizer, criterion
        )
        train_loop_cont.epoch = 5
        train_loop_cont.fit(num_epochs=6)

        train_loop_reload = TrainLoop(
            model_reloaded,
            train_dataloader, val_dataloader, test_dataloader,
            optimizer_reloaded, criterion_reloaded
        )
        train_loop_reload.epoch = 5
        train_loop_reload.fit(num_epochs=6)

        train_pred, _, _ = train_loop_cont.predict_on_train_set()
        val_pred, _, _ = train_loop_cont.predict_on_validation_set()
        test_pred, _, _ = train_loop_cont.predict_on_test_set()

        train_pred_reload, _, _ = train_loop_reload.predict_on_train_set()
        val_pred_reload, _, _ = train_loop_reload.predict_on_validation_set()
        test_pred_reload, _, _ = train_loop_reload.predict_on_test_set()

        train_loss = train_loop_cont.evaluate_loss_on_train_set()
        val_loss = train_loop_cont.evaluate_loss_on_validation_set()
        test_loss = train_loop_cont.evaluate_loss_on_test_set()

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

    def test_e2e_ff_net_continue_training_further_5_epoch(self):
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

        model_loader = PyTorchLocalModelLoader(THIS_DIR)
        model_loader.load_model(train_loop.project_name, train_loop.experiment_name,
                                train_loop.experiment_timestamp,
                                model_save_dir='model', epoch_num=None)

        model_reloaded = FFNet()
        model_reloaded = model_loader.init_model(model_reloaded)
        optimizer_reloaded = optim.Adam(model_reloaded.parameters(), lr=0.001, betas=(0.9, 0.999))
        optimizer_reloaded = model_loader.init_optimizer(optimizer_reloaded)
        criterion_reloaded = nn.NLLLoss()

        for (orig_k, orig_state), (reload_k, reload_state) in zip(model.state_dict().items(), model_reloaded.state_dict().items()):
            self.assertEqual(orig_k, reload_k)
            self.assertEqual(orig_state.tolist(), reload_state.tolist())

        train_loop_cont = TrainLoop(
            model,
            train_dataloader, val_dataloader, test_dataloader,
            optimizer, criterion
        )
        train_loop_cont.epoch = 5
        train_loop_cont.fit(num_epochs=10)

        train_loop_reload = TrainLoop(
            model_reloaded,
            train_dataloader, val_dataloader, test_dataloader,
            optimizer_reloaded, criterion_reloaded
        )
        train_loop_reload.epoch = 5
        train_loop_reload.fit(num_epochs=10)

        train_pred, _, _ = train_loop_cont.predict_on_train_set()
        val_pred, _, _ = train_loop_cont.predict_on_validation_set()
        test_pred, _, _ = train_loop_cont.predict_on_test_set()

        train_pred_reload, _, _ = train_loop_reload.predict_on_train_set()
        val_pred_reload, _, _ = train_loop_reload.predict_on_validation_set()
        test_pred_reload, _, _ = train_loop_reload.predict_on_test_set()

        train_loss = train_loop_cont.evaluate_loss_on_train_set()
        val_loss = train_loop_cont.evaluate_loss_on_validation_set()
        test_loss = train_loop_cont.evaluate_loss_on_test_set()

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

    def test_e2e_ff_net_continue_training_compare_in_memory_checkpoint_end_save(self):
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

        model_loader_final = PyTorchLocalModelLoader(THIS_DIR)
        model_loader_final.load_model(train_loop.project_name, train_loop.experiment_name,
                                      train_loop.experiment_timestamp,
                                      model_save_dir='model', epoch_num=None)

        model_loader_ep5 = PyTorchLocalModelLoader(THIS_DIR)
        model_loader_ep5.load_model(train_loop.project_name, train_loop.experiment_name,
                                    train_loop.experiment_timestamp,
                                    model_save_dir='checkpoint_model', epoch_num=4)

        model_reload_final = FFNet()
        model_reload_final = model_loader_final.init_model(model_reload_final)
        optimizer_reload_final = optim.Adam(model_reload_final.parameters(), lr=0.001, betas=(0.9, 0.999))
        optimizer_reload_final = model_loader_final.init_optimizer(optimizer_reload_final)
        criterion_reload_final = nn.NLLLoss()

        model_reload_ep5 = FFNet()
        model_reload_ep5 = model_loader_ep5.init_model(model_reload_ep5)
        optimizer_reload_ep5 = optim.Adam(model_reload_ep5.parameters(), lr=0.001, betas=(0.9, 0.999))
        optimizer_reload_ep5 = model_loader_ep5.init_optimizer(optimizer_reload_ep5)
        criterion_reload_ep5 = nn.NLLLoss()

        for (orig_k, orig_state), (reload_k, reload_state) in zip(model.state_dict().items(),
                                                                  model_reload_final.state_dict().items()):
            self.assertEqual(orig_k, reload_k)
            self.assertEqual(orig_state.tolist(), reload_state.tolist())

        for (orig_k, orig_state), (reload_k, reload_state) in zip(model.state_dict().items(),
                                                                  model_reload_ep5.state_dict().items()):
            self.assertEqual(orig_k, reload_k)
            self.assertEqual(orig_state.tolist(), reload_state.tolist())

        train_loop_cont = TrainLoop(
            model,
            train_dataloader, val_dataloader, test_dataloader,
            optimizer, criterion
        )
        train_loop_cont.epoch = 5
        train_loop_cont.fit(num_epochs=6)

        train_loop_reload_final = TrainLoop(
            model_reload_final,
            train_dataloader, val_dataloader, test_dataloader,
            optimizer_reload_final, criterion_reload_final
        )
        train_loop_reload_final.epoch = 5
        train_loop_reload_final.fit(num_epochs=6)

        train_loop_reload_ep5 = TrainLoop(
            model_reload_ep5,
            train_dataloader, val_dataloader, test_dataloader,
            optimizer_reload_ep5, criterion_reload_ep5
        )
        train_loop_reload_ep5.epoch = 5
        train_loop_reload_ep5.fit(num_epochs=6)

        train_pred, _, _ = train_loop_cont.predict_on_train_set()
        val_pred, _, _ = train_loop_cont.predict_on_validation_set()
        test_pred, _, _ = train_loop_cont.predict_on_test_set()

        train_pred_reload_final, _, _ = train_loop_reload_final.predict_on_train_set()
        val_pred_reload_final, _, _ = train_loop_reload_final.predict_on_validation_set()
        test_pred_reload_final, _, _ = train_loop_reload_final.predict_on_test_set()

        train_pred_reload_ep5, _, _ = train_loop_reload_ep5.predict_on_train_set()
        val_pred_reload_ep5, _, _ = train_loop_reload_ep5.predict_on_validation_set()
        test_pred_reload_ep5, _, _ = train_loop_reload_ep5.predict_on_test_set()

        self.assertEqual(train_pred.tolist(), train_pred_reload_final.tolist())
        self.assertEqual(val_pred.tolist(), val_pred_reload_final.tolist())
        self.assertEqual(test_pred.tolist(), test_pred_reload_final.tolist())

        self.assertEqual(train_pred.tolist(), train_pred_reload_ep5.tolist())
        self.assertEqual(val_pred.tolist(), val_pred_reload_ep5.tolist())
        self.assertEqual(test_pred.tolist(), test_pred_reload_ep5.tolist())

        train_loss = train_loop_cont.evaluate_loss_on_train_set()
        val_loss = train_loop_cont.evaluate_loss_on_validation_set()
        test_loss = train_loop_cont.evaluate_loss_on_test_set()

        train_loss_reload_final = train_loop_reload_final.evaluate_loss_on_train_set()
        val_loss_reload_final = train_loop_reload_final.evaluate_loss_on_validation_set()
        test_loss_reload_final = train_loop_reload_final.evaluate_loss_on_test_set()

        train_loss_reload_ep5 = train_loop_reload_ep5.evaluate_loss_on_train_set()
        val_loss_reload_ep5 = train_loop_reload_ep5.evaluate_loss_on_validation_set()
        test_loss_reload_ep5 = train_loop_reload_ep5.evaluate_loss_on_test_set()

        self.assertEqual(train_loss, train_loss_reload_final)
        self.assertEqual(val_loss, val_loss_reload_final)
        self.assertEqual(test_loss, test_loss_reload_final)

        self.assertEqual(train_loss, train_loss_reload_ep5)
        self.assertEqual(val_loss, val_loss_reload_ep5)
        self.assertEqual(test_loss, test_loss_reload_ep5)

        project_path = os.path.join(THIS_DIR, 'e2e_train_loop_example')
        if os.path.exists(project_path):
            shutil.rmtree(project_path)

    def test_e2e_ff_net_continue_training_checkpoint_1_back_compare_in_memory_end_save(self):
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

        model_loader_final = PyTorchLocalModelLoader(THIS_DIR)
        model_loader_final.load_model(train_loop.project_name, train_loop.experiment_name,
                                      train_loop.experiment_timestamp,
                                      model_save_dir='model', epoch_num=None)
        model_reload_final = FFNet()
        model_reload_final = model_loader_final.init_model(model_reload_final)
        optimizer_reload_final = optim.Adam(model_reload_final.parameters(), lr=0.001, betas=(0.9, 0.999))
        optimizer_reload_final = model_loader_final.init_optimizer(optimizer_reload_final)
        criterion_reload_final = nn.NLLLoss()

        model_loader_ep4 = PyTorchLocalModelLoader(THIS_DIR)
        model_loader_ep4.load_model(train_loop.project_name, train_loop.experiment_name,
                                    train_loop.experiment_timestamp,
                                    model_save_dir='checkpoint_model', epoch_num=3)
        model_reload_ep4 = FFNet()
        model_reload_ep4 = model_loader_ep4.init_model(model_reload_ep4)
        optimizer_reload_ep4 = optim.Adam(model_reload_ep4.parameters(), lr=0.001, betas=(0.9, 0.999))
        optimizer_reload_ep4 = model_loader_ep4.init_optimizer(optimizer_reload_ep4)
        criterion_reload_ep4 = nn.NLLLoss()

        for (orig_k, orig_state), (reload_k, reload_state) in zip(model.state_dict().items(),
                                                                  model_reload_final.state_dict().items()):
            self.assertEqual(orig_k, reload_k)
            self.assertEqual(orig_state.tolist(), reload_state.tolist())

        for orig_state, reload_state in zip(optimizer.state_dict()['state'].values(),
                                            optimizer_reload_final.state_dict()['state'].values()):
            self.assertEqual(orig_state['step'], reload_state['step'])
            self.assertEqual(orig_state['exp_avg'].tolist(), reload_state['exp_avg'].tolist())
            self.assertEqual(orig_state['exp_avg_sq'].tolist(), reload_state['exp_avg_sq'].tolist())

        for (orig_k, orig_state), (reload_k, reload_state) in zip(model.state_dict().items(),
                                                                  model_reload_ep4.state_dict().items()):
            self.assertEqual(orig_k, reload_k)
            self.assertNotEqual(orig_state.tolist(), reload_state.tolist())

        for orig_state, reload_state in zip(optimizer.state_dict()['state'].values(),
                                            optimizer_reload_ep4.state_dict()['state'].values()):
            self.assertNotEqual(orig_state['step'], reload_state['step'])
            self.assertNotEqual(orig_state['exp_avg'].tolist(), reload_state['exp_avg'].tolist())
            self.assertNotEqual(orig_state['exp_avg_sq'].tolist(), reload_state['exp_avg_sq'].tolist())

        train_loop_reload_final = TrainLoop(
            model_reload_final,
            train_dataloader, val_dataloader, test_dataloader,
            optimizer_reload_final, criterion_reload_final
        )

        train_loop_reload_ep4 = TrainLoop(
            model_reload_ep4,
            train_dataloader, val_dataloader, test_dataloader,
            optimizer_reload_ep4, criterion_reload_ep4
        )
        train_loop_reload_ep4.epoch = 4
        train_loop_reload_ep4.fit(num_epochs=5)

        for (orig_k, orig_state), (reload_k, reload_state) in zip(model.state_dict().items(),
                                                                  model_reload_ep4.state_dict().items()):
            self.assertEqual(orig_k, reload_k)
            self.assertEqual(orig_state.tolist(), reload_state.tolist())

        for orig_state, reload_state in zip(optimizer.state_dict()['state'].values(),
                                            optimizer_reload_ep4.state_dict()['state'].values()):
            self.assertEqual(orig_state['step'], reload_state['step'])
            self.assertEqual(orig_state['exp_avg'].tolist(), reload_state['exp_avg'].tolist())
            self.assertEqual(orig_state['exp_avg_sq'].tolist(), reload_state['exp_avg_sq'].tolist())

        for (orig_k, orig_state), (reload_k, reload_state) in zip(model_reload_final.state_dict().items(),
                                                                  model_reload_ep4.state_dict().items()):
            self.assertEqual(orig_k, reload_k)
            self.assertEqual(orig_state.tolist(), reload_state.tolist())

        for orig_state, reload_state in zip(optimizer_reload_final.state_dict()['state'].values(),
                                            optimizer_reload_ep4.state_dict()['state'].values()):
            self.assertEqual(orig_state['step'], reload_state['step'])
            self.assertEqual(orig_state['exp_avg'].tolist(), reload_state['exp_avg'].tolist())
            self.assertEqual(orig_state['exp_avg_sq'].tolist(), reload_state['exp_avg_sq'].tolist())

        train_pred, _, _ = train_loop.predict_on_train_set()
        val_pred, _, _ = train_loop.predict_on_validation_set()
        test_pred, _, _ = train_loop.predict_on_test_set()

        train_pred_reload_final, _, _ = train_loop_reload_final.predict_on_train_set()
        val_pred_reload_final, _, _ = train_loop_reload_final.predict_on_validation_set()
        test_pred_reload_final, _, _ = train_loop_reload_final.predict_on_test_set()

        train_pred_reload_ep4, _, _ = train_loop_reload_ep4.predict_on_train_set()
        val_pred_reload_ep4, _, _ = train_loop_reload_ep4.predict_on_validation_set()
        test_pred_reload_ep4, _, _ = train_loop_reload_ep4.predict_on_test_set()

        self.assertEqual(train_pred.tolist(), train_pred_reload_final.tolist())
        self.assertEqual(val_pred.tolist(), val_pred_reload_final.tolist())
        self.assertEqual(test_pred.tolist(), test_pred_reload_final.tolist())

        self.assertEqual(train_pred.tolist(), train_pred_reload_ep4.tolist())
        self.assertEqual(val_pred.tolist(), val_pred_reload_ep4.tolist())
        self.assertEqual(test_pred.tolist(), test_pred_reload_ep4.tolist())

        train_loss = train_loop.evaluate_loss_on_train_set()
        val_loss = train_loop.evaluate_loss_on_validation_set()
        test_loss = train_loop.evaluate_loss_on_test_set()

        train_loss_reload_final = train_loop_reload_final.evaluate_loss_on_train_set()
        val_loss_reload_final = train_loop_reload_final.evaluate_loss_on_validation_set()
        test_loss_reload_final = train_loop_reload_final.evaluate_loss_on_test_set()

        train_loss_reload_ep4 = train_loop_reload_ep4.evaluate_loss_on_train_set()
        val_loss_reload_ep4 = train_loop_reload_ep4.evaluate_loss_on_validation_set()
        test_loss_reload_ep4 = train_loop_reload_ep4.evaluate_loss_on_test_set()

        self.assertEqual(train_loss, train_loss_reload_final)
        self.assertEqual(val_loss, val_loss_reload_final)
        self.assertEqual(test_loss, test_loss_reload_final)

        self.assertEqual(train_loss, train_loss_reload_ep4)
        self.assertEqual(val_loss, val_loss_reload_ep4)
        self.assertEqual(test_loss, test_loss_reload_ep4)

        project_path = os.path.join(THIS_DIR, 'e2e_train_loop_example')
        if os.path.exists(project_path):
            shutil.rmtree(project_path)

    def test_e2e_ff_net_continue_training_checkpoint_1_back_continue_train_1_epoch_compare_in_memory_end_save(self):
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

        model_loader_final = PyTorchLocalModelLoader(THIS_DIR)
        model_loader_final.load_model(train_loop.project_name, train_loop.experiment_name,
                                      train_loop.experiment_timestamp,
                                      model_save_dir='model', epoch_num=None)
        model_reload_final = FFNet()
        model_reload_final = model_loader_final.init_model(model_reload_final)
        optimizer_reload_final = optim.Adam(model_reload_final.parameters(), lr=0.001, betas=(0.9, 0.999))
        optimizer_reload_final = model_loader_final.init_optimizer(optimizer_reload_final)
        criterion_reload_final = nn.NLLLoss()

        model_loader_ep4 = PyTorchLocalModelLoader(THIS_DIR)
        model_loader_ep4.load_model(train_loop.project_name, train_loop.experiment_name,
                                    train_loop.experiment_timestamp,
                                    model_save_dir='checkpoint_model', epoch_num=3)
        model_reload_ep4 = FFNet()
        model_reload_ep4 = model_loader_ep4.init_model(model_reload_ep4)
        optimizer_reload_ep4 = optim.Adam(model_reload_ep4.parameters(), lr=0.001, betas=(0.9, 0.999))
        optimizer_reload_ep4 = model_loader_ep4.init_optimizer(optimizer_reload_ep4)
        criterion_reload_ep4 = nn.NLLLoss()

        for (orig_k, orig_state), (reload_k, reload_state) in zip(model.state_dict().items(),
                                                                  model_reload_final.state_dict().items()):
            self.assertEqual(orig_k, reload_k)
            self.assertEqual(orig_state.tolist(), reload_state.tolist())

        for orig_state, reload_state in zip(optimizer.state_dict()['state'].values(),
                                            optimizer_reload_final.state_dict()['state'].values()):
            self.assertEqual(orig_state['step'], reload_state['step'])
            self.assertEqual(orig_state['exp_avg'].tolist(), reload_state['exp_avg'].tolist())
            self.assertEqual(orig_state['exp_avg_sq'].tolist(), reload_state['exp_avg_sq'].tolist())

        for (orig_k, orig_state), (reload_k, reload_state) in zip(model.state_dict().items(),
                                                                  model_reload_ep4.state_dict().items()):
            self.assertEqual(orig_k, reload_k)
            self.assertNotEqual(orig_state.tolist(), reload_state.tolist())

        for orig_state, reload_state in zip(optimizer.state_dict()['state'].values(),
                                            optimizer_reload_ep4.state_dict()['state'].values()):
            self.assertNotEqual(orig_state['step'], reload_state['step'])
            self.assertNotEqual(orig_state['exp_avg'].tolist(), reload_state['exp_avg'].tolist())
            self.assertNotEqual(orig_state['exp_avg_sq'].tolist(), reload_state['exp_avg_sq'].tolist())

        train_loop_cont = TrainLoop(
            model,
            train_dataloader, val_dataloader, test_dataloader,
            optimizer, criterion
        )
        train_loop_cont.epoch = 5
        train_loop_cont.fit(num_epochs=6)

        train_loop_reload_final = TrainLoop(
            model_reload_final,
            train_dataloader, val_dataloader, test_dataloader,
            optimizer_reload_final, criterion_reload_final
        )
        train_loop_reload_final.epoch = 5
        train_loop_reload_final.fit(num_epochs=6)

        train_loop_reload_ep4 = TrainLoop(
            model_reload_ep4,
            train_dataloader, val_dataloader, test_dataloader,
            optimizer_reload_ep4, criterion_reload_ep4
        )
        train_loop_reload_ep4.epoch = 4
        train_loop_reload_ep4.fit(num_epochs=6)

        for (orig_k, orig_state), (reload_k, reload_state) in zip(model.state_dict().items(),
                                                                  model_reload_ep4.state_dict().items()):
            self.assertEqual(orig_k, reload_k)
            self.assertEqual(orig_state.tolist(), reload_state.tolist())

        for orig_state, reload_state in zip(optimizer.state_dict()['state'].values(),
                                            optimizer_reload_ep4.state_dict()['state'].values()):
            self.assertEqual(orig_state['step'], reload_state['step'])
            self.assertEqual(orig_state['exp_avg'].tolist(), reload_state['exp_avg'].tolist())
            self.assertEqual(orig_state['exp_avg_sq'].tolist(), reload_state['exp_avg_sq'].tolist())

        for (orig_k, orig_state), (reload_k, reload_state) in zip(model_reload_final.state_dict().items(),
                                                                  model_reload_ep4.state_dict().items()):
            self.assertEqual(orig_k, reload_k)
            self.assertEqual(orig_state.tolist(), reload_state.tolist())

        for orig_state, reload_state in zip(optimizer_reload_final.state_dict()['state'].values(),
                                            optimizer_reload_ep4.state_dict()['state'].values()):
            self.assertEqual(orig_state['step'], reload_state['step'])
            self.assertEqual(orig_state['exp_avg'].tolist(), reload_state['exp_avg'].tolist())
            self.assertEqual(orig_state['exp_avg_sq'].tolist(), reload_state['exp_avg_sq'].tolist())

        train_pred, _, _ = train_loop_cont.predict_on_train_set()
        val_pred, _, _ = train_loop_cont.predict_on_validation_set()
        test_pred, _, _ = train_loop_cont.predict_on_test_set()

        train_pred_reload_final, _, _ = train_loop_reload_final.predict_on_train_set()
        val_pred_reload_final, _, _ = train_loop_reload_final.predict_on_validation_set()
        test_pred_reload_final, _, _ = train_loop_reload_final.predict_on_test_set()

        train_pred_reload_ep4, _, _ = train_loop_reload_ep4.predict_on_train_set()
        val_pred_reload_ep4, _, _ = train_loop_reload_ep4.predict_on_validation_set()
        test_pred_reload_ep4, _, _ = train_loop_reload_ep4.predict_on_test_set()

        self.assertEqual(train_pred.tolist(), train_pred_reload_final.tolist())
        self.assertEqual(val_pred.tolist(), val_pred_reload_final.tolist())
        self.assertEqual(test_pred.tolist(), test_pred_reload_final.tolist())

        self.assertEqual(train_pred.tolist(), train_pred_reload_ep4.tolist())
        self.assertEqual(val_pred.tolist(), val_pred_reload_ep4.tolist())
        self.assertEqual(test_pred.tolist(), test_pred_reload_ep4.tolist())

        train_loss = train_loop_cont.evaluate_loss_on_train_set()
        val_loss = train_loop_cont.evaluate_loss_on_validation_set()
        test_loss = train_loop_cont.evaluate_loss_on_test_set()

        train_loss_reload_final = train_loop_reload_final.evaluate_loss_on_train_set()
        val_loss_reload_final = train_loop_reload_final.evaluate_loss_on_validation_set()
        test_loss_reload_final = train_loop_reload_final.evaluate_loss_on_test_set()

        train_loss_reload_ep4 = train_loop_reload_ep4.evaluate_loss_on_train_set()
        val_loss_reload_ep4 = train_loop_reload_ep4.evaluate_loss_on_validation_set()
        test_loss_reload_ep4 = train_loop_reload_ep4.evaluate_loss_on_test_set()

        self.assertEqual(train_loss, train_loss_reload_final)
        self.assertEqual(val_loss, val_loss_reload_final)
        self.assertEqual(test_loss, test_loss_reload_final)

        self.assertEqual(train_loss, train_loss_reload_ep4)
        self.assertEqual(val_loss, val_loss_reload_ep4)
        self.assertEqual(test_loss, test_loss_reload_ep4)

        project_path = os.path.join(THIS_DIR, 'e2e_train_loop_example')
        if os.path.exists(project_path):
            shutil.rmtree(project_path)

    def test_e2e_ff_net_continue_training_checkpoint_1_back_continue_train_5_epoch_compare_in_memory_end_save(self):
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

        model_loader_final = PyTorchLocalModelLoader(THIS_DIR)
        model_loader_final.load_model(train_loop.project_name, train_loop.experiment_name,
                                      train_loop.experiment_timestamp,
                                      model_save_dir='model', epoch_num=None)
        model_reload_final = FFNet()
        model_reload_final = model_loader_final.init_model(model_reload_final)
        optimizer_reload_final = optim.Adam(model_reload_final.parameters(), lr=0.001, betas=(0.9, 0.999))
        optimizer_reload_final = model_loader_final.init_optimizer(optimizer_reload_final)
        criterion_reload_final = nn.NLLLoss()

        model_loader_ep4 = PyTorchLocalModelLoader(THIS_DIR)
        model_loader_ep4.load_model(train_loop.project_name, train_loop.experiment_name,
                                    train_loop.experiment_timestamp,
                                    model_save_dir='checkpoint_model', epoch_num=3)
        model_reload_ep4 = FFNet()
        model_reload_ep4 = model_loader_ep4.init_model(model_reload_ep4)
        optimizer_reload_ep4 = optim.Adam(model_reload_ep4.parameters(), lr=0.001, betas=(0.9, 0.999))
        optimizer_reload_ep4 = model_loader_ep4.init_optimizer(optimizer_reload_ep4)
        criterion_reload_ep4 = nn.NLLLoss()

        for (orig_k, orig_state), (reload_k, reload_state) in zip(model.state_dict().items(),
                                                                  model_reload_final.state_dict().items()):
            self.assertEqual(orig_k, reload_k)
            self.assertEqual(orig_state.tolist(), reload_state.tolist())

        for orig_state, reload_state in zip(optimizer.state_dict()['state'].values(),
                                            optimizer_reload_final.state_dict()['state'].values()):
            self.assertEqual(orig_state['step'], reload_state['step'])
            self.assertEqual(orig_state['exp_avg'].tolist(), reload_state['exp_avg'].tolist())
            self.assertEqual(orig_state['exp_avg_sq'].tolist(), reload_state['exp_avg_sq'].tolist())

        for (orig_k, orig_state), (reload_k, reload_state) in zip(model.state_dict().items(),
                                                                  model_reload_ep4.state_dict().items()):
            self.assertEqual(orig_k, reload_k)
            self.assertNotEqual(orig_state.tolist(), reload_state.tolist())

        for orig_state, reload_state in zip(optimizer.state_dict()['state'].values(),
                                            optimizer_reload_ep4.state_dict()['state'].values()):
            self.assertNotEqual(orig_state['step'], reload_state['step'])
            self.assertNotEqual(orig_state['exp_avg'].tolist(), reload_state['exp_avg'].tolist())
            self.assertNotEqual(orig_state['exp_avg_sq'].tolist(), reload_state['exp_avg_sq'].tolist())

        train_loop_cont = TrainLoop(
            model,
            train_dataloader, val_dataloader, test_dataloader,
            optimizer, criterion
        )
        train_loop_cont.epoch = 5
        train_loop_cont.fit(num_epochs=10)

        train_loop_reload_final = TrainLoop(
            model_reload_final,
            train_dataloader, val_dataloader, test_dataloader,
            optimizer_reload_final, criterion_reload_final
        )
        train_loop_reload_final.epoch = 5
        train_loop_reload_final.fit(num_epochs=10)

        train_loop_reload_ep4 = TrainLoop(
            model_reload_ep4,
            train_dataloader, val_dataloader, test_dataloader,
            optimizer_reload_ep4, criterion_reload_ep4
        )
        train_loop_reload_ep4.epoch = 4
        train_loop_reload_ep4.fit(num_epochs=10)

        for (orig_k, orig_state), (reload_k, reload_state) in zip(model.state_dict().items(),
                                                                  model_reload_ep4.state_dict().items()):
            self.assertEqual(orig_k, reload_k)
            self.assertEqual(orig_state.tolist(), reload_state.tolist())

        for orig_state, reload_state in zip(optimizer.state_dict()['state'].values(),
                                            optimizer_reload_ep4.state_dict()['state'].values()):
            self.assertEqual(orig_state['step'], reload_state['step'])
            self.assertEqual(orig_state['exp_avg'].tolist(), reload_state['exp_avg'].tolist())
            self.assertEqual(orig_state['exp_avg_sq'].tolist(), reload_state['exp_avg_sq'].tolist())

        for (orig_k, orig_state), (reload_k, reload_state) in zip(model_reload_final.state_dict().items(),
                                                                  model_reload_ep4.state_dict().items()):
            self.assertEqual(orig_k, reload_k)
            self.assertEqual(orig_state.tolist(), reload_state.tolist())

        for orig_state, reload_state in zip(optimizer_reload_final.state_dict()['state'].values(),
                                            optimizer_reload_ep4.state_dict()['state'].values()):
            self.assertEqual(orig_state['step'], reload_state['step'])
            self.assertEqual(orig_state['exp_avg'].tolist(), reload_state['exp_avg'].tolist())
            self.assertEqual(orig_state['exp_avg_sq'].tolist(), reload_state['exp_avg_sq'].tolist())

        train_pred, _, _ = train_loop_cont.predict_on_train_set()
        val_pred, _, _ = train_loop_cont.predict_on_validation_set()
        test_pred, _, _ = train_loop_cont.predict_on_test_set()

        train_pred_reload_final, _, _ = train_loop_reload_final.predict_on_train_set()
        val_pred_reload_final, _, _ = train_loop_reload_final.predict_on_validation_set()
        test_pred_reload_final, _, _ = train_loop_reload_final.predict_on_test_set()

        train_pred_reload_ep4, _, _ = train_loop_reload_ep4.predict_on_train_set()
        val_pred_reload_ep4, _, _ = train_loop_reload_ep4.predict_on_validation_set()
        test_pred_reload_ep4, _, _ = train_loop_reload_ep4.predict_on_test_set()

        self.assertEqual(train_pred.tolist(), train_pred_reload_final.tolist())
        self.assertEqual(val_pred.tolist(), val_pred_reload_final.tolist())
        self.assertEqual(test_pred.tolist(), test_pred_reload_final.tolist())

        self.assertEqual(train_pred.tolist(), train_pred_reload_ep4.tolist())
        self.assertEqual(val_pred.tolist(), val_pred_reload_ep4.tolist())
        self.assertEqual(test_pred.tolist(), test_pred_reload_ep4.tolist())

        train_loss = train_loop_cont.evaluate_loss_on_train_set()
        val_loss = train_loop_cont.evaluate_loss_on_validation_set()
        test_loss = train_loop_cont.evaluate_loss_on_test_set()

        train_loss_reload_final = train_loop_reload_final.evaluate_loss_on_train_set()
        val_loss_reload_final = train_loop_reload_final.evaluate_loss_on_validation_set()
        test_loss_reload_final = train_loop_reload_final.evaluate_loss_on_test_set()

        train_loss_reload_ep4 = train_loop_reload_ep4.evaluate_loss_on_train_set()
        val_loss_reload_ep4 = train_loop_reload_ep4.evaluate_loss_on_validation_set()
        test_loss_reload_ep4 = train_loop_reload_ep4.evaluate_loss_on_test_set()

        self.assertEqual(train_loss, train_loss_reload_final)
        self.assertEqual(val_loss, val_loss_reload_final)
        self.assertEqual(test_loss, test_loss_reload_final)

        self.assertEqual(train_loss, train_loss_reload_ep4)
        self.assertEqual(val_loss, val_loss_reload_ep4)
        self.assertEqual(test_loss, test_loss_reload_ep4)

        project_path = os.path.join(THIS_DIR, 'e2e_train_loop_example')
        if os.path.exists(project_path):
            shutil.rmtree(project_path)

    def test_e2e_ff_net_callback_continue_train_checkpoint_1_back_cont_train_5_epoch_compare_in_memory_end_save(self):
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

        model_reload_final = FFNet()
        optimizer_reload_final = optim.Adam(model_reload_final.parameters(), lr=0.001, betas=(0.9, 0.999))
        criterion_reload_final = nn.NLLLoss()

        model_reload_ep4 = FFNet()
        optimizer_reload_ep4 = optim.Adam(model_reload_ep4.parameters(), lr=0.001, betas=(0.9, 0.999))
        criterion_reload_ep4 = nn.NLLLoss()

        train_loop_cont = TrainLoop(
            model,
            train_dataloader, val_dataloader, test_dataloader,
            optimizer, criterion
        )
        train_loop_cont.epoch = 5
        train_loop_cont.fit(num_epochs=10)

        train_loop_reload_final = TrainLoop(
            model_reload_final,
            train_dataloader, val_dataloader, test_dataloader,
            optimizer_reload_final, criterion_reload_final
        )
        train_loop_reload_final.epoch = 5
        train_loop_reload_final.fit(num_epochs=10, callbacks=[
            ModelLoadContinueTraining(train_loop.experiment_timestamp, saved_model_dir='model', epoch_num=None,
                                      project_name=train_loop.project_name, experiment_name=train_loop.experiment_name,
                                      local_model_result_folder_path=THIS_DIR, cloud_save_mode='local')
        ])

        train_loop_reload_ep4 = TrainLoop(
            model_reload_ep4,
            train_dataloader, val_dataloader, test_dataloader,
            optimizer_reload_ep4, criterion_reload_ep4
        )
        train_loop_reload_ep4.epoch = 4
        train_loop_reload_ep4.fit(num_epochs=10, callbacks=[
            ModelLoadContinueTraining(train_loop.experiment_timestamp, saved_model_dir='checkpoint_model', epoch_num=3,
                                      project_name=train_loop.project_name, experiment_name=train_loop.experiment_name,
                                      local_model_result_folder_path=THIS_DIR, cloud_save_mode='local')
        ])

        for (orig_k, orig_state), (reload_k, reload_state) in zip(model.state_dict().items(),
                                                                  model_reload_ep4.state_dict().items()):
            self.assertEqual(orig_k, reload_k)
            self.assertEqual(orig_state.tolist(), reload_state.tolist())

        for orig_state, reload_state in zip(optimizer.state_dict()['state'].values(),
                                            optimizer_reload_ep4.state_dict()['state'].values()):
            self.assertEqual(orig_state['step'], reload_state['step'])
            self.assertEqual(orig_state['exp_avg'].tolist(), reload_state['exp_avg'].tolist())
            self.assertEqual(orig_state['exp_avg_sq'].tolist(), reload_state['exp_avg_sq'].tolist())

        for (orig_k, orig_state), (reload_k, reload_state) in zip(model_reload_final.state_dict().items(),
                                                                  model_reload_ep4.state_dict().items()):
            self.assertEqual(orig_k, reload_k)
            self.assertEqual(orig_state.tolist(), reload_state.tolist())

        for orig_state, reload_state in zip(optimizer_reload_final.state_dict()['state'].values(),
                                            optimizer_reload_ep4.state_dict()['state'].values()):
            self.assertEqual(orig_state['step'], reload_state['step'])
            self.assertEqual(orig_state['exp_avg'].tolist(), reload_state['exp_avg'].tolist())
            self.assertEqual(orig_state['exp_avg_sq'].tolist(), reload_state['exp_avg_sq'].tolist())

        train_pred, _, _ = train_loop_cont.predict_on_train_set()
        val_pred, _, _ = train_loop_cont.predict_on_validation_set()
        test_pred, _, _ = train_loop_cont.predict_on_test_set()

        train_pred_reload_final, _, _ = train_loop_reload_final.predict_on_train_set()
        val_pred_reload_final, _, _ = train_loop_reload_final.predict_on_validation_set()
        test_pred_reload_final, _, _ = train_loop_reload_final.predict_on_test_set()

        train_pred_reload_ep4, _, _ = train_loop_reload_ep4.predict_on_train_set()
        val_pred_reload_ep4, _, _ = train_loop_reload_ep4.predict_on_validation_set()
        test_pred_reload_ep4, _, _ = train_loop_reload_ep4.predict_on_test_set()

        self.assertEqual(train_pred.tolist(), train_pred_reload_final.tolist())
        self.assertEqual(val_pred.tolist(), val_pred_reload_final.tolist())
        self.assertEqual(test_pred.tolist(), test_pred_reload_final.tolist())

        self.assertEqual(train_pred.tolist(), train_pred_reload_ep4.tolist())
        self.assertEqual(val_pred.tolist(), val_pred_reload_ep4.tolist())
        self.assertEqual(test_pred.tolist(), test_pred_reload_ep4.tolist())

        train_loss = train_loop_cont.evaluate_loss_on_train_set()
        val_loss = train_loop_cont.evaluate_loss_on_validation_set()
        test_loss = train_loop_cont.evaluate_loss_on_test_set()

        train_loss_reload_final = train_loop_reload_final.evaluate_loss_on_train_set()
        val_loss_reload_final = train_loop_reload_final.evaluate_loss_on_validation_set()
        test_loss_reload_final = train_loop_reload_final.evaluate_loss_on_test_set()

        train_loss_reload_ep4 = train_loop_reload_ep4.evaluate_loss_on_train_set()
        val_loss_reload_ep4 = train_loop_reload_ep4.evaluate_loss_on_validation_set()
        test_loss_reload_ep4 = train_loop_reload_ep4.evaluate_loss_on_test_set()

        self.assertEqual(train_loss, train_loss_reload_final)
        self.assertEqual(val_loss, val_loss_reload_final)
        self.assertEqual(test_loss, test_loss_reload_final)

        self.assertEqual(train_loss, train_loss_reload_ep4)
        self.assertEqual(val_loss, val_loss_reload_ep4)
        self.assertEqual(test_loss, test_loss_reload_ep4)

        project_path = os.path.join(THIS_DIR, 'e2e_train_loop_example')
        if os.path.exists(project_path):
            shutil.rmtree(project_path)

    def test_e2e_ff_net_scheduler_callback_continue_training_5_epoch_compare(self):
        self.set_seeds()
        batch_size = 100
        num_epochs = 10

        train_dataset = TensorDataset(torch.randn(1000, 50), torch.randint(low=0, high=10, size=(1000,)))
        val_dataset = TensorDataset(torch.randn(300, 50), torch.randint(low=0, high=10, size=(300,)))
        test_dataset = TensorDataset(torch.randn(300, 50), torch.randint(low=0, high=10, size=(300,)))

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

        scheduler_cb = [
            LinearWithWarmupScheduler(num_warmup_steps=1, num_training_steps=len(train_dataloader) * num_epochs)
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
        train_loop.fit(num_epochs=num_epochs, callbacks=scheduler_cb)

        model_reload_ep4 = FFNet()
        optimizer_reload_ep4 = optim.Adam(model_reload_ep4.parameters(), lr=0.001, betas=(0.9, 0.999))
        criterion_reload_ep4 = nn.NLLLoss()

        scheduler_cb_reloaded = [
            LinearWithWarmupScheduler(num_warmup_steps=1, num_training_steps=len(train_dataloader) * num_epochs,
                                      last_epoch=(len(train_dataloader) * 4) - 1)
        ]

        train_loop_reload_ep4 = TrainLoop(
            model_reload_ep4,
            train_dataloader, val_dataloader, test_dataloader,
            optimizer_reload_ep4, criterion_reload_ep4
        )
        train_loop_reload_ep4.epoch = 4
        train_loop_reload_ep4.fit(num_epochs=num_epochs, callbacks=[
            ModelLoadContinueTraining(train_loop.experiment_timestamp, saved_model_dir='checkpoint_model', epoch_num=3,
                                      project_name=train_loop.project_name, experiment_name=train_loop.experiment_name,
                                      local_model_result_folder_path=THIS_DIR, cloud_save_mode='local')
        ] + scheduler_cb_reloaded)

        for (orig_k, orig_state), (reload_k, reload_state) in zip(model.state_dict().items(),
                                                                  model_reload_ep4.state_dict().items()):
            self.assertEqual(orig_k, reload_k)
            self.assertEqual(orig_state.tolist(), reload_state.tolist())

        self.check_loaded_representation(optimizer_reload_ep4, scheduler_cb_reloaded, optimizer, scheduler_cb)

        train_pred, _, _ = train_loop.predict_on_train_set()
        val_pred, _, _ = train_loop.predict_on_validation_set()
        test_pred, _, _ = train_loop.predict_on_test_set()

        train_pred_reload_ep4, _, _ = train_loop_reload_ep4.predict_on_train_set()
        val_pred_reload_ep4, _, _ = train_loop_reload_ep4.predict_on_validation_set()
        test_pred_reload_ep4, _, _ = train_loop_reload_ep4.predict_on_test_set()

        self.assertEqual(train_pred.tolist(), train_pred_reload_ep4.tolist())
        self.assertEqual(val_pred.tolist(), val_pred_reload_ep4.tolist())
        self.assertEqual(test_pred.tolist(), test_pred_reload_ep4.tolist())

        train_loss = train_loop.evaluate_loss_on_train_set()
        val_loss = train_loop.evaluate_loss_on_validation_set()
        test_loss = train_loop.evaluate_loss_on_test_set()

        train_loss_reload_ep4 = train_loop_reload_ep4.evaluate_loss_on_train_set()
        val_loss_reload_ep4 = train_loop_reload_ep4.evaluate_loss_on_validation_set()
        test_loss_reload_ep4 = train_loop_reload_ep4.evaluate_loss_on_test_set()

        self.assertEqual(train_loss, train_loss_reload_ep4)
        self.assertEqual(val_loss, val_loss_reload_ep4)
        self.assertEqual(test_loss, test_loss_reload_ep4)

        project_path = os.path.join(THIS_DIR, 'e2e_train_loop_example')
        if os.path.exists(project_path):
            shutil.rmtree(project_path)

    def check_loaded_representation(self, optimizer_reload, scheduler_reload_cb, optimizer, scheduler_cb):
        self.assertEqual(optimizer_reload.state_dict().keys(), optimizer.state_dict().keys())

        loaded_optimizer_state = optimizer_reload.state_dict()['state']
        for state_idx in range(len(loaded_optimizer_state)):
            opti_state = optimizer.state_dict()['state'][state_idx]
            loaded_state = loaded_optimizer_state[state_idx]

            self.assertEqual(opti_state.keys(), loaded_state.keys())
            self.assertEqual(opti_state['step'], loaded_state['step'])
            self.assertEqual(opti_state['exp_avg'].tolist(), loaded_state['exp_avg'].tolist())
            self.assertEqual(opti_state['exp_avg_sq'].tolist(), loaded_state['exp_avg_sq'].tolist())

        self.assertEqual(
            optimizer_reload.state_dict()['param_groups'],
            optimizer.state_dict()['param_groups']
        )

        for scheduler_idx in range(len(scheduler_cb)):
            self.assertEqual(
                scheduler_reload_cb[scheduler_idx].state_dict(),
                scheduler_cb[scheduler_idx].state_dict()
            )

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
