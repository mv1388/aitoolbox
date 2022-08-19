import unittest

import os
import shutil
import random
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AdamW
from torch.cuda.amp import autocast, GradScaler

import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from datasets import load_dataset

from aitoolbox import TrainLoopCheckpointEndSave, TTModel, ModelPerformanceEvaluation, ModelPerformancePrintReport, \
    ModelTrainHistoryPlot, ModelTrainHistoryFileWriter, BinaryClassificationResultPackage
from tests_gpu.test_multi_gpu.ddp_utils import DDPPredictionSave

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

"""
Training taken from: 
    https://pytorch-ignite.ai/tutorials/beginner/02-transformers-text-classification/
    https://colab.research.google.com/github/pytorch-ignite/pytorch-ignite.ai/blob/gh-pages/tutorials/beginner/02-transformers-text-classification.ipynb
"""


class BERTModel(TTModel):
    def __init__(self):
        super().__init__()
        self.hf_model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)

    def forward(self, **kwargs):
        return self.hf_model(**kwargs)

    def get_loss(self, batch_data, criterion, device):
        batch = {k: v.to(device) for k, v in batch_data.items()}
        outputs = self(**batch)
        loss = outputs.loss
        return loss

    def get_predictions(self, batch_data, device):
        batch = {k: v.to(device) for k, v in batch_data.items()}
        outputs = self(**batch)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        return predictions.cpu(), batch["labels"].cpu(), {}


class TestAMPDDPMultiGPUIMDBBERTExperimentTrack(unittest.TestCase):
    def test_amp_ddp_trainloop_core_pytorch_compare(self):
        os.mkdir(f'{THIS_DIR}/ddp_bert_save')

        val_loss_tl, y_pred_tl, y_true_tl = self.train_eval_trainloop(ds_subset_size=1000, num_epochs=2)
        val_loss_pt, y_pred_pt, y_true_pt = self.train_eval_core_pytorch(ds_subset_size=1000, num_epochs=2)

        self.assertEqual(val_loss_tl, val_loss_pt)
        self.assertEqual(y_pred_tl, y_pred_pt)
        self.assertEqual(y_true_tl, y_true_pt)

        project_path = os.path.join(THIS_DIR, 'ddp_bert_save')
        if os.path.exists(project_path):
            shutil.rmtree(project_path)
        project_path = os.path.join(THIS_DIR, 'tl_full_experiment_tracking')
        if os.path.exists(project_path):
            shutil.rmtree(project_path)

    def train_eval_trainloop(self, ds_subset_size, num_epochs):
        self.set_seeds()

        train_data, test_data = self.get_data_sets(ds_subset_size=ds_subset_size)

        train_loader = DataLoader(train_data, shuffle=True, batch_size=8)
        val_loader = DataLoader(test_data, batch_size=8)

        model = BERTModel()
        optimizer = AdamW(model.parameters(), lr=5e-5)

        callbacks = [
            ModelPerformanceEvaluation(BinaryClassificationResultPackage(), {},
                                       on_train_data=True, on_val_data=True),
            ModelPerformancePrintReport(['train_Accuracy', 'val_Accuracy']),
            ModelTrainHistoryPlot(),
            ModelTrainHistoryFileWriter(),
            DDPPredictionSave(dir_path=f'{THIS_DIR}/ddp_bert_save',
                              file_name='tl_ddp_predictions.p')
        ]

        print('Starting train loop')
        tl = TrainLoopCheckpointEndSave(
            model,
            train_loader, val_loader, None,
            optimizer, None,
            project_name='tl_full_experiment_tracking', experiment_name='tutorial_example',
            local_model_result_folder_path=THIS_DIR,
            hyperparams={},
            val_result_package=BinaryClassificationResultPackage(),
            cloud_save_mode=None,
            gpu_mode='ddp',
            use_amp=True
        )
        self.assertEqual(tl.device.type, "cuda")

        tl.fit(num_epochs=num_epochs, callbacks=callbacks)

        with open(f'{THIS_DIR}/ddp_bert_save/tl_ddp_predictions.p', 'rb') as f:
            val_loss, y_pred, y_true = pickle.load(f)

        return val_loss, y_pred, y_true

    def train_eval_core_pytorch(self, ds_subset_size, num_epochs):
        self.set_seeds()

        train_data, test_data = self.get_data_sets(ds_subset_size=ds_subset_size)

        train_loader = DataLoader(train_data, shuffle=True, batch_size=8)
        val_loader = DataLoader(test_data, batch_size=8)

        model_pt = BERTModel()
        optimizer_pt = AdamW(model_pt.parameters(), lr=5e-5)

        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '8888'

        print('Starting the manual DDP training')

        mp.spawn(
            self.manual_ddp_training,
            args=(num_epochs, model_pt, optimizer_pt, train_loader, val_loader),
            nprocs=torch.cuda.device_count()
        )

        val_loss, y_pred, y_true = [], [], []
        for idx in range(torch.cuda.device_count()):
            with open(f'{THIS_DIR}/ddp_bert_save/pt_ddp_predictions_{idx}.p', 'rb') as f:
                val_loss_f, y_pred_f, y_true_f = pickle.load(f)
                val_loss += val_loss_f
                y_pred += y_pred_f
                y_true += y_true_f

        val_loss = np.mean(val_loss)
        return val_loss, y_pred, y_true

    @staticmethod
    def manual_ddp_training(gpu, num_epochs, model_pt, optimizer_pt, train_loader, val_loader):
        rank = gpu
        dist.init_process_group(backend='nccl', init_method='env://', world_size=torch.cuda.device_count(), rank=rank)
        torch.manual_seed(0)
        torch.cuda.set_device(gpu)
        device = torch.device(f"cuda:{gpu}")

        train_sampler = DistributedSampler(dataset=train_loader.dataset, shuffle=True,
                                           num_replicas=torch.cuda.device_count(), rank=rank)
        val_sampler = DistributedSampler(dataset=val_loader.dataset, shuffle=False,
                                         num_replicas=torch.cuda.device_count(), rank=rank)
        train_loader = DataLoader(train_loader.dataset, batch_size=8, sampler=train_sampler)
        val_loader = DataLoader(val_loader.dataset, batch_size=8, sampler=val_sampler)

        model_pt = model_pt.to(device)

        model_pt = DistributedDataParallel(model_pt, device_ids=[gpu])

        scaler = GradScaler()

        model_pt.train()
        for epoch in range(num_epochs):
            print(f'Epoch: {epoch}')
            train_sampler.set_epoch(epoch)

            for i, batch_data in enumerate(train_loader):
                with autocast():
                    batch = {k: v.to(device) for k, v in batch_data.items()}
                    outputs = model_pt(**batch)
                    loss = outputs.loss

                scaler.scale(loss).backward()
                scaler.step(optimizer_pt)
                scaler.update()

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
            for batch_data in val_loader:
                with autocast():
                    batch = {k: v.to(device) for k, v in batch_data.items()}
                    outputs = model_pt(**batch)
                    logits = outputs.logits
                    predictions = torch.argmax(logits, dim=-1)

                    loss_batch = outputs.loss.cpu().item()

                val_pred += predictions.cpu().tolist()
                val_true += batch["labels"].cpu().tolist()
                val_loss.append(loss_batch)

        with open(f'{THIS_DIR}/ddp_bert_save/pt_ddp_predictions_{gpu}.p', 'wb') as f:
            pickle.dump([val_loss, val_pred, val_true], f)

    def get_data_sets(self, ds_subset_size=0):
        self.set_seeds()

        raw_datasets = load_dataset("imdb")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

        def tokenize_function(examples):
            return tokenizer(examples["text"], padding="max_length", truncation=True)

        tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
        tokenized_datasets = tokenized_datasets.remove_columns(["text"])
        tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
        tokenized_datasets.set_format("torch")

        if ds_subset_size == 0:
            train_dataset = tokenized_datasets["train"]
            eval_dataset = tokenized_datasets["test"]
        else:
            train_dataset = tokenized_datasets["train"].shuffle().select(range(ds_subset_size))
            eval_dataset = tokenized_datasets["test"].shuffle().select(range(ds_subset_size))

        return train_dataset, eval_dataset

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
