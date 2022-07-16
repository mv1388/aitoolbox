import unittest

import os
import shutil
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from datasets import load_dataset

from aitoolbox import TrainLoopCheckpointEndSave, TTModel, ModelPerformanceEvaluation, ModelPerformancePrintReport, \
    ModelTrainHistoryPlot, ModelTrainHistoryFileWriter, BinaryClassificationResultPackage

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

"""
Training taken from: 
    https://pytorch-ignite.ai/tutorials/beginner/02-transformers-text-classification/
    https://colab.research.google.com/github/pytorch-ignite/pytorch-ignite.ai/blob/gh-pages/tutorials/beginner/02-transformers-text-classification.ipynb
"""


class BERTModel(TTModel):
    def __init__(self, hf_model):
        super().__init__()
        self.hf_model = hf_model

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


class TestIMDBBERTExperimentTrack(unittest.TestCase):
    def test_trainloop_core_pytorch_compare(self):
        train_data, test_data = self.get_data_sets(ds_subset_size=1000)

        val_loss_tl, y_pred_tl, y_true_tl = self.train_eval_trainloop(train_data, test_data, num_epochs=2)
        val_loss_pt, y_pred_pt, y_true_pt = self.train_eval_core_pytorch(train_data, test_data, num_epochs=2)

        self.assertEqual(val_loss_tl, val_loss_pt)
        self.assertEqual(y_pred_tl, y_pred_pt)
        self.assertEqual(y_true_tl, y_true_pt)

        project_path = os.path.join(THIS_DIR, 'tl_full_experiment_tracking')
        if os.path.exists(project_path):
            shutil.rmtree(project_path)

    def train_eval_trainloop(self, train_data, test_data, num_epochs):
        self.set_seeds()

        train_loader = DataLoader(train_data, shuffle=True, batch_size=8)
        val_loader = DataLoader(test_data, batch_size=8)

        hf_model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)
        model = BERTModel(hf_model)
        optimizer = AdamW(model.parameters(), lr=5e-5)

        callbacks = [
            ModelPerformanceEvaluation(BinaryClassificationResultPackage(), {},
                                       on_train_data=True, on_val_data=True),
            ModelPerformancePrintReport(['train_Accuracy', 'val_Accuracy']),
            ModelTrainHistoryPlot(),
            ModelTrainHistoryFileWriter()
        ]

        print('Start TrainLoop')
        train_loop = TrainLoopCheckpointEndSave(
            model,
            train_loader, val_loader, None,
            optimizer, None,
            project_name='tl_full_experiment_tracking', experiment_name='tutorial_example',
            local_model_result_folder_path=THIS_DIR,
            hyperparams={},
            val_result_package=BinaryClassificationResultPackage(),
            cloud_save_mode=None
        )
        self.assertEqual(train_loop.device.type, "cuda")

        train_loop.fit(num_epochs=num_epochs, callbacks=callbacks)

        val_loss = train_loop.evaluate_loss_on_validation_set(force_prediction=True)
        y_pred, y_true, _ = train_loop.predict_on_validation_set(force_prediction=True)

        return val_loss, y_pred.tolist(), y_true.tolist()

    def train_eval_core_pytorch(self, train_data, test_data, num_epochs):
        self.set_seeds()

        train_loader = DataLoader(train_data, shuffle=True, batch_size=8)
        val_loader = DataLoader(test_data, batch_size=8)

        USE_CUDA = torch.cuda.is_available()
        device = torch.device("cuda" if USE_CUDA else "cpu")
        self.assertEqual(device.type, "cuda")

        model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)
        model = model.to(device)
        optimizer = AdamW(model.parameters(), lr=5e-5)

        print('Starting manual PyTorch training')
        model.train()
        for epoch in range(num_epochs):
            print(f'Epoch: {epoch}')
            for i, batch_data in enumerate(train_loader):
                batch = {k: v.to(device) for k, v in batch_data.items()}
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

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
        model.eval()
        with torch.no_grad():
            for batch_data in val_loader:
                batch = {k: v.to(device) for k, v in batch_data.items()}
                outputs = model(**batch)
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)

                loss_batch = outputs.loss.cpu().item()
                val_pred += predictions.cpu().tolist()
                val_true += batch["labels"].cpu().tolist()
                val_loss.append(loss_batch)
            val_loss = np.mean(val_loss)

        return val_loss, val_pred, val_true

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
