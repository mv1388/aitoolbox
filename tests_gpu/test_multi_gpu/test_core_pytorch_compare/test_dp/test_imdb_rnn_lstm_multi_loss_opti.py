import unittest

import os
import shutil
import random
import numpy as np
import torch
import torch.nn as nn

import torchtext
import torchtext.data

from transformers import AdamW

from aitoolbox import TrainLoop, TTModel
from aitoolbox.torchtrain.multi_loss_optim import MultiLoss, MultiOptimizer

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

"""
Training taken from:
    https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/rnn/rnn_simple_packed_imdb.ipynb
"""


class LSTMClassifier(TTModel):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.fc_1 = nn.Linear(hidden_dim, output_dim)

        self.fc_2_1 = nn.Linear(hidden_dim, 12)
        self.fc_2_2 = nn.Linear(12, output_dim)

    def forward(self, text, text_length):
        # Dirty non optimal fix for DP to work as it cuts batches on 0-th dimension
        # Transpose back into sequence length first
        text = text.transpose(0, 1)

        # [sentence len, batch size] => [sentence len, batch size, embedding size]
        embedded = self.embedding(text)

        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, text_length)

        # [sentence len, batch size, embedding size] =>
        #  output: [sentence len, batch size, hidden size]
        #  hidden: [1, batch size, hidden size]
        _, (final_hidden, c_n) = self.lstm(packed)

        out_1 = self.fc_1(final_hidden.squeeze(0)).squeeze(1)

        out_2 = self.fc_2_1(final_hidden.squeeze(0))
        out_2 = self.fc_2_2(out_2).squeeze(1)

        return out_1, out_2

    def get_loss(self, batch_data, criterion, device):
        text, text_lengths = batch_data.text
        text = text.to(device)
        text_lengths = text_lengths.to(device)

        # Fix for DP to work as it cuts batches on 0-th dimension
        text = text.transpose(0, 1)

        logits_1, logits_2 = self(text, text_lengths)

        loss_1 = criterion(logits_1, batch_data.label.to(device))
        loss_2 = criterion(logits_2, batch_data.label.to(device))

        loss = MultiLoss({'loss_1': loss_1, 'loss_2': loss_2})

        return loss

    def get_predictions(self, batch_data, device):
        text, text_lengths = batch_data.text
        text = text.to(device)
        text_lengths = text_lengths.to(device)

        # Fix for DP to work as it cuts batches on 0-th dimension
        text = text.transpose(0, 1)

        logits, _ = self(text, text_lengths)
        predictions = (torch.sigmoid(logits) > 0.5).long()

        return predictions.cpu(), batch_data.label, {}


class TestMutliLossMutliOptiIMDBLSTM(unittest.TestCase):
    def test_trainloop_core_pytorch_compare(self):
        train_data, test_data, INPUT_DIM = self.get_data_sets()

        val_loss_tl, y_pred_tl, y_true_tl = self.train_eval_trainloop(train_data, test_data, INPUT_DIM, num_epochs=5)
        val_loss_pt, y_pred_pt, y_true_pt = self.train_eval_core_pytorch(train_data, test_data, INPUT_DIM, num_epochs=5)

        self.assertEqual(val_loss_tl, val_loss_pt)
        self.assertEqual(y_pred_tl, y_pred_pt)
        self.assertEqual(y_true_tl, y_true_pt)

        project_path = os.path.join(THIS_DIR, 'data')
        if os.path.exists(project_path):
            shutil.rmtree(project_path)

    def train_eval_trainloop(self, train_data, test_data, INPUT_DIM, num_epochs):
        self.set_seeds()
        LEARNING_RATE = 1e-3
        BATCH_SIZE = 128

        EMBEDDING_DIM = 100
        HIDDEN_DIM = 100
        OUTPUT_DIM = 1

        train_loader, val_loader = torchtext.data.BucketIterator.splits(
            (train_data, test_data),
            batch_size=BATCH_SIZE, sort_within_batch=True
        )

        model = LSTMClassifier(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM)
        optimizer = MultiOptimizer([
            AdamW(model.parameters(), lr=LEARNING_RATE),
            AdamW(model.parameters(), lr=LEARNING_RATE)
        ])
        criterion = nn.BCEWithLogitsLoss()

        print('Start TrainLoop')
        train_loop = TrainLoop(
            model,
            train_loader, val_loader, None,
            optimizer, criterion,
            gpu_mode='dp'
        )
        self.assertEqual(train_loop.device.type, "cuda")

        train_loop.fit(num_epochs=num_epochs)

        val_loss = train_loop.evaluate_loss_on_validation_set(force_prediction=True)
        y_pred, y_true, _ = train_loop.predict_on_validation_set(force_prediction=True)

        return val_loss, y_pred.tolist(), y_true.tolist()

    def train_eval_core_pytorch(self, train_data, test_data, INPUT_DIM, num_epochs):
        self.set_seeds()
        LEARNING_RATE = 1e-3
        BATCH_SIZE = 128

        EMBEDDING_DIM = 100
        HIDDEN_DIM = 100
        OUTPUT_DIM = 1

        train_loader, val_loader = torchtext.data.BucketIterator.splits(
            (train_data, test_data),
            batch_size=BATCH_SIZE, sort_within_batch=True
        )

        USE_CUDA = torch.cuda.is_available()
        device = torch.device("cuda" if USE_CUDA else "cpu")
        self.assertEqual(device.type, "cuda")

        model = LSTMClassifier(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM)
        model = nn.DataParallel(model).to(device)
        optimizer_1 = AdamW(model.parameters(), lr=LEARNING_RATE)
        optimizer_2 = AdamW(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.BCEWithLogitsLoss()

        print('Starting manual PyTorch training')
        model.train()
        for epoch in range(num_epochs):
            print(f'Epoch: {epoch}')
            for i, batch_data in enumerate(train_loader):
                text, text_lengths = batch_data.text
                target = batch_data.label
                text = text.to(device)
                text_lengths = text_lengths.to(device)
                target = target.to(device)

                # Fix for DP to work as it cuts batches on 0-th dimension
                text = text.transpose(0, 1)

                logits_1, logits_2 = model(text, text_lengths)

                loss_1 = criterion(logits_1, target)
                loss_2 = criterion(logits_2, target)

                loss_1.backward(retain_graph=True)
                optimizer_1.step()
                optimizer_1.zero_grad()

                loss_2.backward()
                optimizer_2.step()
                optimizer_2.zero_grad()

            # Imitate what happens in auto_execute_end_of_epoch() in TrainLoop
            for _ in train_loader:
                pass
            for _ in val_loader:
                pass

        print('Evaluating')
        val_loss_1, val_loss_2, val_pred, val_true = [], [], [], []
        model.eval()
        with torch.no_grad():
            for batch_data in val_loader:
                text, text_lengths = batch_data.text
                target = batch_data.label
                text = text.to(device)
                text_lengths = text_lengths.to(device)
                target = target.to(device)

                # Fix for DP to work as it cuts batches on 0-th dimension
                text = text.transpose(0, 1)

                logits_1, logits_2 = model(text, text_lengths)
                loss_batch_1 = criterion(logits_1, target).cpu().item()
                loss_batch_2 = criterion(logits_2, target).cpu().item()
                val_pred += (torch.sigmoid(logits_1) > 0.5).long().cpu().tolist()
                val_true += target.cpu().tolist()
                val_loss_1.append(loss_batch_1)
                val_loss_2.append(loss_batch_2)
            val_loss_1 = np.mean(val_loss_1)
            val_loss_2 = np.mean(val_loss_2)

        return {'loss_1': val_loss_1, 'loss_2': val_loss_2}, val_pred, val_true

    def get_data_sets(self, ds_sample_ratio=1.):
        self.set_seeds()
        VOCABULARY_SIZE = 20000

        TEXT = torchtext.data.Field(lower=True, include_lengths=True)  # necessary for packed_padded_sequence
        LABEL = torchtext.data.LabelField(dtype=torch.float)

        train_data, test_data = torchtext.datasets.IMDB.splits(
            text_field=TEXT, label_field=LABEL,
            root=os.path.join(THIS_DIR, 'data'),
            train='train', test='test'
        )

        if ds_sample_ratio < 1.0:
            _, train_data = train_data.split(split_ratio=1. - ds_sample_ratio)
            _, test_data = test_data.split(split_ratio=1. - ds_sample_ratio)

        TEXT.build_vocab(train_data, max_size=VOCABULARY_SIZE)
        LABEL.build_vocab(train_data)

        INPUT_DIM = len(TEXT.vocab)

        return train_data, test_data, INPUT_DIM

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
