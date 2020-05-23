import unittest

import os
import shutil
import random
import numpy as np
import torch
import torch.nn as nn

import torchtext
import torchtext.data

from aitoolbox import TrainLoopCheckpointEndSave, TTModel, TTDataParallel, \
    ModelPerformanceEvaluation, ModelPerformancePrintReport, \
    ModelTrainHistoryPlot, ModelTrainHistoryFileWriter, BinaryClassificationResultPackage

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

"""
Training taken from:
    https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/rnn/rnn_simple_packed_imdb.ipynb
"""


class RNNClassifier(TTModel):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

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
        _, final_hidden = self.rnn(packed)

        out = self.fc(final_hidden.squeeze(0)).squeeze(1)
        return out

    def get_loss(self, batch_data, criterion, device):
        text, text_lengths = batch_data.text
        text = text.to(device)
        text_lengths = text_lengths.to(device)

        # Fix for DP to work as it cuts batches on 0-th dimension
        text = text.transpose(0, 1)

        logits = self(text, text_lengths)

        loss = criterion(logits, batch_data.label.to(device))
        return loss

    def get_predictions(self, batch_data, device):
        text, text_lengths = batch_data.text
        text = text.to(device)
        text_lengths = text_lengths.to(device)

        # Fix for DP to work as it cuts batches on 0-th dimension
        text = text.transpose(0, 1)

        logits = self(text, text_lengths)
        predictions = (torch.sigmoid(logits) > 0.5).long()

        return predictions.cpu(), batch_data.label, {}


class LSTMClassifier(TTModel):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

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

        out = self.fc(final_hidden.squeeze(0)).squeeze(1)
        return out

    def get_loss(self, batch_data, criterion, device):
        text, text_lengths = batch_data.text
        text = text.to(device)
        text_lengths = text_lengths.to(device)

        # Fix for DP to work as it cuts batches on 0-th dimension
        text = text.transpose(0, 1)

        logits = self(text, text_lengths)

        loss = criterion(logits, batch_data.label.to(device))
        return loss

    def get_predictions(self, batch_data, device):
        text, text_lengths = batch_data.text
        text = text.to(device)
        text_lengths = text_lengths.to(device)

        # Fix for DP to work as it cuts batches on 0-th dimension
        text = text.transpose(0, 1)

        logits = self(text, text_lengths)
        predictions = (torch.sigmoid(logits) > 0.5).long()

        return predictions.cpu(), batch_data.label, {}


class TestIMDBRNNExperimentTrack(unittest.TestCase):
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

        project_path = os.path.join(THIS_DIR, 'tl_full_experiment_tracking')
        if os.path.exists(project_path):
            shutil.rmtree(project_path)

    def test_dp_auto_wrap_trainloop_core_pytorch_compare(self):
        train_data, test_data, INPUT_DIM = self.get_data_sets()

        val_loss_tl, y_pred_tl, y_true_tl = self.train_eval_trainloop(train_data, test_data, INPUT_DIM, num_epochs=5,
                                                                      tl_dp_auto_wrap=True)
        val_loss_pt, y_pred_pt, y_true_pt = self.train_eval_core_pytorch(train_data, test_data, INPUT_DIM, num_epochs=5)

        self.assertEqual(val_loss_tl, val_loss_pt)
        self.assertEqual(y_pred_tl, y_pred_pt)
        self.assertEqual(y_true_tl, y_true_pt)

        project_path = os.path.join(THIS_DIR, 'data')
        if os.path.exists(project_path):
            shutil.rmtree(project_path)

        project_path = os.path.join(THIS_DIR, 'tl_full_experiment_tracking')
        if os.path.exists(project_path):
            shutil.rmtree(project_path)

    def train_eval_trainloop(self, train_data, test_data, INPUT_DIM, num_epochs, tl_dp_auto_wrap=False):
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

        model = RNNClassifier(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM)
        if not tl_dp_auto_wrap:
            model = TTDataParallel(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.BCEWithLogitsLoss()

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
            optimizer, criterion,
            project_name='tl_full_experiment_tracking', experiment_name='tutorial_example',
            local_model_result_folder_path=THIS_DIR,
            hyperparams={},
            val_result_package=BinaryClassificationResultPackage(),
            cloud_save_mode=None
        )
        self.assertEqual(train_loop.device.type, "cuda")

        if not tl_dp_auto_wrap:
            train_loop.fit(num_epochs=num_epochs, callbacks=callbacks)
        else:
            train_loop.fit_data_parallel(num_epochs=num_epochs, callbacks=callbacks)

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

        model = RNNClassifier(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM)
        model = nn.DataParallel(model).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
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

                logits = model(text, text_lengths)
                loss = criterion(logits, target)
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
                text, text_lengths = batch_data.text
                target = batch_data.label
                text = text.to(device)
                text_lengths = text_lengths.to(device)
                target = target.to(device)

                # Fix for DP to work as it cuts batches on 0-th dimension
                text = text.transpose(0, 1)

                logits = model(text, text_lengths)
                loss_batch = criterion(logits, target).cpu().item()
                val_pred += (torch.sigmoid(logits) > 0.5).long().cpu().tolist()
                val_true += target.cpu().tolist()
                val_loss.append(loss_batch)
            val_loss = np.mean(val_loss)

        return val_loss, val_pred, val_true

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


class TestIMDBLSTMExperimentTrack(unittest.TestCase):
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

        project_path = os.path.join(THIS_DIR, 'tl_full_experiment_tracking')
        if os.path.exists(project_path):
            shutil.rmtree(project_path)

    def test_dp_auto_wrap_trainloop_core_pytorch_compare(self):
        train_data, test_data, INPUT_DIM = self.get_data_sets()

        val_loss_tl, y_pred_tl, y_true_tl = self.train_eval_trainloop(train_data, test_data, INPUT_DIM, num_epochs=5,
                                                                      tl_dp_auto_wrap=True)
        val_loss_pt, y_pred_pt, y_true_pt = self.train_eval_core_pytorch(train_data, test_data, INPUT_DIM, num_epochs=5)

        self.assertEqual(val_loss_tl, val_loss_pt)
        self.assertEqual(y_pred_tl, y_pred_pt)
        self.assertEqual(y_true_tl, y_true_pt)

        project_path = os.path.join(THIS_DIR, 'data')
        if os.path.exists(project_path):
            shutil.rmtree(project_path)

        project_path = os.path.join(THIS_DIR, 'tl_full_experiment_tracking')
        if os.path.exists(project_path):
            shutil.rmtree(project_path)

    def train_eval_trainloop(self, train_data, test_data, INPUT_DIM, num_epochs, tl_dp_auto_wrap=False):
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
        if not tl_dp_auto_wrap:
            model = TTDataParallel(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.BCEWithLogitsLoss()

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
            optimizer, criterion,
            project_name='tl_full_experiment_tracking', experiment_name='tutorial_example',
            local_model_result_folder_path=THIS_DIR,
            hyperparams={},
            val_result_package=BinaryClassificationResultPackage(),
            cloud_save_mode=None
        )
        self.assertEqual(train_loop.device.type, "cuda")

        if not tl_dp_auto_wrap:
            train_loop.fit(num_epochs=num_epochs, callbacks=callbacks)
        else:
            train_loop.fit_data_parallel(num_epochs=num_epochs, callbacks=callbacks)

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
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
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

                logits = model(text, text_lengths)
                loss = criterion(logits, target)
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
                text, text_lengths = batch_data.text
                target = batch_data.label
                text = text.to(device)
                text_lengths = text_lengths.to(device)
                target = target.to(device)

                # Fix for DP to work as it cuts batches on 0-th dimension
                text = text.transpose(0, 1)

                logits = model(text, text_lengths)
                loss_batch = criterion(logits, target).cpu().item()
                val_pred += (torch.sigmoid(logits) > 0.5).long().cpu().tolist()
                val_true += target.cpu().tolist()
                val_loss.append(loss_batch)
            val_loss = np.mean(val_loss)

        return val_loss, val_pred, val_true

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
