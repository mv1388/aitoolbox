from tqdm import tqdm
import time
import datetime

import numpy as np

import torch


class TrainLoop:
    def __init__(self, model,
                 train_loader, validation_loader,
                 batch_model_feed_def,
                 optimizer, criterion):
        """

        Args:
            model:
            train_loader:
            validation_loader:
            batch_model_feed_def:
            optimizer:
            criterion:
        """
        self.model = model
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.batch_model_feed_def = batch_model_feed_def
        self.optimizer = optimizer
        self.criterion = criterion

        USE_CUDA = torch.cuda.is_available()
        self.device = torch.device("cuda" if USE_CUDA else "cpu")

    def __call__(self, num_epoch):
        self.do_train(num_epoch)

    def do_train(self, num_epoch):
        """

        Args:
            num_epoch:

        Returns:

        """
        loss_avg = []
        experiment_timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')

        self.model = self.model.to(self.device)
        self.model.train()

        for epoch in range(num_epoch):
            print(f'Epoch: {epoch + 1}')

            for batch_data in tqdm(self.train_loader):
                loss_batch = self.batch_model_feed_def(self.model, batch_data, self.criterion, self.device)

                # print(f'Loss: {loss_batch}')
                loss_avg.append(float(loss_batch))

                self.optimizer.zero_grad()
                loss_batch.backward()
                self.optimizer.step()

            print(f'AVG TRAIN LOSS: {np.mean(loss_avg)}')
            loss_avg = []

            val_loss_batch = self.evaluate_loss_on_validation()
            print(f'VAL LOSS: {val_loss_batch}')

    def evaluate_loss_on_validation(self):
        """

        Returns:

        """
        self.model.eval()
        val_loss_avg = []

        with torch.no_grad():
            for batch_data in tqdm(self.validation_loader):
                val_loss_batch = self.batch_model_feed_def(self.model, batch_data, self.criterion, self.device)

                val_loss_avg.append(float(val_loss_batch))

        self.model.train()

        return np.mean(val_loss_avg)


def train_loop(model,
               train_loader, validation_loader,
               batch_model_feed_def,
               num_epoch, optimizer, criterion):
    """

    Args:
        model:
        train_loader:
        validation_loader:
        batch_model_feed_def:
        num_epoch (int):
        optimizer:
        criterion:

    Returns:

    """
    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda" if USE_CUDA else "cpu")

    loss_avg = []
    experiment_timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')

    model = model.to(device)
    model.train()

    for epoch in range(num_epoch):
        print(f'Epoch: {epoch + 1}')

        for batch_data in tqdm(train_loader):
            loss_batch = batch_model_feed_def(model, batch_data, criterion, device)

            # print(f'Loss: {loss_batch}')
            loss_avg.append(float(loss_batch))

            optimizer.zero_grad()
            loss_batch.backward()
            optimizer.step()

        print(f'AVG TRAIN LOSS: {np.mean(loss_avg)}')
        loss_avg = []

        val_loss_batch = evaluate_loss_on_validation(model, validation_loader,
                                                     batch_model_feed_def, criterion, device)
        print(f'VAL LOSS: {val_loss_batch}')


def evaluate_loss_on_validation(model, validation_loader,
                                batch_model_feed_def, criterion, device):
    """

    Args:
        model:
        validation_loader:
        batch_model_feed_def:
        criterion:
        device:

    Returns:

    """
    model.eval()
    val_loss_avg = []

    with torch.no_grad():
        for batch_data in tqdm(validation_loader):
            val_loss_batch = batch_model_feed_def(model, batch_data, criterion, device)

            val_loss_avg.append(float(val_loss_batch))

    model.train()

    return np.mean(val_loss_avg)


def squad_batch_model_feed(model, batch_data, criterion, device):
    paragraph_batch, paragraph_lengths, question_batch, question_lengths, span = batch_data

    paragraph_batch = paragraph_batch.to(device)
    paragraph_lengths = paragraph_lengths.to(device)
    question_batch = question_batch.to(device)
    question_lengths = question_lengths.to(device)
    span = span.to(device)

    output_start_span, output_end_span = model(paragraph_batch, question_batch, paragraph_lengths, question_lengths)

    loss1 = criterion(output_start_span, span[:, 0].long())
    loss2 = criterion(output_end_span, span[:, 1].long())
    loss = loss1 + loss2

    return loss
