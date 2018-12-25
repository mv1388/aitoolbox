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
            model (torch.nn.modules.Module):
            train_loader (torch.utils.data.DataLoader):
            validation_loader (torch.utils.data.DataLoader):
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

        self.experiment_timestamp = None

        # TODO: implement history tracking
        self.history = None

    def __call__(self, num_epoch):
        self.do_train(num_epoch)

    def do_train(self, num_epoch):
        """

        Args:
            num_epoch (int):

        Returns:

        """
        loss_avg = []
        self.experiment_timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')

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
            float:
        """
        self.model.eval()
        val_loss_avg = []

        with torch.no_grad():
            for batch_data in tqdm(self.validation_loader):
                val_loss_batch = self.batch_model_feed_def(self.model, batch_data, self.criterion, self.device)

                val_loss_avg.append(float(val_loss_batch))

        self.model.train()

        return np.mean(val_loss_avg)


class TrainLoopCheckpoint(TrainLoop):
    def __init__(self, model,
                 train_loader, validation_loader,
                 batch_model_feed_def,
                 optimizer, criterion):
        TrainLoop.__init__(self, model, train_loader, validation_loader, batch_model_feed_def, optimizer, criterion)

