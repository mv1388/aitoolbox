from tqdm import tqdm
import time
import datetime
import numpy as np
import torch


def train_loop(model,
               train_loader, validation_loader,
               batch_model_feed_def,
               num_epoch, optimizer, criterion):

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

    model.eval()
    val_loss_avg = []

    with torch.no_grad():
        for batch_data in tqdm(validation_loader):
            val_loss_batch = batch_model_feed_def(model, batch_data, criterion, device)

            val_loss_avg.append(float(val_loss_batch))

    model.train()

    return np.mean(val_loss_avg)
