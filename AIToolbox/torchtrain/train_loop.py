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

        print(f'AVG LOSS: {np.mean(loss_avg)}')
        loss_avg = []

        val_loss_batch = evaluate_loss_on_validation(model, validation_loader,
                                                     batch_model_feed_def, criterion, device)
        print(f'VAL LOSS: {np.mean(float(val_loss_batch))}')


def evaluate_loss_on_validation(model, validation_loader,
                                batch_model_feed_def, criterion, device):
    model.eval()

    with torch.no_grad():
        val_loss_batch = batch_model_feed_def(model, validation_loader, criterion, device)

    model.train()

    return val_loss_batch


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
