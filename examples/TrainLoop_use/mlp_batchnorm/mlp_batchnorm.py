import time
import numpy as np
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
import torch.nn as nn

from AIToolbox.torchtrain.model import TTFullModel
from AIToolbox.torchtrain.train_loop import TrainLoop
from AIToolbox.torchtrain.callbacks.performance_eval_callbacks import ModelPerformanceEvaluation, ModelPerformancePrintReport
from AIToolbox.experiment.result_package.basic_packages import ClassificationResultPackage


if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True


##########################
### SETTINGS
##########################

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Hyperparameters
random_seed = 1
learning_rate = 0.1
num_epochs = 10
batch_size = 64

# Architecture
num_features = 784
num_hidden_1 = 128
num_hidden_2 = 256
num_classes = 10


##########################
### MNIST DATASET
##########################

# Note transforms.ToTensor() scales input images
# to 0-1 range
train_dataset = datasets.MNIST(root='data',
                               train=True,
                               transform=transforms.ToTensor(),
                               download=True)

test_dataset = datasets.MNIST(root='data',
                              train=False,
                              transform=transforms.ToTensor())


train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True)

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=batch_size,
                         shuffle=False)

# Checking the dataset
for images, labels in train_loader:
    print('Image batch dimensions:', images.shape)
    print('Image label dimensions:', labels.shape)
    break


##########################
### MODEL
##########################

class MultilayerPerceptron(TTFullModel):
    def __init__(self, num_features, num_classes, num_hidden_1, num_hidden_2):
        super(MultilayerPerceptron, self).__init__()

        ### 1st hidden layer
        self.linear_1 = torch.nn.Linear(num_features, num_hidden_1)
        # The following to lones are not necessary,
        # but used here to demonstrate how to access the weights
        # and use a different weight initialization.
        # By default, PyTorch uses Xavier/Glorot initialization, which
        # should usually be preferred.
        self.linear_1.weight.detach().normal_(0.0, 0.1)
        self.linear_1.bias.detach().zero_()
        self.linear_1_bn = torch.nn.BatchNorm1d(num_hidden_1)

        ### 2nd hidden layer
        self.linear_2 = torch.nn.Linear(num_hidden_1, num_hidden_2)
        self.linear_2.weight.detach().normal_(0.0, 0.1)
        self.linear_2.bias.detach().zero_()
        self.linear_2_bn = torch.nn.BatchNorm1d(num_hidden_2)

        ### Output layer
        self.linear_out = torch.nn.Linear(num_hidden_2, num_classes)
        self.linear_out.weight.detach().normal_(0.0, 0.1)
        self.linear_out.bias.detach().zero_()

    def forward(self, x):
        out = self.linear_1(x)
        # note that batchnorm is in the classic
        # sense placed before the activation
        out = self.linear_1_bn(out)
        out = F.relu(out)

        out = self.linear_2(out)
        out = self.linear_2_bn(out)
        out = F.relu(out)

        logits = self.linear_out(out)
        probas = F.softmax(logits, dim=1)
        return logits, probas

    def get_loss(self, batch_data, criterion, device):
        features, targets = batch_data

        features = features.view(-1, 28 * 28).to(device)
        targets = targets.to(device)

        ### FORWARD AND BACK PROP
        logits, probas = self(features)
        cost = criterion(logits, targets)

        return cost

    def get_predictions(self, batch_data, device):
        features, targets = batch_data

        features = features.view(-1, 28 * 28).to(device)
        targets = targets.to(device)

        logits, probas = self(features)

        _, predicted_labels = torch.max(probas, 1)

        return targets.cpu(), predicted_labels.cpu(), {}


torch.manual_seed(random_seed)
model = MultilayerPerceptron(num_features=num_features,
                             num_classes=num_classes,
                             num_hidden_1=num_hidden_1, num_hidden_2=num_hidden_2)

model = model.to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()


##########################
### TRAINING
##########################

args = {"random_seed": 1,
        "learning_rate": 0.1,
        "num_epochs": 10,
        "batch_size": 64,
        "num_features": 784,
        "num_hidden_1": 128,
        "num_hidden_2": 256,
        "num_classes": 10}

callbacks = [ModelPerformanceEvaluation(ClassificationResultPackage(), args,
                                        on_train_data=True, on_val_data=True),
             ModelPerformancePrintReport(['val_loss', 'train_Accuracy', 'val_Accuracy'], strict_metric_reporting=True)]

TrainLoop(model, train_loader, test_loader, test_loader, optimizer, criterion)(num_epochs, callbacks)
