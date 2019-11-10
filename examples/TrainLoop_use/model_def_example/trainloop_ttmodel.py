from __future__ import print_function
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from aitoolbox.torchtrain.train_loop import TrainLoop, TrainLoopModelCheckpointEndSave
from aitoolbox.torchtrain.model import TTModel
from aitoolbox.torchtrain.callbacks.performance_eval import ModelPerformanceEvaluation, ModelPerformancePrintReport
from aitoolbox.experiment.result_package.basic_packages import ClassificationResultPackage

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class Net(TTModel):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def get_loss(self, batch_data, criterion, device):
        data, target = batch_data
        data, target = data.to(device), target.to(device)

        output = self(data)
        loss = criterion(output, target)

        return loss

    def get_predictions(self, batch_data, device):
        data, y_test = batch_data
        data = data.to(device)

        output = self(data)
        y_pred = output.argmax(dim=1, keepdim=False)  # get the index of the max log-probability

        return y_pred.cpu(), y_test, {}


################################################################################################

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')

parser.add_argument('--save-model', action='store_true', default=False,
                    help='For Saving the current Model')
args = parser.parse_args()
use_cuda = not args.no_cuda and torch.cuda.is_available()


device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)


model = Net()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
criterion = F.nll_loss

callbacks = [ModelPerformanceEvaluation(ClassificationResultPackage(), args.__dict__,
                                        on_train_data=True, on_val_data=True),
             ModelPerformancePrintReport(['train_Accuracy', 'val_Accuracy'], strict_metric_reporting=True)]


# TrainLoop(model,
#           train_loader, test_loader, None,
#           optimizer, criterion)(num_epoch=10, callbacks=callbacks)


TrainLoopModelCheckpointEndSave(model,
                                train_loader, test_loader, test_loader,
                                optimizer, criterion,
                                project_name='localRunCNNTest',
                                experiment_name='CNN_MNIST_test',
                                local_model_result_folder_path=THIS_DIR,
                                hyperparams=args.__dict__,
                                test_result_package=ClassificationResultPackage(),
                                cloud_save_mode=None)(num_epoch=5, callbacks=callbacks)
