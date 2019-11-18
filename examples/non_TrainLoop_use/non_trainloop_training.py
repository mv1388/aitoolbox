from tqdm import tqdm
import os
import time
import datetime
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from aitoolbox.experiment.training_history import TrainingHistory
from aitoolbox.experiment.result_package.basic_packages import ClassificationResultPackage
from aitoolbox.experiment.local_experiment_saver import FullPyTorchExperimentLocalSaver

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class Net(nn.Module):
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

experiment_timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S')
training_history = TrainingHistory()


for epoch in range(args.epochs):
    print(f'EPOCH: {epoch}')

    # Training
    model.train()
    for data, target in tqdm(train_loader):
        optimizer.zero_grad()
        predictions = model(data)

        loss = criterion(predictions, target)
        loss.backward()

        optimizer.step()

    # Testing
    model.eval()
    y_pred, y_test = [], []
    for data, target in tqdm(test_loader):
        predictions = model(data)
        y_pred += predictions.tolist()
        y_test += target.tolist()

    result_pkg = ClassificationResultPackage()
    result_pkg.prepare_result_package(y_true=y_test, y_predicted=y_pred,
                                      hyperparameters=args.__dict__, training_history=training_history)

    for metric_name, metric_result in result_pkg.get_results().items():
        training_history.insert_single_result_into_history(metric_name, metric_result)


# Save model & results
experiment_saver = FullPyTorchExperimentLocalSaver(project_name='localRunCNNTest', experiment_name='CNN_MNIST_test',
                                                   local_model_result_folder_path=THIS_DIR)

model_checkpoint = {'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'hyperparams': args.__dict__}
experiment_saver.save_experiment(model_checkpoint, result_pkg, experiment_timestamp=experiment_timestamp)
