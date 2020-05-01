import torch
import torch.nn as nn
import torch.nn.functional as F

from aitoolbox.torchtrain.data.batch_model_feed_defs import AbstractModelFeedDefinition
from aitoolbox.torchtrain.callbacks.abstract import AbstractCallback
from aitoolbox.experiment.result_package.abstract_result_packages import AbstractResultPackage
from aitoolbox.experiment.core_metrics.abstract_metric import AbstractBaseMetric
from aitoolbox.torchtrain.model import TTModel


def function_exists(object_to_check, fn_name):
    if hasattr(object_to_check, fn_name):
        fn_obj = getattr(object_to_check, fn_name, None)
        return callable(fn_obj)
    return False


class Net(nn.Module):
    def __init__(self):
        super().__init__()
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


class NetUnifiedBatchFeed(TTModel):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

        self.dummy_batch = DummyBatch()
        self.prediction_count = 0

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
        return self.dummy_batch

    def get_loss_eval(self, batch_data, criterion, device):
        return self.get_loss(batch_data, criterion, device)

    def get_predictions(self, batch_data, device):
        self.prediction_count += 1
        return torch.FloatTensor([self.prediction_count + 100] * 64).cpu(), \
               torch.FloatTensor([self.prediction_count] * 64).cpu(), {'bla': [self.prediction_count + 200] * 64}


class SmallFFNet(TTModel):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(10, 10)
        self.l2 = nn.Linear(10, 1)
        self.out_act = nn.Sigmoid()

    def forward(self, x):
        out = F.relu(self.l1(x.float()))
        out = self.l2(out)
        return self.out_act(out)

    def get_loss(self, batch_data, criterion, device):
        x, y = batch_data
        pred_y = self(x)
        loss = criterion(pred_y, y)
        return loss

    def get_predictions(self, batch_data, device):
        x, y = batch_data
        pred_y = self(x)
        return pred_y, y, {}
    
    
def keras_dummy_model():
    from keras.models import Sequential
    from keras.layers import Dense

    model = Sequential()
    model.add(Dense(12, input_dim=8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    return model
    
    
class CallbackTracker(AbstractCallback):
    def __init__(self):
        AbstractCallback.__init__(self, 'CallbackTracker1')
        self.callback_calls = []
        self.call_ctr = {'on_train_loop_registration': 0, 'on_epoch_begin': 0, 'on_epoch_end': 0, 'on_train_begin': 0,
                         'on_train_end': 0, 'on_batch_begin': 0, 'on_batch_end': 0,
                         'on_after_gradient_update': 0, 'on_after_optimizer_step': 0}

    def on_train_loop_registration(self):
        self.train_loop_obj.grad_cb_used = True

        self.callback_calls.append('on_train_loop_registration')
        self.call_ctr['on_train_loop_registration'] += 1

    def on_epoch_begin(self):
        self.callback_calls.append('on_epoch_begin')
        self.call_ctr['on_epoch_begin'] += 1

    def on_epoch_end(self):
        self.callback_calls.append('on_epoch_end')
        self.call_ctr['on_epoch_end'] += 1

    def on_train_begin(self):
        self.callback_calls.append('on_train_begin')
        self.call_ctr['on_train_begin'] += 1

    def on_train_end(self):
        self.callback_calls.append('on_train_end')
        self.call_ctr['on_train_end'] += 1

    def on_batch_begin(self):
        self.callback_calls.append('on_batch_begin')
        self.call_ctr['on_batch_begin'] += 1

    def on_batch_end(self):
        self.callback_calls.append('on_batch_end')
        self.call_ctr['on_batch_end'] += 1

    def on_after_gradient_update(self):
        self.callback_calls.append('on_after_gradient_update')
        self.call_ctr['on_after_gradient_update'] += 1

    def on_after_optimizer_step(self):
        self.callback_calls.append('on_after_optimizer_step')
        self.call_ctr['on_after_optimizer_step'] += 1


class CallbackTrackerShort(AbstractCallback):
    def __init__(self):
        AbstractCallback.__init__(self, 'CallbackTracker2')
        self.callback_calls = []
        self.call_ctr = {'on_train_loop_registration': 0, 'on_epoch_begin': 0, 'on_epoch_end': 0, 'on_train_begin': 0,
                         'on_train_end': 0, 'on_batch_begin': 0, 'on_batch_end': 0,
                         'on_after_gradient_update': 0, 'on_after_optimizer_step': 0}

    def on_train_loop_registration(self):
        self.train_loop_obj.grad_cb_used = True

    def on_epoch_begin(self):
        self.callback_calls.append('on_epoch_begin')
        self.call_ctr['on_epoch_begin'] += 1

    def on_epoch_end(self):
        self.callback_calls.append('on_epoch_end')
        self.call_ctr['on_epoch_end'] += 1

    def on_train_end(self):
        self.callback_calls.append('on_train_end')
        self.call_ctr['on_train_end'] += 1

    def on_batch_begin(self):
        self.callback_calls.append('on_batch_begin')
        self.call_ctr['on_batch_begin'] += 1

    def on_after_gradient_update(self):
        self.callback_calls.append('on_after_gradient_update')
        self.call_ctr['on_after_gradient_update'] += 1

    def on_after_optimizer_step(self):
        self.callback_calls.append('on_after_optimizer_step')
        self.call_ctr['on_after_optimizer_step'] += 1


class DeactivateModelFeedDefinition(AbstractModelFeedDefinition):
    def __init__(self):
        self.dummy_batch = DummyBatch()
        self.prediction_count = 0
    
    def get_loss(self, model, batch_data, criterion, device):
        """

        Args:
            model:
            batch_data:
            criterion:
            device:

        Returns:
            PyTorch loss
        """    
        return self.dummy_batch

    def get_loss_eval(self, model, batch_data, criterion, device):
        return self.get_loss(model, batch_data, criterion, device)

    def get_predictions(self, model, batch_data, device):
        """

        Args:
            model:
            batch_data:
            device:

        Returns:
            np.array, np.array, dict: y_test.cpu(), y_pred.cpu(), metadata
        """
        self.prediction_count += 1
        return torch.FloatTensor([self.prediction_count + 100] * 64).cpu(), \
               torch.FloatTensor([self.prediction_count] * 64).cpu(), {'bla': [self.prediction_count + 200] * 64}


class DummyBatch:
    def __init__(self):
        self.back_ctr = 0
        self.item_ctr = 0
    
    def backward(self):
        self.back_ctr += 1

    def item(self):
        self.item_ctr += 1
        return 1.

    def __truediv__(self, other):
        return self


class MiniDummyOptimizer:
    def __init__(self):
        self.zero_grad_ctr = 0
        self.step_ctr = 0

    def zero_grad(self):
        self.zero_grad_ctr += 1

    def step(self):
        self.step_ctr += 1


class DummyOptimizer:
    def __init__(self):
        self.zero_grad_ctr = 0
        self.step_ctr = 0
        
    def zero_grad(self):
        self.zero_grad_ctr += 1
    
    def step(self):
        self.step_ctr += 1

    def state_dict(self):
        print('In state_dict')


class DummyResultPackage(AbstractResultPackage):
    def __init__(self):
        AbstractResultPackage.__init__(self, 'DummyPackage', False)
        self.experiment_path = None

    def prepare_results_dict(self):
        return {'dummy': 111}
        
    def set_experiment_dir_path_for_additional_results(self, project_name, experiment_name, experiment_timestamp,
                                                       local_model_result_folder_path):
        self.experiment_path = f'{local_model_result_folder_path}/{project_name}_{experiment_name}_{experiment_timestamp}'


class DummyResultPackageExtend(DummyResultPackage):
    def __init__(self):
        DummyResultPackage.__init__(self)
        self.ctr = 0.
    
    def prepare_results_dict(self):
        results_dict = {'dummy': 111 + self.ctr, 'extended_dummy': 1323123.44 + self.ctr}
        self.ctr += 12
        return results_dict


class DummyResultPackageExtendV2(DummyResultPackageExtend):
    def __init__(self, additional_results):
        DummyResultPackageExtend.__init__(self)
        self.additional_results = additional_results

    def list_additional_results_dump_paths(self):
        return self.additional_results
    

class DummyResultPackageExtendVariable(DummyResultPackageExtend):
    def __init__(self, result_d):
        DummyResultPackageExtend.__init__(self)
        self.result_d = result_d
        
    def prepare_results_dict(self):
        return self.result_d
        

class DummyAbstractBaseMetric(AbstractBaseMetric):
    def __init__(self, val):
        AbstractBaseMetric.__init__(self, val, val+10, 'dummy_metric', np_array=False)

    def calculate_metric(self):
        return self.y_true
