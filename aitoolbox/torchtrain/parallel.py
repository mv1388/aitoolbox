import functools
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

from aitoolbox.utils.util import copy_function

# Removed old Parallel wrapper implementations in: https://github.com/mv1388/aitoolbox/pull/583


class TTParallelBase:
    def __init__(self, module,
                 default_model_methods=('get_loss', 'get_loss_eval', 'get_predictions')):
        """torchtrain parallel base class used for transferring TTModel functions to the PyTorch Parallel wrappers level

        Args:
            module (aitoolbox.torchtrain.model.TTModel): neural network model
            default_model_methods (list or tuple): list of core methods which are present also in TTModel abstract class
        """
        # Core TTModel methods which every model has
        self.get_loss_fn = copy_function(module.get_loss)
        self.get_loss_eval_fn = copy_function(module.get_loss_eval)
        self.get_predictions_fn = copy_function(module.get_predictions)

        # Transfer any additional sub-methods from TTModel to TTDataParallel
        methods = list(set(dir(module))
                       - set(dir(nn.Module)) - set(vars(module)['_modules'].keys()) - set(default_model_methods))
        additional_methods = [method_name for method_name in methods if callable(getattr(module, method_name, None))]

        for method_name in additional_methods:
            setattr(self, method_name,
                    functools.partial(copy_function(getattr(module, method_name)), self))

        # Optionally transfer additional TTModel attributes to the TTDataParallel level
        if module.transfer_model_attributes is not None and isinstance(module.transfer_model_attributes, (list, tuple)):
            for attr_name in module.transfer_model_attributes:
                setattr(self, attr_name, getattr(module, attr_name))

    def get_loss(self, batch_data, criterion, device):
        return self.get_loss_fn(self, batch_data, criterion, device)

    def get_loss_eval(self, batch_data, criterion, device):
        return self.get_loss_eval_fn(self, batch_data, criterion, device)

    def get_predictions(self, batch_data, device):
        return self.get_predictions_fn(self, batch_data, device)


class TTDataParallel(nn.DataParallel, TTParallelBase):
    def __init__(self, module,
                 default_model_methods=('get_loss', 'get_loss_eval', 'get_predictions'), **kwargs):
        """torchtrain enabled DataParallel

        This DataParallel wrapper works in the same way as the original PyTorch nn.DataParallel. Furthermore, it exposes
        TTModel batch data feeding definitions (additional abstract methods) to the TrainLoop while still enabling
        multi GPU training.

        Args:
            module (aitoolbox.torchtrain.model.TTModel): neural network model
            default_model_methods (list or tuple): list of core methods which are present also in TTModel abstract class
            **kwargs: additional parameters for underlying nn.DataParallel
        """
        nn.DataParallel.__init__(self, module, **kwargs)
        TTParallelBase.__init__(self, module, default_model_methods)


class TTDistributedDataParallel(TTParallelBase, DistributedDataParallel):
    def __init__(self, module,
                 default_model_methods=('get_loss', 'get_loss_eval', 'get_predictions'), **kwargs):
        """torchtrain enabled DistributedDataParallel

        Args:
            module (aitoolbox.torchtrain.model.TTModel): neural network model
            default_model_methods (list or tuple): list of core methods which are present also in TTModel abstract class
            **kwargs: additional parameters for underlying nn.parallel.DistributedDataParallel
        """
        DistributedDataParallel.__init__(self, module, **kwargs)
        TTParallelBase.__init__(self, module, default_model_methods)
