import functools
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
try:
    from apex.parallel import DistributedDataParallel as ApexDistributedDataParallel
    APEX_AVAILABLE = True
except ImportError:
    # ApexDistributedDataParallel = object
    APEX_AVAILABLE = False
try:
    from deepspeed import DeepSpeedLight
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False

from aitoolbox.utils.util import copy_function


class TTParallelBase:
    def __init__(self, module, add_model_attributes=None,
                 default_model_methods=('get_loss', 'get_loss_eval', 'get_predictions')):
        """torchtrain parallel base class used for transferring TTModel functions to the PyTorch Parallel wrappers level

        Args:
            module (aitoolbox.torchtrain.model.TTModel): neural network model
            add_model_attributes (list or tuple or None): additional TTModel attributes which need to be transferred to
                the TTDataParallel level to enable their use in the transferred/exposed class methods
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
        if add_model_attributes is not None and isinstance(add_model_attributes, (list, tuple)):
            for attr_name in add_model_attributes:
                setattr(self, attr_name, getattr(module, attr_name))

    def get_loss(self, batch_data, criterion, device):
        return self.get_loss_fn(self, batch_data, criterion, device)

    def get_loss_eval(self, batch_data, criterion, device):
        return self.get_loss_eval_fn(self, batch_data, criterion, device)

    def get_predictions(self, batch_data, device):
        return self.get_predictions_fn(self, batch_data, device)


class TTDataParallel(nn.DataParallel, TTParallelBase):
    def __init__(self, module, add_model_attributes=None,
                 default_model_methods=('get_loss', 'get_loss_eval', 'get_predictions'), **kwargs):
        """torchtrain enabled DataParallel

        This DataParallel wrapper works in the same way as the original PyTorch nn.DataParallel. Furthermore it exposes
        TTModel batch data feeding definitions (additional abstract methods) to the TrainLoop while still enabling
        multi GPU training.

        Args:
            module (aitoolbox.torchtrain.model.TTModel): neural network model
            add_model_attributes (list or tuple or None): additional TTModel attributes which need to be transferred to
                the TTDataParallel level to enable their use in the transferred/exposed class methods
            default_model_methods (list or tuple): list of core methods which are present also in TTModel abstract class
            **kwargs: additional parameters for underlying nn.DataParallel
        """
        nn.DataParallel.__init__(self, module, **kwargs)
        TTParallelBase.__init__(self, module, add_model_attributes, default_model_methods)


class TTDistributedDataParallel(DistributedDataParallel, TTParallelBase):
    def __init__(self, module, add_model_attributes=None,
                 default_model_methods=('get_loss', 'get_loss_eval', 'get_predictions'), **kwargs):
        """torchtrain enabled DistributedDataParallel

        Args:
            module (aitoolbox.torchtrain.model.TTModel): neural network model
            add_model_attributes (list or tuple or None): additional TTModel attributes which need to be transferred to
                the TTDistributedDataParallel level to enable their use in the transferred/exposed class methods
            default_model_methods (list or tuple): list of core methods which are present also in TTModel abstract class
            **kwargs: additional parameters for underlying nn.parallel.DistributedDataParallel
        """
        DistributedDataParallel.__init__(self, module, **kwargs)
        TTParallelBase.__init__(self, module, add_model_attributes, default_model_methods)


if APEX_AVAILABLE:
    class TTApexDistributedDataParallel(ApexDistributedDataParallel, TTParallelBase):
        def __init__(self, module, add_model_attributes=None,
                     default_model_methods=('get_loss', 'get_loss_eval', 'get_predictions'), **kwargs):
            """torchtrain enabled Nvidia Apex DistributedDataParallel

            Args:
                module (aitoolbox.torchtrain.model.TTModel): neural network model
                add_model_attributes (list or tuple or None): additional TTModel attributes which need to be
                transferred to the TTDataParallel level to enable their use in the transferred/exposed class methods
                default_model_methods (list or tuple): list of core methods which are present also in TTModel
                    abstract class
                **kwargs: additional parameters for underlying apex.parallel.DistributedDataParallel
            """
            ApexDistributedDataParallel.__init__(self, module, **kwargs)
            TTParallelBase.__init__(self, module, add_model_attributes, default_model_methods)


if DEEPSPEED_AVAILABLE:
    class TTDeepSpeedLight(DeepSpeedLight, TTParallelBase):
        def __init__(self, args, model,
                     add_model_attributes=None, default_model_methods=('get_loss', 'get_loss_eval', 'get_predictions'),
                     **kwargs):
            """torchtrain enabled Microsoft DeepSpeed's DeepSpeedLight engine

            Args:
                args (argparse.Namespace): argparser results structured as per DeepSpeed requirements. A dictionary
                    containing local_rank and deepspeed_config file location.
                model (aitoolbox.torchtrain.model.TTModel): neural network model
                add_model_attributes (list or tuple or None): additional TTModel attributes which need to be transferred
                    to the TTDeepSpeedLight level to enable their use in the transferred/exposed class methods
                default_model_methods (list or tuple): list of core methods which are present also in TTModel
                    abstract class
                **kwargs: additional parameters for the underlying ``deepspeed.DeepSpeedLight`` class

                    Possible arguments: https://deepspeed.readthedocs.io/en/latest/initialize.html
            """
            DeepSpeedLight.__init__(self, args, model, **kwargs)
            TTParallelBase.__init__(self, model, add_model_attributes, default_model_methods)


# class TTDataParallelExplicit(nn.DataParallel):
#     def __init__(self, module, add_model_attributes=None,
#                  default_model_methods=('get_loss', 'get_loss_eval', 'get_predictions'), **kwargs):
#         """torchtrain enabled DataParallel
#
#         This DataParallel wrapper works in the same way as the original PyTorch nn.DataParallel. Furthermore it exposes
#         TTModel batch data feeding definitions (additional abstract methods) to the TrainLoop while still enabling
#         multi GPU training.
#
#         Args:
#             module (aitoolbox.torchtrain.model.TTModel): neural network model
#             add_model_attributes (list or tuple or None): additional TTModel attributes which need to be transferred to
#                 the TTDataParallel level to enable their use in the transferred/exposed class methods
#             default_model_methods (list or tuple): list of core methods which are present also in TTModel abstract class
#             **kwargs: additional parameters for underlying nn.DataParallel
#         """
#         super().__init__(module, **kwargs)
#
#         # Core TTModel methods which every model has
#         self.get_loss_fn = copy_function(module.get_loss)
#         self.get_loss_eval_fn = copy_function(module.get_loss_eval)
#         self.get_predictions_fn = copy_function(module.get_predictions)
#
#         # Transfer any additional sub-methods from TTModel to TTDataParallel
#         methods = list(set(dir(module))
#                        - set(dir(nn.Module)) - set(vars(module)['_modules'].keys()) - set(default_model_methods))
#         additional_methods = [method_name for method_name in methods if callable(getattr(module, method_name, None))]
#
#         for method_name in additional_methods:
#             setattr(self, method_name,
#                     functools.partial(copy_function(getattr(module, method_name)), self))
#
#         # Optionally transfer additional TTModel attributes to the TTDataParallel level
#         if add_model_attributes is not None and isinstance(add_model_attributes, (list, tuple)):
#             for attr_name in add_model_attributes:
#                 setattr(self, attr_name, getattr(module, attr_name))
#
#     def get_loss(self, batch_data, criterion, device):
#         return self.get_loss_fn(self, batch_data, criterion, device)
#
#     def get_loss_eval(self, batch_data, criterion, device):
#         return self.get_loss_eval_fn(self, batch_data, criterion, device)
#
#     def get_predictions(self, batch_data, device):
#         return self.get_predictions_fn(self, batch_data, device)
#
#
# class TTDistributedDataParallelExplicit(DistributedDataParallel):
#     def __init__(self, module, add_model_attributes=None,
#                  default_model_methods=('get_loss', 'get_loss_eval', 'get_predictions'), **kwargs):
#         """torchtrain enabled DistributedDataParallel
#
#         Args:
#             module (aitoolbox.torchtrain.model.TTModel): neural network model
#             add_model_attributes (list or tuple or None): additional TTModel attributes which need to be transferred to
#                 the TTDistributedDataParallel level to enable their use in the transferred/exposed class methods
#             default_model_methods (list or tuple): list of core methods which are present also in TTModel abstract class
#             **kwargs: additional parameters for underlying nn.parallel.DistributedDataParallel
#         """
#         super().__init__(module, **kwargs)
#
#         # Core TTModel methods which every model has
#         self.get_loss_fn = copy_function(module.get_loss)
#         self.get_loss_eval_fn = copy_function(module.get_loss_eval)
#         self.get_predictions_fn = copy_function(module.get_predictions)
#
#         # Transfer any additional sub-methods from TTModel to TTDistributedDataParallel
#         methods = list(set(dir(module))
#                        - set(dir(nn.Module)) - set(vars(module)['_modules'].keys()) - set(default_model_methods))
#         additional_methods = [method_name for method_name in methods if callable(getattr(module, method_name, None))]
#
#         for method_name in additional_methods:
#             setattr(self, method_name,
#                     functools.partial(copy_function(getattr(module, method_name)), self))
#
#         # Optionally transfer additional TTModel attributes to the TTDistributedDataParallel level
#         if add_model_attributes is not None and isinstance(add_model_attributes, (list, tuple)):
#             for attr_name in add_model_attributes:
#                 setattr(self, attr_name, getattr(module, attr_name))
#
#     def get_loss(self, batch_data, criterion, device):
#         return self.get_loss_fn(self, batch_data, criterion, device)
#
#     def get_loss_eval(self, batch_data, criterion, device):
#         return self.get_loss_eval_fn(self, batch_data, criterion, device)
#
#     def get_predictions(self, batch_data, device):
#         return self.get_predictions_fn(self, batch_data, device)
#
#
# class TTApexDistributedDataParallelExplicit(ApexDistributedDataParallel):
#     def __init__(self, module, add_model_attributes=None,
#                  default_model_methods=('get_loss', 'get_loss_eval', 'get_predictions'), **kwargs):
#         """torchtrain enabled Nvidia Apex DistributedDataParallel
#
#         Args:
#             module (aitoolbox.torchtrain.model.TTModel): neural network model
#             add_model_attributes (list or tuple or None): additional TTModel attributes which need to be transferred to
#                 the TTDataParallel level to enable their use in the transferred/exposed class methods
#             default_model_methods (list or tuple): list of core methods which are present also in TTModel abstract class
#             **kwargs: additional parameters for underlying apex.parallel.DistributedDataParallel
#         """
#         super().__init__(module, **kwargs)
#
#         # Core TTModel methods which every model has
#         self.get_loss_fn = copy_function(module.get_loss)
#         self.get_loss_eval_fn = copy_function(module.get_loss_eval)
#         self.get_predictions_fn = copy_function(module.get_predictions)
#
#         # Transfer any additional sub-methods from TTModel to TTApexDistributedDataParallel
#         methods = list(set(dir(module))
#                        - set(dir(nn.Module)) - set(vars(module)['_modules'].keys()) - set(default_model_methods))
#         additional_methods = [method_name for method_name in methods if callable(getattr(module, method_name, None))]
#
#         for method_name in additional_methods:
#             setattr(self, method_name,
#                     functools.partial(copy_function(getattr(module, method_name)), self))
#
#         # Optionally transfer additional TTModel attributes to the TTApexDistributedDataParallel level
#         if add_model_attributes is not None and isinstance(add_model_attributes, (list, tuple)):
#             for attr_name in add_model_attributes:
#                 setattr(self, attr_name, getattr(module, attr_name))
#
#     def get_loss(self, batch_data, criterion, device):
#         return self.get_loss_fn(self, batch_data, criterion, device)
#
#     def get_loss_eval(self, batch_data, criterion, device):
#         return self.get_loss_eval_fn(self, batch_data, criterion, device)
#
#     def get_predictions(self, batch_data, device):
#         return self.get_predictions_fn(self, batch_data, device)
