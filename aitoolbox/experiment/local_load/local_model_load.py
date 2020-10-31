from abc import ABC, abstractmethod
import os
from collections import OrderedDict
import torch

from aitoolbox.experiment.local_save.folder_create import ExperimentFolder
from aitoolbox.torchtrain.schedulers.basic import AbstractScheduler


class AbstractLocalModelLoader(ABC):
    @abstractmethod
    def load_model(self, project_name, experiment_name, experiment_timestamp, model_save_dir, epoch_num=None, **kwargs):
        """Model loading method all the model loaders need to implement

        Args:
            project_name (str): root name of the project
            experiment_name (str): name of the particular experiment
            experiment_timestamp (str): time stamp at the start of training
            model_save_dir (str): name of the folder inside experiment folder where the model is saved
            epoch_num (int or None): epoch number of the model checkpoint or none if loading final model
            **kwargs: additional parameters for specific framework model loader

        Returns:
            model
        """
        pass


class PyTorchLocalModelLoader(AbstractLocalModelLoader):
    def __init__(self, local_model_result_folder_path):
        """PyTorch saved model loader and initializer

        Args:
            local_model_result_folder_path (str): root local path where project folder will be created
        """
        self.local_model_result_folder_path = os.path.expanduser(local_model_result_folder_path)
        self.model_representation = None

    def load_model(self, project_name, experiment_name, experiment_timestamp, model_save_dir='checkpoint_model',
                   epoch_num=None, map_location=None):
        """Model loading interface compatible with the experiment folder structure maintained by the AIToolbox TrainLoop

        Args:
            project_name (str): root name of the project
            experiment_name (str): name of the particular experiment
            experiment_timestamp (str): time stamp at the start of training
            model_save_dir (str): name of the folder inside experiment folder where the model is saved
            epoch_num (int or None): epoch number of the model checkpoint or none if loading final model
            map_location (str or None):

        Returns:
            model
        """
        _, experiment_dir_path = ExperimentFolder.get_base_folder_paths(project_name, experiment_name,
                                                                        experiment_timestamp,
                                                                        self.local_model_result_folder_path)
        if epoch_num is None:
            model_name = f'model_{experiment_name}_{experiment_timestamp}.pth'
        else:
            model_name = f'model_{experiment_name}_{experiment_timestamp}_E{epoch_num}.pth'

        model_path = os.path.join(experiment_dir_path, model_save_dir, model_name)

        self.model_representation = torch.load(model_path, map_location=map_location)

        # Fix for back-compatibility
        if 'schedulers_state_dict' not in self.model_representation:
            self.model_representation['schedulers_state_dict'] = []

        return self.model_representation

    def load_model_from_path(self, model_path, map_location=None):
        """General model loading when the AIToolbox TrainLoop experiment folder structure is not used

        Args:
            model_path (str): full path to the model
            map_location (str or None): a function, :class:`torch.device`, string or a dict specifying how to remap
                storage locations

        Returns:
            model
        """
        self.model_representation = torch.load(model_path, map_location=map_location)
        return self.model_representation

    def check_if_model_loaded(self):
        if self.model_representation is None:
            raise ValueError('Model has not yet been loaded. Please call load_model() first.')

    def init_model(self, model, used_data_parallel=False):
        """Initialize provided PyTorch model with the loaded model weights

        For this function to work, load_model() must be first called to read the model representation into memory.

        Args:
            model (TTModel or nn.Module): PyTorch model
            used_data_parallel (bool): if the saved model was nn.DataParallel or normal model

        Returns:
            PyTorch model
        """
        self.check_if_model_loaded()

        state_dict = self.model_representation['model_state_dict']

        if used_data_parallel:
            state_dict = OrderedDict()
            for k, v in self.model_representation['model_state_dict'].items():
                name = k[7:]  # remove `module.`
                state_dict[name] = v

        model.load_state_dict(state_dict)
        return model

    def init_optimizer(self, optimizer, device='cuda'):
        """Initialize the optimizer based on saved model/optimizer checkpoint

        Args:
            optimizer: PyTorch optimizer
            device (str): device id

        Returns:
            PyTorch optimizer
        """
        self.check_if_model_loaded()

        optimizer.load_state_dict(self.model_representation['optimizer_state_dict'])

        # for state in optimizer.state.values():
        #     for k, v in state.items():
        #         if isinstance(v, torch.Tensor):
        #             state[k] = v.to(device)

        return optimizer

    def init_scheduler(self, scheduler_callbacks_list, ignore_saved=False, ignore_missing_saved=False):
        """Initialize the list of schedulers based on saved model/optimizer/scheduler checkpoint

        Args:
            scheduler_callbacks_list (list): list of scheduler (callbacks)
            ignore_saved (bool): if exception should be raised in the case there are found scheduler snapshots
                in the checkpoint, but not schedulers are provided to this method
            ignore_missing_saved (bool): if exception should be raised in the case schedulers are provided to
                this method but no saved scheduler snapshots can be found in the checkpoint

        Returns:
            list: list of initialized scheduler (callbacks)
        """
        self.check_if_model_loaded()
        loaded_schedulers = self.model_representation['schedulers_state_dict']

        if len(loaded_schedulers) == 0 and len(scheduler_callbacks_list) > 0:
            if not ignore_missing_saved:
                raise KeyError('Schedulers_state_dict not found in the loaded model representation but you provided '
                               'schedulers to TrainLoop.')
            return scheduler_callbacks_list

        if len(loaded_schedulers) > 0 and len(scheduler_callbacks_list) == 0:
            if not ignore_saved:
                raise ValueError('No schedulers were provided to the TrainLoop, however scheduler state_dicts were'
                                 'found saved in the loaded model representation.')
            return scheduler_callbacks_list

        if len(scheduler_callbacks_list) != len(loaded_schedulers):
            raise ValueError('Number of provided schedulers does not match the number of loaded scheduler state_dicts. '
                             f'Number of given schedulers: {len(scheduler_callbacks_list)} and number of loaded'
                             f"scheduler state_dicts: {len(loaded_schedulers)}")

        # Initialize the scheduler callbacks with the saved scheduler states
        for sch_cb, sch_state_dict in zip(scheduler_callbacks_list, loaded_schedulers):
            if not isinstance(sch_cb, AbstractScheduler):
                raise TypeError('Provided scheduler is not inherited from AbstractScheduler')

            sch_cb.load_state_dict(sch_state_dict)

        return scheduler_callbacks_list

    def init_amp(self, amp_scaler):
        """Initialize AMP GradScaler

        Args:
            amp_scaler (torch.cuda.amp.GradScaler): AMP GradScaler

        Returns:
            torch.cuda.amp.GradScaler: initialized AMP GradScaler
        """
        amp_scaler.load_state_dict(self.model_representation['amp'])
        return amp_scaler
