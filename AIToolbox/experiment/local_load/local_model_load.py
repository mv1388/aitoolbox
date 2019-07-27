from abc import ABC, abstractmethod
import os

import torch


class AbstractLocalModelLoader(ABC):
    @abstractmethod
    def load_model(self, model_name, project_name, experiment_name, **kwargs):
        """

        Args:
            model_name (str): model file name
            project_name (str): root name of the project
            experiment_name (str): name of the particular experiment
            **kwargs:

        Returns:

        """
        pass


class PyTorchLocalModelLoader(AbstractLocalModelLoader):
    def __init__(self, local_model_result_folder_path):
        """

        Args:
            local_model_result_folder_path (str): root local path where project folder will be created
        """
        self.local_model_result_folder_path = os.path.expanduser(local_model_result_folder_path)
        self.model_load = None

    def load_model(self, model_name, project_name, experiment_name, map_location=None):
        """

        Args:
            model_name (str): model file name
            project_name (str): root name of the project
            experiment_name (str): name of the particular experiment
            map_location (str or None):

        Returns:

        """
        # model_path = os.path.join(self.local_model_result_folder_path, project_name, experiment_name, model_name)
        model_path = os.path.join(self.local_model_result_folder_path, model_name)

        self.model_load = torch.load(model_path, map_location=map_location)

        return self.model_load

    def check_if_model_loaded(self):
        if self.model_load is None:
            raise ValueError('Model has not yet been loaded. Please call load_model() first.')

    def init_model(self, model):
        """

        Args:
            model:

        Returns:

        """
        self.check_if_model_loaded()

        model.load_state_dict(self.model_load['model_state_dict'])
        return model

    def init_optimizer(self, optimizer, device='cuda'):
        """

        Args:
            optimizer:
            device (str):

        Returns:

        """
        self.check_if_model_loaded()

        optimizer.load_state_dict(self.model_load['optimizer_state_dict'])

        # for state in optimizer.state.values():
        #     for k, v in state.items():
        #         if isinstance(v, torch.Tensor):
        #             state[k] = v.to(device)

        return optimizer
