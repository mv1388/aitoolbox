from abc import ABC, abstractmethod
import os
import time
import datetime

import torch


class AbstractLocalModelSaver(ABC):
    @abstractmethod
    def save_model(self, model, project_name, experiment_name, experiment_timestamp=None, epoch=None, protect_existing_folder=True):
        pass


class BaseLocalModelSaver:
    def __init__(self, local_model_result_folder_path='~/project/model_result',
                 checkpoint_model=False):
        """

        Args:
            local_model_result_folder_path (str):
            checkpoint_model (bool):
        """
        self.local_model_result_folder_path = os.path.expanduser(local_model_result_folder_path)
        self.checkpoint_model = checkpoint_model

    def create_experiment_local_folder_structure(self, project_name, experiment_name, experiment_timestamp):
        """

        Args:
            project_name (str):
            experiment_name (str):
            experiment_timestamp (str):

        Returns:
            str:
        """
        project_path = os.path.join(self.local_model_result_folder_path, project_name)
        if not os.path.exists(project_path):
            os.mkdir(project_path)

        experiment_path = os.path.join(self.local_model_result_folder_path, project_name,
                                       experiment_name + '_' + experiment_timestamp)
        if not os.path.exists(experiment_path):
            os.mkdir(experiment_path)

        experiment_model_path = os.path.join(experiment_path,
                                             'model' if not self.checkpoint_model else 'checkpoint_model')
        if not os.path.exists(experiment_model_path):
            os.mkdir(experiment_model_path)

        return experiment_model_path


class KerasLocalModelSaver(AbstractLocalModelSaver, BaseLocalModelSaver):
    def __init__(self, local_model_result_folder_path='~/project/model_result',
                 checkpoint_model=False):
        """

        Args:
            local_model_result_folder_path (str):
            checkpoint_model (bool):
        """
        BaseLocalModelSaver.__init__(self, local_model_result_folder_path, checkpoint_model)

    def save_model(self, model, project_name, experiment_name, experiment_timestamp=None, epoch=None, protect_existing_folder=True):
        """

        Args:
            model (keras.engine.training.Model):
            project_name (str):
            experiment_name (str):
            experiment_timestamp (str or None):
            epoch (int or None):
            protect_existing_folder (bool):

        Returns:
            (str, str, str, str):

        """
        if experiment_timestamp is None:
            experiment_timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')

        experiment_model_local_path = self.create_experiment_local_folder_structure(project_name, experiment_name, experiment_timestamp)

        if epoch is None:
            model_name = 'model_{}_{}.h5'.format(experiment_name, experiment_timestamp)
            model_weights_name = 'modelWeights_{}_{}.h5'.format(experiment_name, experiment_timestamp)
        else:
            model_name = 'model_{}_{}_E{}.h5'.format(experiment_name, experiment_timestamp, epoch)
            model_weights_name = 'modelWeights_{}_{}_E{}.h5'.format(experiment_name, experiment_timestamp, epoch)

        model_local_path = os.path.join(experiment_model_local_path, model_name)
        model_weights_local_path = os.path.join(experiment_model_local_path, model_weights_name)

        model.save(model_local_path)
        model.save_weights(model_weights_local_path)

        return model_name, model_weights_name, model_local_path, model_weights_local_path


class TensorFlowLocalModelSaver(AbstractLocalModelSaver, BaseLocalModelSaver):
    def __init__(self, local_model_result_folder_path='~/project/model_result',
                 checkpoint_model=False):
        """

        Args:
            local_model_result_folder_path (str):
            checkpoint_model (bool):
        """
        BaseLocalModelSaver.__init__(self, local_model_result_folder_path, checkpoint_model)

    def save_model(self, model, project_name, experiment_name, experiment_timestamp=None, epoch=None, protect_existing_folder=True):
        raise NotImplementedError


class PyTorchLocalModelSaver(AbstractLocalModelSaver, BaseLocalModelSaver):
    def __init__(self, local_model_result_folder_path='~/project/model_result',
                 checkpoint_model=False):
        """

        Args:
            local_model_result_folder_path (str):
            checkpoint_model (bool):
        """
        BaseLocalModelSaver.__init__(self, local_model_result_folder_path, checkpoint_model)

    def save_model(self, model, project_name, experiment_name, experiment_timestamp=None, epoch=None, protect_existing_folder=True):
        """

        Args:
            model (torch.nn.modules.Module):
            project_name (str):
            experiment_name (str):
            experiment_timestamp (str or None):
            epoch (int or None):
            protect_existing_folder (bool):

        Returns:
            (str, str, str, str):

        """
        if experiment_timestamp is None:
            experiment_timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')

        experiment_model_local_path = self.create_experiment_local_folder_structure(project_name, experiment_name, experiment_timestamp)

        if epoch is None:
            model_name = 'model_{}_{}.pth'.format(experiment_name, experiment_timestamp)
            model_weights_name = 'modelWeights_{}_{}.pth'.format(experiment_name, experiment_timestamp)
        else:
            model_name = 'model_{}_{}_E{}.pth'.format(experiment_name, experiment_timestamp, epoch)
            model_weights_name = 'modelWeights_{}_{}_E{}.pth'.format(experiment_name, experiment_timestamp, epoch)

        model_local_path = os.path.join(experiment_model_local_path, model_name)
        model_weights_local_path = os.path.join(experiment_model_local_path, model_weights_name)

        torch.save(model, model_local_path)
        torch.save(model.state_dict(), model_weights_local_path)

        return model_name, model_weights_name, model_local_path, model_weights_local_path
