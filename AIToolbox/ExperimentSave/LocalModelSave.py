from abc import ABC, abstractmethod
import os
import time
import datetime


class AbstractLocalModelSaver(ABC):
    @abstractmethod
    def save_model(self, model, project_name, experiment_name, experiment_timestamp=None, protect_existing_folder=True):
        pass


class SmartLocalModelSaver:
    def __init__(self, local_model_result_folder_path='~/project/model_result'):
        """

        Args:
            local_model_result_folder_path (str):
        """
        self.local_model_result_folder_path = os.path.expanduser(local_model_result_folder_path)

    def create_experiment_local_folder_structure(self, project_name, experiment_name, experiment_timestamp):
        """

        Args:
            project_name (str):
            experiment_name (str):
            experiment_timestamp (str):

        Returns:
            str:
        """
        if not os.path.exists(os.path.join(self.local_model_result_folder_path, project_name)):
            os.mkdir(os.path.join(self.local_model_result_folder_path, project_name))

        experiment_path = os.path.join(self.local_model_result_folder_path, project_name,
                                       experiment_name + '_' + experiment_timestamp)

        if not os.path.exists(experiment_path):
            os.mkdir(experiment_path)

        experiment_model_path = os.path.join(experiment_path, 'model')
        os.mkdir(experiment_model_path)
        return experiment_model_path


class KerasLocalModelSaver(AbstractLocalModelSaver, SmartLocalModelSaver):
    def __init__(self, local_model_result_folder_path='~/project/model_result'):
        """

        Args:
            local_model_result_folder_path (str):
        """
        SmartLocalModelSaver.__init__(self, local_model_result_folder_path)

    def save_model(self, model, project_name, experiment_name, experiment_timestamp=None, protect_existing_folder=True):
        """

        Args:
            model (keras.engine.training.Model):
            project_name (str):
            experiment_name (str):
            experiment_timestamp (str or None):
            protect_existing_folder (bool):

        Returns:
            (str, str, str, str):

        """
        if experiment_timestamp is None:
            experiment_timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')

        experiment_model_local_path = self.create_experiment_local_folder_structure(project_name, experiment_name, experiment_timestamp)

        model_name = 'model_{}_{}.h5'.format(experiment_name, experiment_timestamp)
        model_local_path = os.path.join(experiment_model_local_path, model_name)
        model_weights_name = 'modelWeights_{}_{}.h5'.format(experiment_name, experiment_timestamp)
        model_weights_local_path = os.path.join(experiment_model_local_path, model_weights_name)

        model.save(model_local_path)
        model.save_weights(model_weights_local_path)

        return model_name, model_weights_name, model_local_path, model_weights_local_path


class TensorFlowLocalModelSaver(AbstractLocalModelSaver, SmartLocalModelSaver):
    def __init__(self, local_model_result_folder_path='~/project/model_result'):
        """

        Args:
            local_model_result_folder_path (str):
        """
        SmartLocalModelSaver.__init__(self, local_model_result_folder_path)

    def save_model(self, model, project_name, experiment_name, experiment_timestamp=None, protect_existing_folder=True):
        raise NotImplementedError


class PyTorchLocalModelSaver(AbstractLocalModelSaver, SmartLocalModelSaver):
    def __init__(self, local_model_result_folder_path='~/project/model_result'):
        """

        Args:
            local_model_result_folder_path (str):
        """
        SmartLocalModelSaver.__init__(self, local_model_result_folder_path)

    def save_model(self, model, project_name, experiment_name, experiment_timestamp=None, protect_existing_folder=True):
        raise NotImplementedError
