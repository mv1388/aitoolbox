from abc import ABC, abstractmethod
import os
import time
import datetime

from aitoolbox.experiment.local_save.folder_create import ExperimentFolder


class AbstractLocalModelSaver(ABC):
    @abstractmethod
    def save_model(self, model, project_name, experiment_name, experiment_timestamp=None,
                   epoch=None, iteration_idx=None,
                   protect_existing_folder=True):
        """Model saving method which all the model savers have to implement to give an expected API to other components
        
        Args:
            model (keras.Model or dict): model representation. If used with PyTorch it is a simple
                dict under the hood. In the case of Keras training this would be the keras Model.
            project_name (str): root name of the project
            experiment_name (str): name of the particular experiment
            experiment_timestamp (str or None): time stamp at the start of training
            epoch (int or None): in which epoch the model is being saved
            iteration_idx (int or None): at which training iteration the model is being saved
            protect_existing_folder (bool): can override potentially already existing folder or not

        Returns:
            (str, str): model_name, model_local_path
        """
        pass


class BaseLocalModelSaver:
    def __init__(self, local_model_result_folder_path='~/project/model_result',
                 checkpoint_model=False):
        """Base functionality for all the local model savers

        Args:
            local_model_result_folder_path (str): root local path where project folder will be created
            checkpoint_model (bool): if the model is coming from the mid-training checkpoint
        """
        self.local_model_result_folder_path = os.path.expanduser(local_model_result_folder_path)
        self.checkpoint_model = checkpoint_model

    def create_experiment_local_models_folder(self, project_name, experiment_name, experiment_timestamp):
        """Creates experiment local folder hierarchy and place the 'models' folder in it

        Args:
            project_name (str): root name of the project
            experiment_name (str): name of the particular experiment
            experiment_timestamp (str): time stamp at the start of training

        Returns:
            str: path to the created models folder in the experiment base folder
        """
        experiment_path = ExperimentFolder.create_base_folder(project_name, experiment_name, experiment_timestamp,
                                                              self.local_model_result_folder_path)

        experiment_model_path = os.path.join(experiment_path,
                                             'model' if not self.checkpoint_model else 'checkpoint_model')
        if not os.path.exists(experiment_model_path):
            os.mkdir(experiment_model_path)

        return experiment_model_path


class PyTorchLocalModelSaver(AbstractLocalModelSaver, BaseLocalModelSaver):
    def __init__(self, local_model_result_folder_path='~/project/model_result',
                 checkpoint_model=False):
        """PyTorch experiment local model saver

        Args:
            local_model_result_folder_path (str): root local path where project folder will be created
            checkpoint_model (bool): if the model is coming from the mid-training checkpoint
        """
        BaseLocalModelSaver.__init__(self, local_model_result_folder_path, checkpoint_model)

    def save_model(self, model, project_name, experiment_name, experiment_timestamp=None,
                   epoch=None, iteration_idx=None,
                   protect_existing_folder=True):
        """Save the PyTorch model representation dict to the local drive

        Args:
            model (dict): PyTorch model represented as a dict of weights, optimizer state and other necessary info.
            project_name (str): root name of the project
            experiment_name (str): name of the particular experiment
            experiment_timestamp (str or None): time stamp at the start of training
            epoch (int or None): in which epoch the model is being saved
            iteration_idx (int or None): at which training iteration the model is being saved
            protect_existing_folder (bool): can override potentially already existing folder or not

        Returns:
            (str, str): model_name, model_local_path
        """
        self.check_model_dict_contents(model)

        if experiment_timestamp is None:
            experiment_timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S')

        experiment_model_local_path = self.create_experiment_local_models_folder(project_name, experiment_name,
                                                                                 experiment_timestamp)
        iteration_suffix = f'_ITER{iteration_idx}' if iteration_idx is not None else ''

        if epoch is None:
            model_name = f'model_{experiment_name}_{experiment_timestamp}.pth'
        else:
            model_name = f'model_{experiment_name}_{experiment_timestamp}_E{epoch}{iteration_suffix}.pth'

        model_local_path = os.path.join(experiment_model_local_path, model_name)

        import torch
        torch.save(model, model_local_path)

        return model_name, model_local_path

    @staticmethod
    def check_model_dict_contents(model):
        """Check if PyTorch model save dict contains all the necessary elements for the training state reconstruction

        Args:
            model (dict): PyTorch model represented as a dict of weights, optimizer state and other necessary info.

        Raises:
            ValueError

        Returns:
            None
        """
        for required_element in ['model_state_dict', 'optimizer_state_dict', 'epoch', 'hyperparams']:
            if required_element not in model:
                raise ValueError(f'Required element of the model dict {required_element} is missing. Given model'
                                 f'dict has the following elements: {model.keys()}')


class KerasLocalModelSaver(AbstractLocalModelSaver, BaseLocalModelSaver):
    def __init__(self, local_model_result_folder_path='~/project/model_result',
                 checkpoint_model=False):
        """Keras experiment local model saver

        Args:
            local_model_result_folder_path (str): root local path where project folder will be created
            checkpoint_model (bool): if the model is coming from the mid-training checkpoint
        """
        BaseLocalModelSaver.__init__(self, local_model_result_folder_path, checkpoint_model)

    def save_model(self, model, project_name, experiment_name, experiment_timestamp=None,
                   epoch=None, iteration_idx=None,
                   protect_existing_folder=True):
        """Save the Keras model to the local drive

        Args:
            model (keras.Model): Keras model
            project_name (str): root name of the project
            experiment_name (str): name of the particular experiment
            experiment_timestamp (str or None): time stamp at the start of training
            epoch (int or None): in which epoch the model is being saved
            iteration_idx (int or None): at which training iteration the model is being saved
            protect_existing_folder (bool): can override potentially already existing folder or not

        Returns:
            (str, str): model_name, model_local_path
        """
        if experiment_timestamp is None:
            experiment_timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S')

        experiment_model_local_path = self.create_experiment_local_models_folder(project_name, experiment_name,
                                                                                 experiment_timestamp)
        iteration_suffix = f'_ITER{iteration_idx}' if iteration_idx is not None else ''

        if epoch is None:
            model_name = f'model_{experiment_name}_{experiment_timestamp}.h5'
        else:
            model_name = f'model_{experiment_name}_{experiment_timestamp}_E{epoch}{iteration_suffix}.h5'

        model_local_path = os.path.join(experiment_model_local_path, model_name)

        model.save(model_local_path)

        return model_name, model_local_path


class LocalSubOptimalModelRemover:
    def __init__(self, metric_name, num_best_kept=2):
        """Removes the tracked saved models which become suboptimal when new models are trained in subsequent epochs

        Useful when interested in saving the limited local disk space, especially when dealing with large model which
        take a lot of disk space.

        Args:
            metric_name (str): one of the metric names that will be calculated and will appear in the train_history dict
                in the TrainLoop
            num_best_kept (int): number of best performing models which are kept when removing suboptimal model
                checkpoints
        """
        self.metric_name = metric_name
        self.decrease_metric = 'loss' in metric_name

        self.num_best_kept = num_best_kept

        self.default_metrics_list = ['loss', 'accumulated_loss', 'val_loss']
        self.is_default_metric = metric_name in self.default_metrics_list
        self.non_default_metric_buffer = None

        self.model_save_history = []
        
    def decide_if_remove_suboptimal_model(self, history, new_model_dump_paths):
        """Make decision if suboptimal model should be removed due to the introduction of the new and better model

        Args:
            history (aitoolbox.experiment.training_history.TrainingHistory): training performance history
            new_model_dump_paths (list): new saved models paths which will begin to be tracked
            
        Returns:
            None
        """
        if not self.is_default_metric:
            if self.non_default_metric_buffer is not None:
                if self.metric_name in history:
                    self.model_save_history.append((self.non_default_metric_buffer, history[self.metric_name][-1]))
                else:
                    print(f'Provided metric {self.metric_name} not found on the list of evaluated metrics: {history.keys()}')
                    
            self.non_default_metric_buffer = new_model_dump_paths
        else:
            self.model_save_history.append((new_model_dump_paths, history[self.metric_name][-1]))

        if len(self.model_save_history) > self.num_best_kept:
            self.model_save_history = sorted(self.model_save_history, key=lambda x: x[1], reverse=not self.decrease_metric)

            model_paths_to_rm, _ = self.model_save_history.pop()

            print(f'Removing suboptimal models. Paths to be removed: {model_paths_to_rm}')
            self.rm_suboptimal_model(model_paths_to_rm)

    @staticmethod
    def rm_suboptimal_model(rm_model_paths):
        """Utility to remove the file

        Args:
            rm_model_paths (list): list of string paths
            
        Returns:
            None
        """
        for rm_path in rm_model_paths:
            os.remove(rm_path)
