from abc import ABC, abstractmethod
import os
import time
import datetime


class AbstractLocalModelSaver(ABC):
    @abstractmethod
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
            (str, str, str, str): model_name, model_weights_name, model_local_path, model_weights_local_path
        """
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
            (str, str, str, str): model_name, model_weights_name, model_local_path, model_weights_local_path
        """
        if experiment_timestamp is None:
            experiment_timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')

        experiment_model_local_path = self.create_experiment_local_folder_structure(project_name, experiment_name, experiment_timestamp)

        if epoch is None:
            model_name = f'model_{experiment_name}_{experiment_timestamp}.h5'
            model_weights_name = f'modelWeights_{experiment_name}_{experiment_timestamp}.h5'
        else:
            model_name = f'model_{experiment_name}_{experiment_timestamp}_E{epoch}.h5'
            model_weights_name = f'modelWeights_{experiment_name}_{experiment_timestamp}_E{epoch}.h5'

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

        raise NotImplementedError

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
            (str, str, str, str): model_name, model_weights_name, model_local_path, model_weights_local_path
        """
        if experiment_timestamp is None:
            experiment_timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')

        experiment_model_local_path = self.create_experiment_local_folder_structure(project_name, experiment_name, experiment_timestamp)

        if epoch is None:
            model_name = f'model_{experiment_name}_{experiment_timestamp}.pth'
            model_weights_name = f'modelWeights_{experiment_name}_{experiment_timestamp}.pth'
        else:
            model_name = f'model_{experiment_name}_{experiment_timestamp}_E{epoch}.pth'
            model_weights_name = f'modelWeights_{experiment_name}_{experiment_timestamp}_E{epoch}.pth'

        model_local_path = os.path.join(experiment_model_local_path, model_name)
        model_weights_local_path = os.path.join(experiment_model_local_path, model_weights_name)

        import torch
        torch.save(model, model_local_path)
        torch.save(model.state_dict(), model_weights_local_path)

        return model_name, model_weights_name, model_local_path, model_weights_local_path


class LocalSubOptimalModelRemover:
    def __init__(self, metric_name, num_best_kept=2):
        """

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
        """

        Args:
            history (AIToolbox.experiment_save.training_history.TrainingHistory):
            new_model_dump_paths (list):
            
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
            self.rm_suboptimal_model(model_paths_to_rm)

    @staticmethod
    def rm_suboptimal_model(rm_model_paths):
        """

        Args:
            rm_model_paths (list): list of string paths
            
        Returns:
            None
        """
        for rm_path in rm_model_paths:
            os.remove(rm_path)
