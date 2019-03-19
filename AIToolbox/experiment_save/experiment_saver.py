from abc import ABC, abstractmethod
import time
import datetime

from AIToolbox.AWS.model_save import KerasS3ModelSaver, PyTorchS3ModelSaver
from AIToolbox.AWS.results_save import S3ResultsSaver
from AIToolbox.experiment_save.local_model_save import PyTorchLocalModelSaver, KerasLocalModelSaver, TensorFlowLocalModelSaver
from AIToolbox.experiment_save.local_results_save import LocalResultsSaver


class AbstractExperimentSaver(ABC):
    @abstractmethod
    def save_experiment(self, model, result_package, experiment_timestamp=None,
                        save_true_pred_labels=False, separate_files=False,
                        protect_existing_folder=True):
        pass


class FullKerasExperimentS3Saver(AbstractExperimentSaver):
    def __init__(self, project_name, experiment_name,
                 bucket_name='model-result', local_model_result_folder_path='~/project/model_result'):
        """

        Args:
            project_name (str):
            experiment_name (str):
            bucket_name (str):
            local_model_result_folder_path (str):
        """
        self.project_name = project_name
        self.experiment_name = experiment_name

        self.keras_model_saver = KerasS3ModelSaver(bucket_name=bucket_name,
                                                   local_model_result_folder_path=local_model_result_folder_path)
        self.results_saver = S3ResultsSaver(bucket_name=bucket_name,
                                            local_model_result_folder_path=local_model_result_folder_path)

    def save_experiment(self, model, result_package, experiment_timestamp=None,
                        save_true_pred_labels=False, separate_files=False,
                        protect_existing_folder=True):
        """

        Args:
            model (keras.engine.training.Model):
            result_package (AIToolbox.ExperimentSave.result_package.AbstractResultPackage):
            experiment_timestamp (str):
            save_true_pred_labels (bool):
            separate_files (bool):
            protect_existing_folder (bool):

        Returns:
            (str, str)
        """
        if experiment_timestamp is None:
            experiment_timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')

        s3_model_path, _ = self.keras_model_saver.save_model(model=model,
                                                             project_name=self.project_name,
                                                             experiment_name=self.experiment_name,
                                                             experiment_timestamp=experiment_timestamp,
                                                             protect_existing_folder=protect_existing_folder)

        s3_results_path, _ = self.results_saver.save_experiment_results(result_package=result_package,
                                                                        project_name=self.project_name,
                                                                        experiment_name=self.experiment_name,
                                                                        experiment_timestamp=experiment_timestamp,
                                                                        save_true_pred_labels=save_true_pred_labels,
                                                                        separate_files=separate_files,
                                                                        protect_existing_folder=protect_existing_folder)
        return s3_model_path, s3_results_path


class FullPyTorchExperimentS3Saver(AbstractExperimentSaver):
    def __init__(self, project_name, experiment_name,
                 bucket_name='model-result', local_model_result_folder_path='~/project/model_result'):
        """

        Args:
            project_name (str):
            experiment_name (str):
            bucket_name (str):
            local_model_result_folder_path (str):
        """
        self.project_name = project_name
        self.experiment_name = experiment_name

        self.pytorch_model_saver = PyTorchS3ModelSaver(bucket_name=bucket_name,
                                                       local_model_result_folder_path=local_model_result_folder_path)
        self.results_saver = S3ResultsSaver(bucket_name=bucket_name,
                                            local_model_result_folder_path=local_model_result_folder_path)

    def save_experiment(self, model, result_package, experiment_timestamp=None,
                        save_true_pred_labels=False, separate_files=False,
                        protect_existing_folder=True):
        """

        Args:
            model (torch.nn.modules.Module):
            result_package (AIToolbox.ExperimentSave.result_package.AbstractResultPackage):
            experiment_timestamp (str):
            save_true_pred_labels (bool):
            separate_files (bool):
            protect_existing_folder (bool):

        Returns:
            (str, str)
        """
        if experiment_timestamp is None:
            experiment_timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')

        s3_model_path, _ = self.pytorch_model_saver.save_model(model=model,
                                                               project_name=self.project_name,
                                                               experiment_name=self.experiment_name,
                                                               experiment_timestamp=experiment_timestamp,
                                                               protect_existing_folder=protect_existing_folder)

        s3_results_path, _ = self.results_saver.save_experiment_results(result_package=result_package,
                                                                        project_name=self.project_name,
                                                                        experiment_name=self.experiment_name,
                                                                        experiment_timestamp=experiment_timestamp,
                                                                        save_true_pred_labels=save_true_pred_labels,
                                                                        separate_files=separate_files,
                                                                        protect_existing_folder=protect_existing_folder)
        return s3_model_path, s3_results_path


class FullPyTorchExperimentLocalSaver(AbstractExperimentSaver):
    def __init__(self, project_name, experiment_name, local_model_result_folder_path='~/project/model_result'):
        """

        Args:
            project_name (str):
            experiment_name (str):
            local_model_result_folder_path (str):
        """
        self.project_name = project_name
        self.experiment_name = experiment_name
        
        self.pytorch_model_saver = PyTorchLocalModelSaver(local_model_result_folder_path)
        self.results_saver = LocalResultsSaver(local_model_result_folder_path)
    
    def save_experiment(self, model, result_package, experiment_timestamp=None,
                        save_true_pred_labels=False, separate_files=False,
                        protect_existing_folder=True):
        """

        Args:
            model (torch.nn.modules.Module):
            result_package (AIToolbox.ExperimentSave.result_package.AbstractResultPackage):
            experiment_timestamp (str):
            save_true_pred_labels (bool):
            separate_files (bool):
            protect_existing_folder (bool):

        Returns:
            list:
        """
        if experiment_timestamp is None:
            experiment_timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')
            
        _, _, model_local_path, model_weights_local_path = \
            self.pytorch_model_saver.save_model(model=model,
                                                project_name=self.project_name,
                                                experiment_name=self.experiment_name,
                                                experiment_timestamp=experiment_timestamp,
                                                protect_existing_folder=protect_existing_folder)
        
        saved_paths = [model_local_path, model_weights_local_path]
        
        if not separate_files:
            _, results_file_local_path = \
                self.results_saver.save_experiment_results(result_package=result_package,
                                                           project_name=self.project_name,
                                                           experiment_name=self.experiment_name,
                                                           experiment_timestamp=experiment_timestamp,
                                                           save_true_pred_labels=save_true_pred_labels,
                                                           protect_existing_folder=protect_existing_folder)
            saved_paths.append(results_file_local_path)
            
        else:
            saved_results_paths = \
                self.results_saver.save_experiment_results_separate_files(result_package=result_package,
                                                                          project_name=self.project_name,
                                                                          experiment_name=self.experiment_name,
                                                                          experiment_timestamp=experiment_timestamp,
                                                                          save_true_pred_labels=save_true_pred_labels,
                                                                          protect_existing_folder=protect_existing_folder)
            saved_paths += [path for _, path in saved_results_paths]
            
        return saved_paths
