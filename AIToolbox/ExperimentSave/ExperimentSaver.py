from abc import ABC, abstractmethod
import time
import datetime

from AIToolbox.AWS.ModelSave import KerasS3ModelSaver
from AIToolbox.AWS.ResultsSave import S3ResultsSaver


class AbstractExperimentSaver(ABC):
    @abstractmethod
    def save_experiment(self, model, result_package, save_true_pred_labels=False, separate_files=False,
                        protect_existing_folder=True):
        pass


class FullKerasExperimentS3Saver(AbstractExperimentSaver):
    def __init__(self, project_name, experiment_name, experiment_timestamp=None,
                 bucket_name='model-result', local_model_result_folder_path='~/project/model_result'):
        """

        Args:
            project_name (str):
            experiment_name (str):
            experiment_timestamp (str):
            bucket_name (str):
            local_model_result_folder_path (str):
        """
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.experiment_timestamp = experiment_timestamp
        if self.experiment_timestamp is None:
            self.experiment_timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')

        self.keras_model_saver = KerasS3ModelSaver(bucket_name, local_model_result_folder_path)
        self.results_saver = S3ResultsSaver(bucket_name, local_model_result_folder_path)

    def save_experiment(self, model, result_package, save_true_pred_labels=False, separate_files=False,
                        protect_existing_folder=True):
        """

        Args:
            model (keras.engine.training.Model):
            result_package (AIToolbox.ExperimentSave.ResultPackage.AbstractResultPackage):
            save_true_pred_labels (bool):
            separate_files (bool):
            protect_existing_folder (bool):

        Returns:
            (str, str)
        """
        s3_model_path, _ = self.keras_model_saver.save_model(model,
                                                             self.project_name, self.experiment_name,
                                                             self.experiment_timestamp,
                                                             protect_existing_folder)
        s3_results_path, _ = self.results_saver.save_experiment_results(result_package,
                                                                        self.project_name, self.experiment_name, self.experiment_timestamp,
                                                                        save_true_pred_labels,separate_files,
                                                                        protect_existing_folder)
        return s3_model_path, s3_results_path
