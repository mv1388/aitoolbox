from abc import ABC, abstractmethod
import boto3
import os
import time
import datetime

from AIToolbox.experiment_save.local_save.local_results_save import LocalResultsSaver


class AbstractResultsSaver(ABC):
    @abstractmethod
    def save_experiment_results(self, result_package, project_name, experiment_name, experiment_timestamp=None,
                                save_true_pred_labels=False, separate_files=False, protect_existing_folder=True):
        """
        
        Args:
            result_package (AIToolbox.experiment_save.result_package.abstract_result_packages.AbstractResultPackage):
            project_name (str):
            experiment_name (str):
            experiment_timestamp (str):
            save_true_pred_labels (bool):
            separate_files (bool):
            protect_existing_folder (bool):

        Returns:
            (str, str): results_file_s3_path, experiment_timestamp
        """
        pass


class BaseResultsSaver:
    def __init__(self, bucket_name='model-result', local_results_folder_path='~/project/model_result'):
        """

        Args:
            bucket_name (str):
            local_results_folder_path (str):
        """
        self.bucket_name = bucket_name
        self.s3_client = boto3.client('s3')
        self.s3_resource = boto3.resource('s3')

        self.local_model_result_folder_path = os.path.expanduser(local_results_folder_path)

    def save_file(self, local_file_path, cloud_file_path):
        """

        Args:
            local_file_path (str):
            cloud_file_path (str):

        Returns:
            None
        """
        self.s3_client.upload_file(os.path.expanduser(local_file_path),
                                   self.bucket_name, cloud_file_path)

    def create_experiment_cloud_storage_folder_structure(self, project_name, experiment_name, experiment_timestamp):
        """

        Args:
            project_name (str):
            experiment_name (str):
            experiment_timestamp (str):

        Returns:
            str:
        """
        experiment_cloud_path = os.path.join(project_name,
                                             experiment_name + '_' + experiment_timestamp,
                                             'results')
        return experiment_cloud_path


class S3ResultsSaver(AbstractResultsSaver, BaseResultsSaver):
    def __init__(self, bucket_name='model-result', local_model_result_folder_path='~/project/model_result'):
        """

        Args:
            bucket_name (str):
            local_model_result_folder_path (str):
        """
        BaseResultsSaver.__init__(self, bucket_name, local_model_result_folder_path)
        self.local_results_saver = LocalResultsSaver(local_model_result_folder_path)

    def save_experiment_results(self, result_package, project_name, experiment_name, experiment_timestamp=None,
                                save_true_pred_labels=False, separate_files=False, protect_existing_folder=True):
        """

        Args:
            result_package (AIToolbox.experiment_save.result_package.abstract_result_packages.AbstractResultPackage):
            project_name (str):
            experiment_name (str):
            experiment_timestamp (str):
            save_true_pred_labels (bool):
            separate_files (bool):
            protect_existing_folder (bool):

        Returns:
            (str, str):
        """
        if experiment_timestamp is None:
            experiment_timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')

        if not separate_files:
            saved_local_results_details = self.local_results_saver.save_experiment_results(result_package,
                                                                                           project_name,
                                                                                           experiment_name,
                                                                                           experiment_timestamp,
                                                                                           save_true_pred_labels,
                                                                                           protect_existing_folder)
        else:
            saved_local_results_details = \
                self.local_results_saver.save_experiment_results_separate_files(result_package,
                                                                                project_name,
                                                                                experiment_name,
                                                                                experiment_timestamp,
                                                                                save_true_pred_labels,
                                                                                protect_existing_folder)

        experiment_s3_path = self.create_experiment_cloud_storage_folder_structure(project_name, experiment_name, experiment_timestamp)

        for results_file_name, results_file_local_path in saved_local_results_details:
            results_file_s3_path = os.path.join(experiment_s3_path, results_file_name)
            self.save_file(local_file_path=results_file_local_path, cloud_file_path=results_file_s3_path)

        # saved_local_results_details[0][0] used to extract the main results file path which should be the first element
        # of the list with the support files' paths following
        main_results_s3_file_path = os.path.join(self.bucket_name, experiment_s3_path,
                                                 saved_local_results_details[0][0])

        return main_results_s3_file_path, experiment_timestamp
