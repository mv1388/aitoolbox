from abc import ABC, abstractmethod
import os
import time
import datetime

from aitoolbox.cloud.AWS.data_access import BaseDataSaver
from aitoolbox.experiment.local_save.local_results_save import LocalResultsSaver


class AbstractResultsSaver(ABC):
    @abstractmethod
    def save_experiment_results(self, result_package, training_history,
                                project_name, experiment_name, experiment_timestamp=None,
                                save_true_pred_labels=False, separate_files=False, protect_existing_folder=True):
        """
        
        Args:
            result_package (aitoolbox.experiment.result_package.abstract_result_packages.AbstractResultPackage):
            training_history (aitoolbox.experiment.training_history.TrainingHistory):
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


class BaseResultsSaver(BaseDataSaver):
    def __init__(self, bucket_name='model-result', cloud_dir_prefix=''):
        """Base experiment results saving to AWS S3 functionality

        Args:
            bucket_name (str): S3 bucket into which the files will be saved
            cloud_dir_prefix (str): destination folder path inside selected bucket
        """
        BaseDataSaver.__init__(self, bucket_name)
        self.cloud_dir_prefix = cloud_dir_prefix

    def create_experiment_cloud_storage_folder_structure(self, project_name, experiment_name, experiment_timestamp):
        """

        Args:
            project_name (str): root name of the project
            experiment_name (str): name of the particular experiment
            experiment_timestamp (str): time stamp at the start of training

        Returns:
            str: experiment cloud path
        """
        experiment_cloud_path = os.path.join(self.cloud_dir_prefix,
                                             project_name,
                                             experiment_name + '_' + experiment_timestamp,
                                             'results')
        return experiment_cloud_path


class S3ResultsSaver(AbstractResultsSaver, BaseResultsSaver):
    def __init__(self, bucket_name='model-result', cloud_dir_prefix='',
                 local_model_result_folder_path='~/project/model_result'):
        """AWS S3 results saver

        It first saves the results files to local drive and then uploads them to S3

        Args:
            bucket_name (str): name of the bucket in the S3 to which the results files will be saved
            cloud_dir_prefix (str): destination folder path inside selected bucket
            local_model_result_folder_path (str): root local path where project folder will be created
        """
        BaseResultsSaver.__init__(self, bucket_name, cloud_dir_prefix)
        self.local_results_saver = LocalResultsSaver(local_model_result_folder_path)

    def save_experiment_results(self, result_package, training_history,
                                project_name, experiment_name, experiment_timestamp=None,
                                save_true_pred_labels=False, separate_files=False, protect_existing_folder=True):
        """Save produced experiment results recorded in the result package to the results file on local drive and upload
            them to S3

        Args:
            result_package (aitoolbox.experiment.result_package.abstract_result_packages.AbstractResultPackage):
            training_history (aitoolbox.experiment.training_history.TrainingHistory):
            project_name (str): root name of the project
            experiment_name (str): name of the particular experiment
            experiment_timestamp (str or None): time stamp at the start of training
            save_true_pred_labels (bool): should ground truth labels also be saved
            separate_files (bool): should the results be saved in separate pickle files or should all of the results
                be batched together in a single results file
            protect_existing_folder (bool): can override potentially already existing folder or not

        Returns:
            (str, str): results file path on S3, experiment timestamp
        """
        if experiment_timestamp is None:
            experiment_timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S')

        if not separate_files:
            saved_local_results_details = self.local_results_saver.save_experiment_results(result_package,
                                                                                           training_history,
                                                                                           project_name,
                                                                                           experiment_name,
                                                                                           experiment_timestamp,
                                                                                           save_true_pred_labels,
                                                                                           protect_existing_folder)
        else:
            saved_local_results_details = \
                self.local_results_saver.save_experiment_results_separate_files(result_package,
                                                                                training_history,
                                                                                project_name,
                                                                                experiment_name,
                                                                                experiment_timestamp,
                                                                                save_true_pred_labels,
                                                                                protect_existing_folder)

        experiment_s3_path = self.create_experiment_cloud_storage_folder_structure(project_name, experiment_name, experiment_timestamp)

        for results_file_path_in_s3_results_dir, results_file_local_path in saved_local_results_details:
            results_file_s3_path = os.path.join(experiment_s3_path, results_file_path_in_s3_results_dir)
            self.save_file(local_file_path=results_file_local_path, cloud_file_path=results_file_s3_path)

        # saved_local_results_details[0][0] used to extract the main results file path which should be the first element
        # of the list with the support files' paths following
        main_results_s3_file_path = os.path.join(self.bucket_name, experiment_s3_path,
                                                 saved_local_results_details[0][0])

        return main_results_s3_file_path, experiment_timestamp
