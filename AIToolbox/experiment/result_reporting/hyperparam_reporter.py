import os
import shutil

from AIToolbox.experiment.local_save.local_results_save import BaseLocalResultsSaver as ResultsSaver
from AIToolbox.cloud.AWS.model_save import BaseModelSaver
from AIToolbox.cloud.AWS.results_save import BaseResultsSaver
from AIToolbox.cloud.GoogleCloud.model_save import BaseModelGoogleStorageSaver
from AIToolbox.cloud.GoogleCloud.results_save import BaseResultsGoogleStorageSaver


class HyperParameterReporter:
    def __init__(self, project_name, experiment_name, experiment_timestamp, local_model_result_folder_path):
        """

        Args:
            project_name (str): root name of the project
            experiment_name (str): name of the particular experiment
            experiment_timestamp (str): time stamp of the training start
            local_model_result_folder_path (str): root local path where project folder will be created
        """
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.experiment_timestamp = experiment_timestamp
        self.local_model_result_folder_path = os.path.expanduser(local_model_result_folder_path)

        _, self.experiment_dir_path, _ = ResultsSaver.form_experiment_local_folders_paths(project_name, experiment_name,
                                                                                          experiment_timestamp,
                                                                                          local_model_result_folder_path)
        empty_results_pth = ResultsSaver.create_experiment_local_folders(project_name, experiment_name,
                                                                         experiment_timestamp,
                                                                         local_model_result_folder_path)
        # Just a hack to remove the /results folder which gets created by create_experiment_local_folders,
        # but in this use case stays empty and is thus not needed
        shutil.rmtree(empty_results_pth)

        self.file_name = 'hyperparams_list.txt'
        self.local_hyperparams_file_path = os.path.join(self.experiment_dir_path, self.file_name)

    def save_hyperparams_to_text_file(self, hyperparams, sort_names=False):
        """

        Args:
            hyperparams (dict): hyper-parameters listed in the dict
            sort_names (bool): should presented hyper-param names be listed alphabetically

        Returns:
            str: path to the saved hyper-param text file
        """
        param_names = sorted(hyperparams.keys()) if sort_names else hyperparams.keys()

        with open(self.local_hyperparams_file_path, 'w') as f:
            for k in param_names:
                f.write(f'{k}:\t{hyperparams[k]}\n')

        return self.local_hyperparams_file_path

    def copy_to_cloud_storage(self, local_hyperparams_file_path, cloud_saver):
        """

        Args:
            local_hyperparams_file_path (str):
            cloud_saver (BaseModelSaver or BaseResultsSaver or BaseModelGoogleStorageSaver or BaseResultsGoogleStorageSaver):

        Returns:
            str: path where the file was saved in the cloud storage
        """
        cloud_folder_path = cloud_saver \
            .create_experiment_cloud_storage_folder_structure(self.project_name, self.experiment_name,
                                                              self.experiment_timestamp).rsplit('/', 1)[0]

        cloud_file_path = os.path.join(cloud_folder_path, self.file_name)
        cloud_saver.save_file(local_hyperparams_file_path, cloud_file_path)

        return cloud_file_path
