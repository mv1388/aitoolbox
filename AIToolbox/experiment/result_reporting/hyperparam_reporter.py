import os
from shutil import copyfile

from AIToolbox.experiment.local_save.folder_create import ExperimentFolderCreator as FolderCreator
from AIToolbox.cloud.AWS.model_save import BaseModelSaver
from AIToolbox.cloud.AWS.results_save import BaseResultsSaver
from AIToolbox.cloud.GoogleCloud.model_save import BaseModelGoogleStorageSaver
from AIToolbox.cloud.GoogleCloud.results_save import BaseResultsGoogleStorageSaver


class HyperParameterReporter:
    def __init__(self, project_name, experiment_name, experiment_timestamp, local_model_result_folder_path):
        """Writer of selected hyperparameters to human-readable text file on disk

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

        self.experiment_dir_path = FolderCreator.create_experiment_base_folder(project_name, experiment_name,
                                                                               experiment_timestamp,
                                                                               local_model_result_folder_path)

        self.file_name = 'hyperparams_list.txt'
        self.local_hyperparams_file_path = os.path.join(self.experiment_dir_path, self.file_name)

    def save_hyperparams_to_text_file(self, hyperparams, sort_names=False):
        """Save hyperparameters dict into text file on disk

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

    def copy_to_cloud_storage(self, local_hyperparams_file_path, cloud_saver, file_name=None):
        """Copy saved text local file into cloud storage

        Args:
            local_hyperparams_file_path (str): path to hyperparams file stored on local disk. File to be uploaded to cloud
            cloud_saver (BaseModelSaver or BaseResultsSaver or BaseModelGoogleStorageSaver or BaseResultsGoogleStorageSaver):
            file_name (str or None): manually specify the file name to be saved to the cloud instead of taking
                the default from self.file_name

        Returns:
            str: path where the file was saved in the cloud storage
        """
        cloud_folder_path = cloud_saver \
            .create_experiment_cloud_storage_folder_structure(self.project_name, self.experiment_name,
                                                              self.experiment_timestamp).rsplit('/', 1)[0]

        cloud_file_path = os.path.join(cloud_folder_path, self.file_name if file_name is None else file_name)
        cloud_saver.save_file(local_hyperparams_file_path, cloud_file_path)

        return cloud_file_path

    def save_experiment_python_file(self, hyperparams):
        """Saves the python experiment file to the project folder

        Python experiment file is file in which the main training procedure is defined. File from which the TrainLoop
        is executed

        Args:
            hyperparams (dict): hyper-parameters listed in the dict

        Returns:
            str: path to the saved hyper-param text file
        """
        if 'experiment_file_path' in hyperparams:
            destination_file_path = os.path.join(self.experiment_dir_path,
                                                 os.path.basename(hyperparams['experiment_file_path']))
            copyfile(hyperparams['experiment_file_path'], destination_file_path)
            return destination_file_path
        else:
            print('experiment_file_path experiment execution file path missing in the hyperparams dict. '
                  'Consequently not copying the file to the experiment dir.')
