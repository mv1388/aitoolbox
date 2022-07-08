import os
import shutil

from aitoolbox.experiment.local_save.folder_create import ExperimentFolder as FolderCreator
from aitoolbox.utils.file_system import zip_folder
from aitoolbox.cloud.AWS.model_save import BaseModelSaver
from aitoolbox.cloud.AWS.results_save import BaseResultsSaver
from aitoolbox.cloud.GoogleCloud.model_save import BaseModelGoogleStorageSaver
from aitoolbox.cloud.GoogleCloud.results_save import BaseResultsGoogleStorageSaver


class HyperParamSourceReporter:
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

        self.experiment_dir_path = FolderCreator.create_base_folder(project_name, experiment_name, experiment_timestamp,
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
            local_hyperparams_file_path (str): path to hyperparams file stored on local disk.
                File to be uploaded to cloud
            cloud_saver (BaseModelSaver or BaseResultsSaver or BaseModelGoogleStorageSaver or BaseResultsGoogleStorageSaver):
                cloud saver object
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
            hyperparams (dict): hyper-parameters listed in the dict. In order for this function to work, the dict needs
                to include `experiment_file_path` key.

        Returns:
            str: path to the saved main python experiment file
        """
        if 'experiment_file_path' in hyperparams:
            try:
                destination_file_path = os.path.join(self.experiment_dir_path,
                                                     os.path.basename(hyperparams['experiment_file_path']))
                shutil.copyfile(hyperparams['experiment_file_path'], destination_file_path)
                return destination_file_path
            except FileNotFoundError:
                print('experiment_file_path leading to the non-existent file. Possibly this error is related to'
                      'problematic automatic experiment python file path deduction when running the TrainLoop from'
                      'jupyter notebook. When using jupyter notebook you should manually specify the experiment'
                      'python file path as the value for experiment_file_path in hyperparams dict.')
        else:
            print('experiment_file_path experiment execution file path missing in the hyperparams dict. '
                  'Consequently not copying the file to the experiment dir.')

    def save_experiment_source_files(self, hyperparams):
        """Saves all the experiment source files into single source code zip

        Args:
            hyperparams (dict): hyper-parameters listed in the dict. In order for this function to work, the dict needs
                to include `source_dirs_paths` key.

        Returns:
            str: path to the saved experiment source code zip
        """
        if 'source_dirs_paths' in hyperparams and \
                type(hyperparams['source_dirs_paths']) in [list, tuple] and len(hyperparams['source_dirs_paths']) > 0:
            src_dir_in_project_dir = os.path.join(self.experiment_dir_path, 'source_code')
            os.mkdir(src_dir_in_project_dir)

            for src_dir_path in hyperparams['source_dirs_paths']:
                src_dir_path = os.path.expanduser(src_dir_path)
                destination_file_path = os.path.join(src_dir_in_project_dir, os.path.basename(src_dir_path))
                shutil.copytree(src_dir_path, destination_file_path)

            zip_path = zip_folder(src_dir_in_project_dir, src_dir_in_project_dir)
            shutil.rmtree(src_dir_in_project_dir)

            return zip_path
