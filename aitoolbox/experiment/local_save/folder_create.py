import os


class ExperimentFolder:
    @staticmethod
    def create_base_folder(project_name, experiment_name, experiment_timestamp,
                           local_model_result_folder_path):
        """Create local folder hierarchy for the experiment tracking

        Args:
            project_name (str): root name of the project
            experiment_name (str): name of the particular experiment
            experiment_timestamp (str): time stamp at the start of training
            local_model_result_folder_path (str): root local path where project folder will be created

        Returns:
            str: path to the created experiment base folder
        """
        project_path, experiment_path = \
            ExperimentFolder.get_base_folder_paths(project_name, experiment_name, experiment_timestamp,
                                                   local_model_result_folder_path)

        if not os.path.exists(project_path):
            os.mkdir(project_path)
        if not os.path.exists(experiment_path):
            os.mkdir(experiment_path)

        return experiment_path

    @staticmethod
    def get_base_folder_paths(project_name, experiment_name, experiment_timestamp,
                              local_model_result_folder_path):
        """Generate local folder hierarchy paths for the experiment tracking

        Does not actually create the folders, just generates the folder paths

        Args:
            project_name (str): root name of the project
            experiment_name (str): name of the particular experiment
            experiment_timestamp (str): time stamp at the start of training
            local_model_result_folder_path (str): root local path where project folder will be created

        Returns:
            str, str: path to the main project folder, path to the particular experiment folder
        """
        project_dir_path = os.path.join(os.path.expanduser(local_model_result_folder_path), project_name)
        experiment_dir_path = os.path.join(project_dir_path, experiment_name + '_' + experiment_timestamp)

        return project_dir_path, experiment_dir_path
