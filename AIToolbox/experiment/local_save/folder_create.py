import os


class ExperimentFolderCreator:
    @staticmethod
    def create_experiment_base_folder(project_name, experiment_name, experiment_timestamp,
                                      local_model_result_folder_path):
        """

        Args:
            project_name (str):
            experiment_name (str):
            experiment_timestamp (str):
            local_model_result_folder_path (str):

        Returns:
            str:
        """
        project_path, experiment_path = \
            ExperimentFolderCreator.get_experiment_base_folder_paths(project_name, experiment_name,
                                                                     experiment_timestamp,
                                                                     local_model_result_folder_path)

        if not os.path.exists(project_path):
            os.mkdir(project_path)
        if not os.path.exists(experiment_path):
            os.mkdir(experiment_path)

        return experiment_path

    @staticmethod
    def get_experiment_base_folder_paths(project_name, experiment_name, experiment_timestamp,
                                         local_model_result_folder_path):
        """

        Args:
            project_name (str):
            experiment_name (str):
            experiment_timestamp (str):
            local_model_result_folder_path (str):

        Returns:
            str, str:
        """
        project_dir_path = os.path.join(os.path.expanduser(local_model_result_folder_path), project_name)
        experiment_dir_path = os.path.join(project_dir_path, experiment_name + '_' + experiment_timestamp)

        return project_dir_path, experiment_dir_path
