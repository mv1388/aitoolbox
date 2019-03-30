from abc import ABC, abstractmethod
import os
import time
import datetime
import pickle
import json


class AbstractLocalResultsSaver(ABC):
    @abstractmethod
    def save_experiment_results(self, result_package, project_name, experiment_name, experiment_timestamp=None,
                                save_true_pred_labels=False, protect_existing_folder=True):
        """
        
        Args:
            result_package (AIToolbox.experiment_save.result_package.abstract_result_packages.AbstractResultPackage):
            project_name (str):
            experiment_name (str):
            experiment_timestamp (str or None):
            save_true_pred_labels (bool):
            protect_existing_folder (bool):

        Returns:
            list: list of list with this format: [[results_file_name, results_file_local_path], ... [ , ]] 
                Each file should be a new list specifying the file name and its full path
                
                The first file path should be pointing to the main experiment results file.
        """
        pass

    @abstractmethod
    def save_experiment_results_separate_files(self, result_package, project_name, experiment_name, experiment_timestamp,
                                               save_true_pred_labels=False, protect_existing_folder=True):
        """
        
        Args:
            result_package (AIToolbox.experiment_save.result_package.abstract_result_packages.AbstractResultPackage):
            project_name (str):
            experiment_name (str):
            experiment_timestamp (str or None):
            save_true_pred_labels (bool):
            protect_existing_folder (bool):

        Returns:
            list: list of list with this format: [[results_file_name, results_file_local_path], ... [ , ]] 
                Each file should be a new list specifying the file name and its full path
                
                The first file path should be pointing to the main experiment results file.
        """
        pass


class BaseLocalResultsSaver:
    def __init__(self, local_model_result_folder_path='~/project/model_result', file_format='pickle'):
        """

        Args:
            local_model_result_folder_path (str):
            file_format (str): pickle or json
        """
        self.local_model_result_folder_path = os.path.expanduser(local_model_result_folder_path)
        self.file_format = file_format

        if self.file_format not in ['pickle', 'json']:
            print('Warning: file format should be: pickle or json. Setting it to default pickle')
            self.file_format = 'pickle'

    def create_experiment_local_folder_structure(self, project_name, experiment_name, experiment_timestamp):
        """

        Args:
            project_name (str):
            experiment_name (str):
            experiment_timestamp (str):

        Returns:
            str:
        """
        project_path, experiment_path, experiment_results_path = \
            self.form_experiment_local_folders_paths(project_name, experiment_name, experiment_timestamp,
                                                     self.local_model_result_folder_path)
        if not os.path.exists(project_path):
            os.mkdir(project_path)
        if not os.path.exists(experiment_path):
            os.mkdir(experiment_path)
        os.mkdir(experiment_results_path)
        return experiment_results_path

    @staticmethod
    def form_experiment_local_folders_paths(project_name, experiment_name, experiment_timestamp,
                                            local_model_result_folder_path):
        """

        Args:
            project_name (str):
            experiment_name (str):
            experiment_timestamp (str):
            local_model_result_folder_path (str):

        Returns:
            str, str, str: project_dir_path, experiment_dir_path, experiment_results_dir_path
        """
        project_dir_path = os.path.join(local_model_result_folder_path, project_name)
        experiment_dir_path = os.path.join(project_dir_path, experiment_name + '_' + experiment_timestamp)
        experiment_results_dir_path = os.path.join(experiment_dir_path, 'results')

        return project_dir_path, experiment_dir_path, experiment_results_dir_path

    def save_file(self, result_dict, file_name_w_type, file_local_path_w_type):
        """

        Args:
            result_dict (dict):
            file_name_w_type (str):
            file_local_path_w_type (str):

        Returns:
            (str, str)
        """
        if self.file_format == 'pickle':
            file_name = file_name_w_type + '.p'
            file_local_path = file_local_path_w_type + '.p'
            with open(file_local_path, 'wb') as f:
                pickle.dump(result_dict, f)
            return file_name, file_local_path
        elif self.file_format == 'json':
            file_name = file_name_w_type + '.json'
            file_local_path = file_local_path_w_type + '.json'
            with open(file_local_path, 'w') as f:
                json.dump(result_dict, f, sort_keys=True, indent=4)
            return file_name, file_local_path
        else:
            raise ValueError('file_format setting not supported: can select only pickle or json')


class LocalResultsSaver(AbstractLocalResultsSaver, BaseLocalResultsSaver):
    def __init__(self, local_model_result_folder_path='~/project/model_result', file_format='pickle'):
        """

        Args:
            local_model_result_folder_path (str):
            file_format (str)
        """
        BaseLocalResultsSaver.__init__(self, local_model_result_folder_path, file_format)

    def save_experiment_results(self, result_package, project_name, experiment_name, experiment_timestamp=None,
                                save_true_pred_labels=False, protect_existing_folder=True):
        """

        Args:
            result_package (AIToolbox.experiment_save.result_package.abstract_result_packages.AbstractResultPackage):
            project_name (str):
            experiment_name (str):
            experiment_timestamp (str or None):
            save_true_pred_labels (bool):
            protect_existing_folder (bool):

        Returns:
            list: list of list with this format: [[results_file_name, results_file_local_path], ... [ , ]] 
                Each file should be a new list specifying the file name and its full path.
                
                The first file path should be pointing to the main experiment results file.
        """
        if experiment_timestamp is None:
            experiment_timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')

        experiment_results_local_path = self.create_experiment_local_folder_structure(project_name, experiment_name,
                                                                                      experiment_timestamp)

        results = result_package.get_results()
        hyperparameters = result_package.get_hyperparameters()
        training_history = result_package.get_training_history()

        additional_results_dump_paths = result_package.get_additional_results_dump_paths()

        exp_results_hyperparam_dict = {'experiment_name': experiment_name,
                                       'experiment_results_local_path': experiment_results_local_path,
                                       'results': results,
                                       'hyperparameters': hyperparameters,
                                       'training_history': training_history}

        if save_true_pred_labels:
            exp_results_hyperparam_dict = {'y_true': result_package.y_true, 'y_predicted': result_package.y_predicted,
                                           **exp_results_hyperparam_dict}

        results_file_name_w_type = f'results_hyperParams_hist_{experiment_name}_{experiment_timestamp}'
        results_file_local_path_w_type = os.path.join(experiment_results_local_path, results_file_name_w_type)

        results_file_name, results_file_local_path = self.save_file(exp_results_hyperparam_dict,
                                                                    results_file_name_w_type,
                                                                    results_file_local_path_w_type)
        
        experiment_results_paths = [[results_file_name, results_file_local_path]] 
        
        if additional_results_dump_paths is not None:
            experiment_results_paths += additional_results_dump_paths
        
        return experiment_results_paths

    def save_experiment_results_separate_files(self, result_package, project_name, experiment_name,
                                               experiment_timestamp=None, save_true_pred_labels=False,
                                               protect_existing_folder=True):
        """

        Args:
            result_package (AIToolbox.experiment_save.result_package.abstract_result_packages.AbstractResultPackage):
            project_name (str):
            experiment_name (str):
            experiment_timestamp (str or None):
            save_true_pred_labels (bool):
            protect_existing_folder (bool):

        Returns:
            list: list of list with this format: [[results_file_name, results_file_local_path], ... [ , ]] 
                Each file should be a new list specifying the file name and its full path.
                
                The first file path should be pointing to the main experiment results file.
        """
        experiment_results_local_path = self.create_experiment_local_folder_structure(project_name, experiment_name,
                                                                                      experiment_timestamp)

        results = result_package.get_results()
        hyperparameters = result_package.get_hyperparameters()
        training_history = result_package.get_training_history()

        additional_results_dump_paths = result_package.get_additional_results_dump_paths()

        experiment_results_dict = {'experiment_name': experiment_name,
                                   'experiment_results_local_path': experiment_results_local_path,
                                   'results': results}

        experiment_hyperparam_dict = {'experiment_name': experiment_name,
                                      'experiment_results_local_path': experiment_results_local_path,
                                      'hyperparameters': hyperparameters}

        experiment_train_hist_dict = {'experiment_name': experiment_name,
                                      'experiment_results_local_path': experiment_results_local_path,
                                      'hyperparameters': training_history}

        results_file_name_w_type = f'results_{experiment_name}_{experiment_timestamp}'
        results_file_local_path_w_type = os.path.join(experiment_results_local_path, results_file_name_w_type)
        results_file_name, results_file_local_path = self.save_file(experiment_results_dict,
                                                                    results_file_name_w_type,
                                                                    results_file_local_path_w_type)

        hyperparams_file_name_w_type = f'hyperparams_{experiment_name}_{experiment_timestamp}'
        hyperparams_file_local_path_w_type = os.path.join(experiment_results_local_path, hyperparams_file_name_w_type)
        hyperparams_file_name, hyperparams_file_local_path = self.save_file(experiment_hyperparam_dict,
                                                                            hyperparams_file_name_w_type,
                                                                            hyperparams_file_local_path_w_type)

        train_hist_file_name_w_type = f'train_history_{experiment_name}_{experiment_timestamp}'
        train_hist_file_local_path_w_type = os.path.join(experiment_results_local_path, train_hist_file_name_w_type)
        train_hist_file_name, train_hist_file_local_path = self.save_file(experiment_train_hist_dict,
                                                                          train_hist_file_name_w_type,
                                                                          train_hist_file_local_path_w_type)

        saved_results_paths = [[results_file_name, results_file_local_path],
                               [hyperparams_file_name, hyperparams_file_local_path],
                               [train_hist_file_name, train_hist_file_local_path]]

        if save_true_pred_labels:
            experiment_true_pred_labels_dict = {'y_true': result_package.y_true, 'y_predicted': result_package.y_predicted}

            labels_file_name_w_type = f'true_pred_labels_{experiment_name}_{experiment_timestamp}'
            labels_file_local_path_w_type = os.path.join(experiment_results_local_path, labels_file_name_w_type)
            labels_file_name, labels_file_local_path = self.save_file(experiment_true_pred_labels_dict,
                                                                      labels_file_name_w_type,
                                                                      labels_file_local_path_w_type)
            saved_results_paths.append([labels_file_name, labels_file_local_path])
            
        if additional_results_dump_paths is not None:
            saved_results_paths += additional_results_dump_paths

        return saved_results_paths
