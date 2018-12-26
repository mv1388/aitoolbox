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
        pass

    @abstractmethod
    def save_experiment_results_separate_files(self, result_package, project_name, experiment_name, experiment_timestamp,
                                               save_true_pred_labels=False, protect_existing_folder=True):
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
        if not os.path.exists(os.path.join(self.local_model_result_folder_path, project_name)):
            os.mkdir(os.path.join(self.local_model_result_folder_path, project_name))

        experiment_path = os.path.join(self.local_model_result_folder_path, project_name,
                                       experiment_name + '_' + experiment_timestamp)

        if not os.path.exists(experiment_path):
            os.mkdir(experiment_path)

        experiment_results_path = os.path.join(experiment_path, 'results')
        os.mkdir(experiment_results_path)
        return experiment_results_path

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
            result_package (AIToolbox.experiment_save.result_package.AbstractResultPackage):
            project_name (str):
            experiment_name (str):
            experiment_timestamp (str or None):
            save_true_pred_labels (bool):
            protect_existing_folder (bool):

        Returns:
            paths to locally saved results... so that S3 version can take them and upload to S3

        """
        if experiment_timestamp is None:
            experiment_timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')

        experiment_results_local_path = self.create_experiment_local_folder_structure(project_name, experiment_name,
                                                                                      experiment_timestamp)

        results = result_package.get_results()
        hyperparameters = result_package.get_hyperparameters()
        training_history = result_package.get_training_history()

        exp_results_hyperparam_dict = {'experiment_name': experiment_name,
                                       'experiment_results_local_path': experiment_results_local_path,
                                       'results': results,
                                       'hyperparameters': hyperparameters,
                                       'training_history': training_history}

        if save_true_pred_labels:
            exp_results_hyperparam_dict = {'y_true': result_package.y_true.tolist(), 'y_predicted': result_package.y_predicted.tolist(),
                                           **exp_results_hyperparam_dict}

        results_file_name_w_type = f'results_hyperParams_hist_{experiment_name}_{experiment_timestamp}'
        results_file_local_path_w_type = os.path.join(experiment_results_local_path, results_file_name_w_type)

        results_file_name, results_file_local_path = self.save_file(exp_results_hyperparam_dict,
                                                                    results_file_name_w_type,
                                                                    results_file_local_path_w_type)

        return results_file_name, results_file_local_path

    def save_experiment_results_separate_files(self, result_package, project_name, experiment_name,
                                               experiment_timestamp=None, save_true_pred_labels=False,
                                               protect_existing_folder=True):
        """

        Args:
            result_package (AIToolbox.ExperimentSave.result_package.AbstractResultPackage):
            project_name (str):
            experiment_name (str):
            experiment_timestamp (str or None):
            save_true_pred_labels (bool):
            protect_existing_folder (bool):

        Returns:
            paths to locally saved results... so that S3 version can take them and upload to S3

        """
        experiment_results_local_path = self.create_experiment_local_folder_structure(project_name, experiment_name,
                                                                                      experiment_timestamp)

        results = result_package.get_results()
        hyperparameters = result_package.get_hyperparameters()
        training_history = result_package.get_training_history()

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

        train_hist_file_name_w_type = 'train_history_{}_{}'.format(experiment_name, experiment_timestamp)
        train_hist_file_local_path_w_type = os.path.join(experiment_results_local_path, train_hist_file_name_w_type)
        train_hist_file_name, train_hist_file_local_path = self.save_file(experiment_results_dict,
                                                                          train_hist_file_name_w_type,
                                                                          train_hist_file_local_path_w_type)

        saved_results_paths = [[results_file_name, results_file_local_path],
                               [hyperparams_file_name, hyperparams_file_local_path],
                               [train_hist_file_name, train_hist_file_local_path]]

        if save_true_pred_labels:
            experiment_true_pred_labels_dict = {'y_true': result_package.y_true.tolist(), 'y_predicted': result_package.y_predicted.tolist()}

            labels_file_name_w_type = 'true_pred_labels_{}_{}'.format(experiment_name, experiment_timestamp)
            labels_file_local_path_w_type = os.path.join(experiment_results_local_path, labels_file_name_w_type)
            labels_file_name, labels_file_local_path = self.save_file(experiment_true_pred_labels_dict,
                                                                      labels_file_name_w_type,
                                                                      labels_file_local_path_w_type)
            saved_results_paths.append([labels_file_name, labels_file_local_path])

        return saved_results_paths
