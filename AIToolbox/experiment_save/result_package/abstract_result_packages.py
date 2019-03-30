import copy
from abc import ABC, abstractmethod
import shutil
import numpy as np

from AIToolbox.experiment_save.training_history import TrainingHistory


class AbstractResultPackage(ABC):
    def __init__(self, pkg_name=None, strict_content_check=False, **kwargs):
        """

        Functions which the user should potentially override in a specific result package:
            - prepare_results_dict()
            - list_additional_results_dump_paths()
            - set_experiment_dir_path_for_additional_results()

        Args:
            pkg_name (str or None):
            strict_content_check (bool):
            **kwargs (dict):
        """
        self.pkg_name = pkg_name
        self.strict_content_check = strict_content_check

        self.y_true = None
        self.y_predicted = None
        self.additional_results = None
        self.additional_results_dump_paths = None
        self.results_dict = None

        self.hyperparameters = None
        self.training_history = None
        self.package_metadata = kwargs

    @abstractmethod
    def prepare_results_dict(self):
        """ Perform result package building and save the result into self.results_dict

        Mostly this consists of executing calculation of selected performance metrics and returning their result dicts.
        If you want to use multiple performance metrics you have to combine them in the single self.results_dict
        at the end by doing this:
            self.results_dict = {**metric_dict_1, **metric_dict_2}

        Returns:
            None
        """
        pass

    def prepare_result_package(self, y_true, y_predicted, hyperparameters=None, training_history=None, **kwargs):
        """

        Args:
            y_true (numpy.array or list):
            y_predicted (numpy.array or list):
            hyperparameters (dict or None):
            training_history (AIToolbox.ExperimentSave.training_history.TrainingHistory):
            **kwargs (dict):

        Returns:
            None
        """
        self.y_true = np.array(y_true)
        self.y_predicted = np.array(y_predicted)

        self.results_dict = None
        self.hyperparameters = hyperparameters
        self.training_history = training_history
        self.additional_results = kwargs

        self.prepare_results_dict()

    def get_results(self):
        """

        Returns:
            dict:
        """
        if self.results_dict is None:
            self.warn_about_result_data_problem('Warning: Results dict missing')

        return self.results_dict

    def get_hyperparameters(self):
        """

        Returns:
            dict:
        """
        self.qa_check_hyperparameters_dict()
        return self.hyperparameters

    def get_training_history(self):
        """Extract training history dict from the training history object

        Returns:
            dict:
        """
        # History QA check is (automatically) done in the history object and not here in the result package
        return self.training_history.get_train_history()

    def get_additional_results_dump_paths(self):
        """

        Returns:
            list or None: list of lists of string paths if it is not None.
                Each element of the list should be list of: [[results_file_name, results_file_local_path], ... [,]]
        """
        self.additional_results_dump_paths = self.list_additional_results_dump_paths()
        self.qa_check_additional_results_dump_paths()
        return self.additional_results_dump_paths

    def list_additional_results_dump_paths(self):
        """Specify the list of meta data files you also want to save & upload to s3 during the experiment saving procedure

        By default there are no additional files that are saved as the return is None. If you want to save your
        specific additional files produced during the training procedure, then override this method specifying
        the file paths.

        If you want to save a whole folder of files, use zip_additional_results_dump() function to zip it into a single
        file and save this zip instead.

        The specified files are any additional data you would want to include into the experiment folder in addition to
        the model save files and performance evaluation report files. For example a zip of attention heatmap pictures
        in the machine translation projects.

        Returns:
            list or None: list of lists of string paths if it is not None.
                Each element of the list should be list of: [[results_file_name, results_file_local_path], ... [,]]
        """
        return None

    def set_experiment_dir_path_for_additional_results(self, project_name, experiment_name, experiment_timestamp,
                                                       local_model_result_folder_path):
        """Set experiment folder path after potential timestamps have already been generated.

        Experiment folder setting for additional metadata results output is needed only in certain result packages,
        for example in QuestionAnswerResultPackage where the self.output_text_dir initially has only the name of
        the folder where the results text predictions for each example should be stored. This function when implemented
        reforms the folder name so that it becomes a full path placing the folder inside the experiment folder (for
        which the timestamp at the start of train loop is needed).

        Another use of this function is in MachineTranslationResultPackage where the attention heatmap pictures are
        stored as additional metadata results.

        As can be seen from the fact that the train loop mechanism is mentioned, this method's functionality is
        primarily used for PyTorch experiments.

        Args:
            project_name (str):
            experiment_name (str):
            experiment_timestamp (str):
            local_model_result_folder_path (str):

        Returns:

        """
        pass

    def qa_check_hyperparameters_dict(self):
        """

        Returns:
            None:
        """
        if self.hyperparameters is None:
            self.warn_about_result_data_problem('Warning: Hyperparameters missing')

        # Check if hyperparameters dict had bash script element and is not None
        if 'bash_script_path' not in self.hyperparameters or self.hyperparameters['bash_script_path'] is None:
            self.warn_about_result_data_problem('bash_script_path missing from hyperparameters dict. '
                                                'Potential solution: add it to the model config file and parse '
                                                'with argparser.')

        # Check if hyperparameters dict had python experiment script element and is not None
        if 'python_experiment_file_path' not in self.hyperparameters or \
                self.hyperparameters['python_experiment_file_path'] is None:
            self.warn_about_result_data_problem('python_experiment_file_path missing from hyperparameters dict. '
                                                'Potential solution: add it to the model config file and parse '
                                                'with argparser.')

    def qa_check_additional_results_dump_paths(self):
        """

        Returns:

        """
        if self.additional_results_dump_paths is not None:
            if type(self.additional_results_dump_paths) is not list:
                raise ValueError(f'Additional results dump paths are given but are not in the list format. '
                                 f'Dump paths: {self.additional_results_dump_paths}')

            for el in self.additional_results_dump_paths:
                if type(el) is not list:
                    raise ValueError(f'Element inside additional results dump is not a list. Element is: {el}')
                if len(el) != 2:
                    raise ValueError(f'Element must be a list of len 2. Element is : {el}')
                if type(el[0]) is not str or type(el[1]) is not str:
                    raise ValueError(f'One of the path elements is not string. Element is : {el}')

    def warn_about_result_data_problem(self, msg):
        """

        Args:
            msg (str):

        Returns:
            None:
        """
        if self.strict_content_check:
            raise ValueError(msg)
        else:
            print(msg)

    @staticmethod
    def zip_additional_results_dump(source_dir_path, zip_path):
        """

        Args:
            source_dir_path (str):
            zip_path (str): specify the path with the zip name but without the '.zip' at the end

        Returns:
            str: the full path to the produced zip file (with the .zip extension appended)
        """
        shutil.make_archive(zip_path, 'zip', source_dir_path)
        return zip_path + '.zip'

    def __str__(self):
        self.warn_if_results_dict_not_defined()
        return '\n'.join([f'{result_metric}: {self.results_dict[result_metric]}'
                          for result_metric in self.results_dict])

    def __len__(self):
        self.warn_if_results_dict_not_defined()
        return len(self.results_dict)

    def __add__(self, other):
        """

        Args:
            other (AIToolbox.experiment_save.abstract_result_package.AbstractResultPackage or dict):

        Returns:
            AIToolbox.experiment_save.result_package.MultipleResultPackageWrapper:
        """
        return self.add_merge_multi_pkg_wrap(other)

    def __radd__(self, other):
        """

        Args:
            other (AIToolbox.experiment_save.abstract_result_package.AbstractResultPackage or dict):

        Returns:
            AIToolbox.experiment_save.result_package.MultipleResultPackageWrapper:
        """
        return self.add_merge_multi_pkg_wrap(other)

    def add_merge_multi_pkg_wrap(self, other_object):
        """

        Args:
            other_object (AIToolbox.experiment_save.abstract_result_package.AbstractResultPackage or dict):

        Returns:
            AIToolbox.experiment_save.result_package.MultipleResultPackageWrapper:
        """
        self.warn_if_results_dict_not_defined()
        other_object_pkg = self.create_other_object_pkg(other_object)

        self_object_copy = copy.deepcopy(self)

        multi_result_pkg = MultipleResultPackageWrapper()
        multi_result_pkg.prepare_result_package([self_object_copy, other_object_pkg],
                                                self_object_copy.hyperparameters, self_object_copy.training_history)

        return multi_result_pkg

    @staticmethod
    def create_other_object_pkg(other_object):
        """

        Args:
            other_object (AIToolbox.experiment_save.abstract_result_package.AbstractResultPackage or dict):

        Returns:
            AIToolbox.experiment_save.result_package.AbstractResultPackage:
        """
        if isinstance(other_object, AbstractResultPackage):
            other_object.warn_if_results_dict_not_defined()
            other_object_pkg = copy.deepcopy(other_object)
        elif type(other_object) is dict:
            other_object_copy = copy.deepcopy(other_object)
            other_object_pkg = PreCalculatedResultPackage(other_object_copy)
        else:
            raise ValueError(f'Addition supported on the AbstractResultPackage objects and dicts. Given {type(other_object)}')

        return other_object_pkg

    def __iadd__(self, other):
        """

        Args:
            other (AIToolbox.experiment_save.abstract_result_package.AbstractResultPackage or dict):

        Returns:
            AIToolbox.experiment_save.result_package.AbstractResultPackage:
        """
        return self.add_merge_dicts(other)

    def add_merge_dicts(self, other):
        """

        Args:
            other (AIToolbox.experiment_save.abstract_result_package.AbstractResultPackage or dict):

        Returns:
            AIToolbox.experiment_save.result_package.AbstractResultPackage:
        """
        self.warn_if_results_dict_not_defined()

        if isinstance(other, AbstractResultPackage):
            other.warn_if_results_dict_not_defined()
            return self.merge_dicts(other.results_dict)
        elif type(other) is dict:
            return self.merge_dicts(other)
        else:
            raise ValueError(f'Addition supported on the AbstractResultPackage objects and dicts. Given {type(other)}')

    def merge_dicts(self, other_results_dict):
        """

        Args:
            other_results_dict (dict):

        Returns:
            AIToolbox.experiment_save.abstract_result_package.AbstractResultPackage:
        """
        def results_duplicated(self_results_dict, other_results_dict_dup):
            for result_name in other_results_dict_dup:
                if result_name in self_results_dict:
                    return True
            return False

        if not results_duplicated(self.results_dict, other_results_dict):
            self.results_dict = {**self.results_dict, **other_results_dict}
            return self
        else:
            raise ValueError(f'Duplicated metric results found in the results_dict. '
                             f'Trying to merge the following results_dict metrics: '
                             f'{self.results_dict.keys()} and {other_results_dict.keys()}')

    def __contains__(self, item):
        self.warn_if_results_dict_not_defined()
        if item in self.results_dict:
            return True
        else:
            return False

    def __getitem__(self, item):
        self.warn_if_results_dict_not_defined()
        if item in self.results_dict:
            return self.results_dict[item]
        else:
            raise KeyError(f'Key {item} can not be found in the results_dict. '
                           f'Currently present keys: {self.results_dict.keys()}')

    def warn_if_results_dict_not_defined(self):
        if self.results_dict is None:
            raise ValueError(f'results_dict is not set yet. Currently it is {self.results_dict}')


class PreCalculatedResultPackage(AbstractResultPackage):
    def __init__(self, results_dict, strict_content_check=False, **kwargs):
        """

        Args:
            results_dict (dict):
            strict_content_check (bool):
            **kwargs (dict):
        """
        AbstractResultPackage.__init__(self, pkg_name='PreCalculatedResult',
                                       strict_content_check=strict_content_check, **kwargs)
        self.results_dict = results_dict
        self.training_history = TrainingHistory({}, [], strict_content_check)
        self.hyperparameters = {}

    def prepare_results_dict(self):
        pass


class MultipleResultPackageWrapper(AbstractResultPackage):
    def __init__(self, strict_content_check=False, **kwargs):
        """

        Args:
            strict_content_check (bool):
            **kwargs (dict):
        """
        AbstractResultPackage.__init__(self, pkg_name='MultipleResultWrapper',
                                       strict_content_check=strict_content_check, **kwargs)
        self.result_packages = None

    def prepare_result_package(self, result_packages, hyperparameters=None, training_history=None, **kwargs):
        """

        Args:
            result_packages (list):
            hyperparameters:
            training_history:
            **kwargs:

        Returns:

        """
        self.results_dict = None
        self.result_packages = result_packages
        self.hyperparameters = hyperparameters
        self.training_history = training_history
        self.additional_results = kwargs

        self.prepare_results_dict()

    def prepare_results_dict(self):
        self.results_dict = {}
        self.y_true = {}
        self.y_predicted = {}
        self.additional_results = {self.pkg_name: self.additional_results} if self.additional_results != {} else {}

        for i, result_pkg in enumerate(self.result_packages):
            if result_pkg.pkg_name is not None:
                suffix = '' if result_pkg.pkg_name not in self.results_dict else str(i)
                package_name = result_pkg.pkg_name + suffix
            else:
                package_name = f'ResultPackage{i}'

            self.results_dict[package_name] = result_pkg.get_results()
            self.y_true[package_name] = result_pkg.y_true
            self.y_predicted[package_name] = result_pkg.y_predicted
            self.additional_results[package_name] = result_pkg.additional_results

    def get_additional_results_dump_paths(self):
        """

        Returns:
            list or None: list of lists of string paths if it is not None.
                Each element of the list should be list of: [[results_file_name, results_file_local_path], ... [,]]
        """
        self.additional_results_dump_paths = self.list_additional_results_dump_paths()

        sub_packages_paths = []
        for pkg in self.result_packages:
            pkg_additional_paths = pkg.get_additional_results_dump_paths()
            if pkg_additional_paths is not None:
                sub_packages_paths += pkg_additional_paths

        if self.additional_results_dump_paths is None and len(sub_packages_paths) > 0:
            self.additional_results_dump_paths = sub_packages_paths
        elif self.additional_results_dump_paths is not None and len(sub_packages_paths) > 0:
            self.additional_results_dump_paths += sub_packages_paths

        self.qa_check_additional_results_dump_paths()
        return self.additional_results_dump_paths

    def __str__(self):
        return '\n'.join([f'--> {pkg.pkg_name}:\n{str(pkg)}' for pkg in self.result_packages])

    def __len__(self):
        return len(self.result_packages)

    def add_merge_multi_pkg_wrap(self, other_object):
        """

        Args:
            other_object (AIToolbox.experiment_save.result_package.abstract_result_packages.AbstractResultPackage or dict):

        Returns:
            AIToolbox.experiment_save.result_package.abstract_result_packages.MultipleResultPackageWrapper:
        """
        self.warn_if_results_dict_not_defined()
        other_object_pkg = self.create_other_object_pkg(other_object)
        self_multi_result_pkg = copy.deepcopy(self)

        other_object_pkg_list = [other_object_pkg] if type(other_object_pkg) is not MultipleResultPackageWrapper \
            else other_object_pkg.result_packages

        self_multi_result_pkg.prepare_result_package(self_multi_result_pkg.result_packages + other_object_pkg_list,
                                                     self_multi_result_pkg.hyperparameters,
                                                     self_multi_result_pkg.training_history,
                                                     **self_multi_result_pkg.package_metadata)
        return self_multi_result_pkg

    def __iadd__(self, other):
        """

        Args:
            other (AIToolbox.experiment_save.abstract_result_package.AbstractResultPackage or dict):

        Returns:
            AIToolbox.experiment_save.result_package.abstract_result_packages.MultipleResultPackageWrapper:
        """
        self.warn_if_results_dict_not_defined()
        other_object_pkg = self.create_other_object_pkg(other)

        other_object_pkg_list = [other_object_pkg] if type(other_object_pkg) is not MultipleResultPackageWrapper \
            else other_object_pkg.result_packages

        self.prepare_result_package(self.result_packages + other_object_pkg_list,
                                    self.hyperparameters, self.training_history, **self.package_metadata)
        return self
