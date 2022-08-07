import copy
from abc import ABC, abstractmethod
import numpy as np

from aitoolbox.utils import file_system
from aitoolbox.experiment.training_history import TrainingHistory


class AbstractResultPackage(ABC):
    def __init__(self, pkg_name=None, strict_content_check=False, np_array=True, **kwargs):
        """Base Result package used to derive specific result packages from

        Functions which the user should potentially override in a specific result package:

            * :meth:`aitoolbox.experiment.result_package.abstract_result_packages.AbstractResultPackage.prepare_results_dict`
            * :meth:`aitoolbox.experiment.result_package.abstract_result_packages.AbstractResultPackage.list_additional_results_dump_paths`
            * :meth:`aitoolbox.experiment.result_package.abstract_result_packages.AbstractResultPackage.set_experiment_dir_path_for_additional_results`

        Args:
            pkg_name (str or None): result package name used just for clarity
            strict_content_check (bool): should just print warning or raise the error and crash
            np_array (bool or str): how the inputs should be handled. Should the package try to automatically guess, or
                you want to manually decide whether to leave the inputs as they are or convert them to np.array.
                Possible options: True, False, 'auto'

                Be slightly careful with 'auto' as it sometimes doesn't work,
                so it is preferable to explicitly use True/False
            **kwargs (dict): additional package_metadata for the result package
        """
        self.pkg_name = pkg_name
        self.strict_content_check = strict_content_check
        self.np_array = np_array

        self.y_true = None
        self.y_predicted = None
        self.additional_results = None
        self.additional_results_dump_paths = None
        self.results_dict = None

        self.requires_loss = False

        self.hyperparameters = None
        self.package_metadata = kwargs

    @abstractmethod
    def prepare_results_dict(self):
        """Perform result package building

        Mostly this consists of executing calculation of selected performance metrics and returning their result dicts.
        If you want to use multiple performance metrics you have to combine them in the single self.results_dict
        at the end by doing this:
            return {**metric_dict_1, **metric_dict_2}

        Returns:
            dict: calculated result dict
        """
        pass

    def prepare_result_package(self, y_true, y_predicted, hyperparameters=None, **kwargs):
        """Prepares the result package by taking labels and running them through the specified metrics

        This function is automatically called from the torchtrain callbacks to evaluate the provided callback.
        The main feature of this function is the call to the user-derived prepare_results_dict() function of
        the implemented result package where the metrics evaluation logic is implemented.

        Args:
            y_true (numpy.array or list): ground truth targets
            y_predicted (numpy.array or list): predicted targets
            hyperparameters (dict or None): dictionary filled with the set hyperparameters
            **kwargs (dict): additional results for the result package

        Returns:
            None
        """
        if self.np_array is True:
            self.y_true = np.array(y_true)
            self.y_predicted = np.array(y_predicted)
        elif self.np_array == 'auto':
            # This option sometimes doesnt't work and the explicit True/False specification is needed by the user
            self.y_true = self.auto_y_input_array_convert(y_true)
            self.y_predicted = self.auto_y_input_array_convert(y_predicted)
        else:
            self.y_true = y_true
            self.y_predicted = y_predicted

        self.results_dict = None
        self.hyperparameters = hyperparameters
        self.additional_results = kwargs

        self.results_dict = self.prepare_results_dict()

    @staticmethod
    def auto_y_input_array_convert(y_array):
        """Try to automatically decide if array should be left as it is or convert to np.array

        Not working in all the situations so relying on it at all times is not recommended.
        Especially for costly experiments rely rather on your own judgement and explicitly define
        if np.array conversion is needed.

        TODO: make it smarter so 'auto' option can be used more often

        Args:
            y_array (list):

        Returns:
            list or numpy.array:
        """
        previous_len = len(y_array[0])
        np_array_ok = True

        for el in y_array:
            if len(el) != previous_len:
                np_array_ok = False
                break

        if np_array_ok:
            return np.array(y_array)
        else:
            return y_array

    def get_results(self):
        """Get calculated results dict

        Returns:
            dict: results dict
        """
        if self.results_dict is None:
            self.warn_about_result_data_problem('Warning: Results dict missing')

        return self.results_dict

    def get_hyperparameters(self):
        """Get hyperparameters in a dict form

        Returns:
            dict: hyperparameters dict
        """
        self.qa_check_hyperparameters_dict()
        return self.hyperparameters

    def get_additional_results_dump_paths(self):
        """Return paths to the additional results which are stored to local drive when the package is evaluated

        For example if package plots attention heatmaps and saves pictures to disk, this function will return
        paths to these picture files. This is achieved via the call to the user-implemented function
        list_additional_results_dump_paths().

        Returns:
            list or None: list of lists of string paths if it is not None.
                Each element of the list should be list of: [[results_file_name, results_file_local_path], ... [,]]
        """
        self.additional_results_dump_paths = self.list_additional_results_dump_paths()
        self.qa_check_additional_results_dump_paths()
        return self.additional_results_dump_paths

    def list_additional_results_dump_paths(self):
        """Specify the list of metadata files you also want to save & upload to s3 during the experiment saving procedure

        By default, there are no additional files that are saved as the return is None. If you want to save your
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
            project_name (str): root name of the project
            experiment_name (str): name of the particular experiment
            experiment_timestamp (str): time stamp at the start of training
            local_model_result_folder_path (str): root local path where project folder will be created

        Returns:
            None
        """
        pass

    def qa_check_hyperparameters_dict(self):
        """Quality check the hyperparams dict

        Returns:
            None
        """
        if self.hyperparameters is None:
            self.warn_about_result_data_problem('Warning: Hyperparameters missing')

        # Check if hyperparameters dict had bash script element and is not None
        if 'bash_script_path' not in self.hyperparameters or self.hyperparameters['bash_script_path'] is None:
            self.warn_about_result_data_problem('bash_script_path missing from hyperparameters dict. '
                                                'Potential solution: add it to the model config file and parse '
                                                'with argparser.')

        # Check if hyperparameters dict had python experiment script element and is not None
        if 'experiment_file_path' not in self.hyperparameters or \
                self.hyperparameters['experiment_file_path'] is None:
            self.warn_about_result_data_problem('experiment_file_path missing from hyperparameters dict. '
                                                'Potential solution: add it to the model config file and parse '
                                                'with argparser.')

    def qa_check_additional_results_dump_paths(self):
        """Quality check the additional results path

        Returns:
            None
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
        """Generic function for writing out warnings

        Either just printing out the warning or throw the error exception.

        Args:
            msg (str): warning message either printed or written in the raised error

        Raises:
            ValueError

        Returns:
            None
        """
        if self.strict_content_check:
            raise ValueError(msg)
        else:
            print(msg)

    @staticmethod
    def zip_additional_results_dump(source_dir_path, zip_path):
        """Utility function for zipping a folder into .zip archive

        Args:
            source_dir_path (str): path to the folder that is going to be zipped
            zip_path (str): specify the path of the zip file which will be created

        Returns:
            str: the full path to the produced zip file (with the .zip extension appended)
        """
        return file_system.zip_folder(source_dir_path, zip_path)

    def __str__(self):
        self.warn_if_results_dict_not_defined()
        return '\n'.join([f'{result_metric}: {self.results_dict[result_metric]}'
                          for result_metric in self.results_dict])

    def __len__(self):
        self.warn_if_results_dict_not_defined()
        return len(self.results_dict)

    def __add__(self, other):
        """Concatenate result packages

        Combines results from both result packages into a single one.

        Args:
            other (aitoolbox.experiment.result_package.abstract_result_packages.AbstractResultPackage or dict): another
                result package to be concatenated

        Returns:
            aitoolbox.experiment.result_package.MultipleResultPackageWrapper: merged result package
        """
        return self.add_merge_multi_pkg_wrap(other)

    def __radd__(self, other):
        """Concatenate result package

        Args:
            other (aitoolbox.experiment.result_package.abstract_result_packages.AbstractResultPackage or dict): another
                result package to be concatenated

        Returns:
            aitoolbox.experiment.result_package.MultipleResultPackageWrapper: merged result package
        """
        return self.add_merge_multi_pkg_wrap(other)

    def add_merge_multi_pkg_wrap(self, other_object):
        """Result package merge

        Args:
            other_object (aitoolbox.experiment.result_package.abstract_result_packages.AbstractResultPackage or dict):
                another result package to be merged with the current package

        Returns:
            aitoolbox.experiment.result_package.abstract_result_packages.MultipleResultPackageWrapper: merged result
                package
        """
        self.warn_if_results_dict_not_defined()
        other_object_pkg = self._create_other_object_pkg(other_object)

        self_object_copy = copy.deepcopy(self)

        multi_result_pkg = MultipleResultPackageWrapper()
        multi_result_pkg.prepare_result_package([self_object_copy, other_object_pkg], self_object_copy.hyperparameters)

        return multi_result_pkg

    @staticmethod
    def _create_other_object_pkg(other_object):
        """Util to deep copy and wrap results into the simple result package

        Args:
            other_object (aitoolbox.experiment.result_package.abstract_result_packages.AbstractResultPackage or dict):
                results package or results dict

        Returns:
            AbstractResultPackage | MultipleResultPackageWrapper: deep copy of results wrapped in the simple result
                package
        """
        if isinstance(other_object, AbstractResultPackage):
            other_object.warn_if_results_dict_not_defined()
            other_object_pkg = copy.deepcopy(other_object)
        elif type(other_object) is dict:
            other_object_copy = copy.deepcopy(other_object)
            other_object_pkg = PreCalculatedResultPackage(other_object_copy)
        else:
            raise ValueError(f'Addition supported on the AbstractResultPackage objects and dicts. '
                             f'Given {type(other_object)}')

        return other_object_pkg

    def __iadd__(self, other):
        """Append result package

        Args:
            other (aitoolbox.experiment.result_package.abstract_result_packages.AbstractResultPackage or dict): another
                result package to be appended to the current package

        Returns:
            aitoolbox.experiment.result_package.abstract_result_packages.AbstractResultPackage: merged result package
        """
        return self.add_merge_dicts(other)

    def add_merge_dicts(self, other):
        """Append result package to the current one

        Args:
            other (aitoolbox.experiment.result_package.abstract_result_packages.AbstractResultPackage or dict): another
                result package to be appended to the current package

        Returns:
            aitoolbox.experiment.result_package.abstract_result_packages.AbstractResultPackage: merged result package
        """
        self.warn_if_results_dict_not_defined()

        if isinstance(other, AbstractResultPackage):
            other.warn_if_results_dict_not_defined()
            return self._merge_dicts(other.results_dict)
        elif type(other) is dict:
            return self._merge_dicts(other)
        else:
            raise ValueError(f'Addition supported on the AbstractResultPackage objects and dicts. Given {type(other)}')

    def _merge_dicts(self, other_results_dict):
        """Results dict merge util

        Args:
            other_results_dict (dict): another results dict to be added to the results dict in the current result
                package

        Returns:
            aitoolbox.experiment.result_package.abstract_result_packages.AbstractResultPackage: merged result package
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
        """Result package which doesn't have any evaluation logic but just accepts pre-calculated results dict

        Args:
            results_dict (dict): pre-calculated results dict
            strict_content_check (bool): should just print warning or raise the error and crash
            **kwargs (dict): result package additional meta-data
        """
        AbstractResultPackage.__init__(self, pkg_name='PreCalculatedResult',
                                       strict_content_check=strict_content_check, **kwargs)
        self.results_dict = results_dict
        self.training_history = TrainingHistory()
        self.hyperparameters = {}

    def prepare_results_dict(self):
        pass


class MultipleResultPackageWrapper(AbstractResultPackage):
    def __init__(self, strict_content_check=False, **kwargs):
        """Wrapper result package which combines multiple evaluated result packages into a single result package

        Args:
            strict_content_check (bool): should just print warning or raise the error and crash
            **kwargs (dict): result package additional meta-data
        """
        AbstractResultPackage.__init__(self, pkg_name='MultipleResultWrapper',
                                       strict_content_check=strict_content_check, **kwargs)
        self.result_packages = None

    def prepare_result_package(self, result_packages, hyperparameters=None, **kwargs):
        """Prepares the multiple result package by merging the results from both result packages

        Args:
            result_packages (list): list of result packages where each of them is object inherited from
                aitoolbox.experiment.result_package.abstract_result_packages.AbstractResultPackage.
                If you want to add raw results in dict form, this dict first needs to be wrapped into
                aitoolbox.experiment.result_package.abstract_result_packages.PreCalculatedResultPackage to satisfy
                the result package object requirement.
            hyperparameters (dict or None): hyperparameters dict
            **kwargs: result package additional meta-data

        Returns:
            None
        """
        self.results_dict = None
        self.result_packages = result_packages
        self.hyperparameters = hyperparameters
        self.additional_results = kwargs

        self.results_dict = self.prepare_results_dict()

    def prepare_results_dict(self):
        results_dict = {}
        self.y_true = {}
        self.y_predicted = {}
        self.additional_results = {self.pkg_name: self.additional_results} if self.additional_results != {} else {}

        for i, result_pkg in enumerate(self.result_packages):
            if result_pkg.pkg_name is not None:
                suffix = '' if result_pkg.pkg_name not in results_dict else str(i)
                package_name = result_pkg.pkg_name + suffix
            else:
                package_name = f'ResultPackage{i}'

            results_dict[package_name] = result_pkg.get_results()
            self.y_true[package_name] = result_pkg.y_true
            self.y_predicted[package_name] = result_pkg.y_predicted
            self.additional_results[package_name] = result_pkg.additional_results

        return results_dict

    def get_additional_results_dump_paths(self):
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
        """Get number of result packages inside the multi result package wrapper

        Returns:
            int: number of result packages inside this multi package wrapper
        """
        return len(self.result_packages)

    def add_merge_multi_pkg_wrap(self, other_object):
        self.warn_if_results_dict_not_defined()
        other_object_pkg = self._create_other_object_pkg(other_object)
        self_multi_result_pkg = copy.deepcopy(self)

        other_object_pkg_list = [other_object_pkg] if type(other_object_pkg) is not MultipleResultPackageWrapper \
            else other_object_pkg.result_packages

        self_multi_result_pkg.prepare_result_package(self_multi_result_pkg.result_packages + other_object_pkg_list,
                                                     self_multi_result_pkg.hyperparameters,
                                                     **self_multi_result_pkg.package_metadata)
        return self_multi_result_pkg

    def __iadd__(self, other):
        self.warn_if_results_dict_not_defined()
        other_object_pkg = self._create_other_object_pkg(other)

        other_object_pkg_list = [other_object_pkg] if type(other_object_pkg) is not MultipleResultPackageWrapper \
            else other_object_pkg.result_packages

        self.prepare_result_package(self.result_packages + other_object_pkg_list,
                                    self.hyperparameters, **self.package_metadata)
        return self
