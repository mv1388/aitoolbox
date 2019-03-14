from abc import ABC, abstractmethod
import copy
import numpy as np

from AIToolbox.experiment_save.core_metrics.base_metric import AbstractBaseMetric
from AIToolbox.experiment_save.core_metrics.classification import AccuracyMetric, ROCAUCMetric, \
    PrecisionRecallCurveAUCMetric, F1ScoreMetric
from AIToolbox.experiment_save.core_metrics.regression import MeanSquaredErrorMetric, MeanAbsoluteErrorMetric


class AbstractResultPackage(ABC):
    def __init__(self, strict_content_check=False, **kwargs):
        """

        Args:
            strict_content_check (bool):
            **kwargs (dict):
        """
        self.pkg_name = None
        self.strict_content_check = strict_content_check

        self.y_true = None
        self.y_predicted = None
        self.additional_results = None
        self.results_dict = None
        
        self.hyperparameters = None
        self.training_history = None
        self.package_metadata = kwargs

    @abstractmethod
    def prepare_results_dict(self):
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
        self.training_history = training_history.get_train_history()
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
        """

        Returns:
            dict:
        """
        # History QA check is (automatically) done in the history object and not here in the result package
        return self.training_history

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
            other (AIToolbox.experiment_save.result_package.AbstractResultPackage or dict):

        Returns:
            AIToolbox.experiment_save.result_package.MultipleResultPackageWrapper:
        """
        return self.add_merge_multi_pkg_wrap(other)

    # def __radd__(self, other):
    #     """
    #
    #     Args:
    #         other (AIToolbox.experiment_save.result_package.AbstractResultPackage or dict):
    #
    #     Returns:
    #         AIToolbox.experiment_save.result_package.MultipleResultPackageWrapper:
    #     """
    #     return self.add_merge_multi_pkg_wrap(other)

    def add_merge_multi_pkg_wrap(self, other_object):
        """

        Args:
            other_object (AIToolbox.experiment_save.result_package.AbstractResultPackage or dict):

        Returns:
            AIToolbox.experiment_save.result_package.MultipleResultPackageWrapper:
        """
        self.warn_if_results_dict_not_defined()
        other_object_pkg = self.create_other_object_pkg(other_object)

        self_object_copy = copy.deepcopy(self)

        multi_result_pkg = MultipleResultPackageWrapper([self_object_copy, other_object_pkg])
        multi_result_pkg.prepare_result_package(self_object_copy.y_true, self_object_copy.y_predicted,
                                                self_object_copy.hyperparameters, self_object_copy.training_history)

        return multi_result_pkg

    @staticmethod
    def create_other_object_pkg(other_object):
        """

        Args:
            other_object (AIToolbox.experiment_save.result_package.AbstractResultPackage or dict):

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
            other (AIToolbox.experiment_save.result_package.AbstractResultPackage or dict):

        Returns:
            AIToolbox.experiment_save.result_package.AbstractResultPackage:
        """
        return self.add_merge_dicts(other)

    def add_merge_dicts(self, other):
        """

        Args:
            other (AIToolbox.experiment_save.result_package.AbstractResultPackage or dict):

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
            AIToolbox.experiment_save.result_package.AbstractResultPackage:
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
        self.results_dict = results_dict
        AbstractResultPackage.__init__(self, strict_content_check, **kwargs)

    def prepare_results_dict(self):
        pass


class GeneralResultPackage(AbstractResultPackage):
    def __init__(self, metrics_list, strict_content_check=False, **kwargs):
        """

        Args:
            metrics_list (list): List of objects which are inherited from
                AIToolbox.experiment_save.core_metrics.BaseMetric.AbstractBaseMetric
            strict_content_check (bool):
            **kwargs (dict):
        """
        self.metrics_list = metrics_list
        AbstractResultPackage.__init__(self, strict_content_check, **kwargs)

    def prepare_results_dict(self):
        """

        Returns:
            None:
        """
        self.qa_check_hyperparameters_dict()
        self.results_dict = {}

        for metric in self.metrics_list:
            metric_result_dict = metric(self.y_true, self.y_predicted).get_metric_dict()
            self.results_dict = {**self.results_dict, **metric_result_dict}

    def qa_check_metrics_list(self):
        """

        Returns:
            None
        """
        if len(self.metrics_list) == 0:
            self.warn_about_result_data_problem('Metrics list is empty')

        for metric in self.metrics_list:
            if not isinstance(metric, AbstractBaseMetric):
                self.warn_about_result_data_problem('Metric is not inherited from AbstractBaseMetric class')


class BinaryClassificationResultPackage(AbstractResultPackage):
    def __init__(self, strict_content_check=False, **kwargs):
        """

        Args:
            strict_content_check (bool):
            **kwargs (dict):
        """
        AbstractResultPackage.__init__(self, strict_content_check, **kwargs)

    def prepare_results_dict(self):
        """

        Returns:
            None:
        """
        accuracy_result = AccuracyMetric(self.y_true, self.y_predicted).get_metric_dict()
        roc_auc_result = ROCAUCMetric(self.y_true, self.y_predicted).get_metric_dict()
        pr_auc_result = PrecisionRecallCurveAUCMetric(self.y_true, self.y_predicted).get_metric_dict()
        f1_score_result = F1ScoreMetric(self.y_true, self.y_predicted).get_metric_dict()

        self.results_dict = {**accuracy_result, **roc_auc_result, **pr_auc_result, **f1_score_result}


class ClassificationResultPackage(AbstractResultPackage):
    def __init__(self, strict_content_check=False, **kwargs):
        """

        Without Precision-Recall metric which is available only for binary classification problems.

        Args:
            strict_content_check (bool):
            **kwargs (dict):
        """
        AbstractResultPackage.__init__(self, strict_content_check, **kwargs)

    def prepare_results_dict(self):
        """

        Returns:
            None:
        """
        accuracy_result = AccuracyMetric(self.y_true, self.y_predicted).get_metric_dict()

        # Causing problems in default mode. With proper selection of parameters it could work for multiclass
        # roc_auc_result = ROCAUCMetric(self.y_true, self.y_predicted).get_metric_dict()
        # f1_score_result = F1ScoreMetric(self.y_true, self.y_predicted).get_metric_dict()
        # self.results_dict = {**accuracy_result, **roc_auc_result}

        self.results_dict = accuracy_result

        
class RegressionResultPackage(AbstractResultPackage):
    def __init__(self, strict_content_check=False, **kwargs):
        """

        Args:
            strict_content_check (bool):
            **kwargs (dict):
        """
        AbstractResultPackage.__init__(self, strict_content_check, **kwargs)

    def prepare_results_dict(self):
        """

        Returns:
            None:
        """
        mse_result = MeanSquaredErrorMetric(self.y_true, self.y_predicted).get_metric_dict()
        mae_result = MeanAbsoluteErrorMetric(self.y_true, self.y_predicted).get_metric_dict()

        self.results_dict = {**mse_result, **mae_result}


class MultipleResultPackageWrapper(AbstractResultPackage):
    def __init__(self, result_packages, strict_content_check=False, **kwargs):
        """

        Args:
            result_packages (list):
            strict_content_check (bool):
            **kwargs (dict):
        """
        self.result_packages = result_packages
        AbstractResultPackage.__init__(self, strict_content_check, **kwargs)

    def prepare_results_dict(self):
        self.results_dict = {}

        for i, result_pkg in enumerate(self.result_packages):
            if result_pkg.pkg_name is not None:
                suffix = '' if result_pkg.pkg_name not in self.results_dict else str(i)
                self.results_dict[result_pkg.pkg_name + suffix] = result_pkg.get_results()
            else:
                self.results_dict[f'ResultPackage{i}'] = result_pkg.get_results()

    def add_merge_multi_pkg_wrap(self, other_object):
        """

        Args:
            other_object (AIToolbox.experiment_save.result_package.AbstractResultPackage or dict):

        Returns:
            AIToolbox.experiment_save.result_package.MultipleResultPackageWrapper:
        """
        self.warn_if_results_dict_not_defined()
        other_object_pkg = self.create_other_object_pkg(other_object)

        self_multi_result_pkg = copy.deepcopy(self)

        self_multi_result_pkg.result_packages.append(other_object_pkg)
        self_multi_result_pkg.prepare_result_package(self_multi_result_pkg.y_true, self_multi_result_pkg.y_predicted,
                                                     self_multi_result_pkg.hyperparameters,
                                                     self_multi_result_pkg.training_history,
                                                     **self_multi_result_pkg.package_metadata)
        return self_multi_result_pkg

    def __iadd__(self, other):
        """

        Args:
            other (AIToolbox.experiment_save.result_package.AbstractResultPackage or dict):

        Returns:
            AIToolbox.experiment_save.result_package.MultipleResultPackageWrapper:
        """
        self.warn_if_results_dict_not_defined()
        other_object_pkg = self.create_other_object_pkg(other)

        self.result_packages.append(other_object_pkg)
        self.prepare_result_package(self.y_true, self.y_predicted,
                                    self.hyperparameters, self.training_history, **self.package_metadata)
        return self
