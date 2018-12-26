from abc import ABC, abstractmethod
import numpy as np

from AIToolbox.experiment_save.core_metrics.base_metric import AbstractBaseMetric

from AIToolbox.experiment_save.core_metrics.classification import AccuracyMetric, ROCAUCMetric, \
    PrecisionRecallCurveAUCMetric, F1ScoreMetric
from AIToolbox.experiment_save.core_metrics.regression import MeanSquaredErrorMetric, MeanAbsoluteErrorMetric


class AbstractResultPackage(ABC):
    def __init__(self, y_true, y_predicted, hyperparameters=None, training_history=None,
                 strict_content_check=False):
        """

        Args:
            y_true (numpy.array or list):
            y_predicted (numpy.array or list):
            hyperparameters (dict or None):
            training_history (AIToolbox.ExperimentSave.training_history.AbstractTrainingHistory):
            strict_content_check (bool):
        """
        self.pkg_name = None
        self.strict_content_check = strict_content_check

        self.y_true = np.array(y_true)
        self.y_predicted = np.array(y_predicted)

        self.results_dict = None
        self.hyperparameters = hyperparameters
        self.training_history = training_history.get_train_history()

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

    @abstractmethod
    def prepare_results_dict(self):
        pass


class GeneralResultPackage(AbstractResultPackage):
    def __init__(self, y_true, y_predicted, metrics_list, hyperparameters=None, training_history=None,
                 strict_content_check=False):
        """

        Args:
            y_true (numpy.array or list):
            y_predicted (numpy.array or list):
            metrics_list (list): List of objects which are inherited from
                AIToolbox.experiment_save.core_metrics.BaseMetric.AbstractBaseMetric
            hyperparameters (dict):
            training_history (AIToolbox.ExperimentSave.training_history.AbstractTrainingHistory):
            strict_content_check (bool):
        """
        self.metrics_list = metrics_list
        AbstractResultPackage.__init__(self, y_true, y_predicted, hyperparameters, training_history, strict_content_check)

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
    def __init__(self, y_true, y_predicted, hyperparameters=None, training_history=None, strict_content_check=False):
        """

        Args:
            y_true (numpy.array or list):
            y_predicted (numpy.array or list):
            hyperparameters (dict):
            training_history (AIToolbox.ExperimentSave.training_history.AbstractTrainingHistory):
            strict_content_check (bool):
        """
        AbstractResultPackage.__init__(self, y_true, y_predicted, hyperparameters, training_history, strict_content_check)

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
    def __init__(self, y_true, y_predicted, hyperparameters=None, training_history=None, strict_content_check=False):
        """

        Without Precision-Recall metric which is available only for binary classification problems.

        Args:
            y_true (numpy.array or list):
            y_predicted (numpy.array or list):
            hyperparameters (dict):
            training_history (AIToolbox.ExperimentSave.training_history.AbstractTrainingHistory):
            strict_content_check (bool):
        """
        AbstractResultPackage.__init__(self, y_true, y_predicted, hyperparameters, training_history, strict_content_check)

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
    def __init__(self, y_true, y_predicted, hyperparameters=None, training_history=None, strict_content_check=False):
        """

        Args:
            y_true (numpy.array or list):
            y_predicted (numpy.array or list):
            hyperparameters (dict):
            training_history (AIToolbox.ExperimentSave.training_history.AbstractTrainingHistory):
            strict_content_check (bool):
        """
        AbstractResultPackage.__init__(self, y_true, y_predicted, hyperparameters, training_history, strict_content_check)

    def prepare_results_dict(self):
        """

        Returns:
            None:
        """
        mse_result = MeanSquaredErrorMetric(self.y_true, self.y_predicted).get_metric_dict()
        mae_result = MeanAbsoluteErrorMetric(self.y_true, self.y_predicted).get_metric_dict()

        self.results_dict = {**mse_result, **mae_result}


class MultipleResultPackageWrapper(AbstractResultPackage):
    def __init__(self, result_packages, hyperparameters=None, training_history=None, strict_content_check=False):
        """

        Args:
            result_packages (list):
            hyperparameters (dict):
            training_history (AIToolbox.ExperimentSave.training_history.AbstractTrainingHistory):
            strict_content_check (bool):
        """
        self.result_packages = result_packages
        AbstractResultPackage.__init__(self, None, None, hyperparameters, training_history, strict_content_check)

    def prepare_results_dict(self):
        self.results_dict = {}

        for i, result_pkg in enumerate(self.result_packages):
            if result_pkg.pkg_name is not None:
                suffix = '' if result_pkg.pkg_name not in self.results_dict else str(i)
                self.results_dict[result_pkg.pkg_name + suffix] = result_pkg.get_results()
            else:
                self.results_dict[f'ResultPackage{i}'] = result_pkg.get_results()
