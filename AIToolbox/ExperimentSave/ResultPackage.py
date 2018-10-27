from abc import ABC, abstractmethod
import os

from AIToolbox.ExperimentSave.MetricsGeneral.Classification import AccuracyMetric, ROCAUCMetric


class AbstractResultPackage(ABC):
    def __init__(self, y_true, y_predicted, hyperparameters=None,
                 strict_content_check=False):
        """

        Args:
            y_true (numpy.array or list):
            y_predicted (numpy.array or list):
            hyperparameters (dict):
            strict_content_check (bool):
        """
        self.strict_content_check = strict_content_check

        self.y_true = y_true
        self.y_predicted = y_predicted

        self.results_dict = None
        self.hyperparameters = hyperparameters

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


class ClassificationResultPackage(AbstractResultPackage):
    def __init__(self, y_true, y_predicted, hyperparameters=None, strict_content_check=False):
        """

        Args:
            y_true (numpy.array or list):
            y_predicted (numpy.array or list):
            hyperparameters (dict):
            strict_content_check (bool):
        """
        AbstractResultPackage.__init__(self, y_true, y_predicted, hyperparameters, strict_content_check)

    def prepare_results_dict(self):
        """

        Returns:
            None:
        """
        accuracy_result = AccuracyMetric(self.y_true, self.y_predicted).get_metric_dict()
        roc_auc_result = ROCAUCMetric(self.y_true, self.y_predicted).get_metric_dict()

        self.results_dict = {**accuracy_result, **roc_auc_result}
