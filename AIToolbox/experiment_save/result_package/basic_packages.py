import copy

from AIToolbox.experiment_save.result_package.abstract_result_package import AbstractResultPackage
from AIToolbox.experiment_save.core_metrics.abstract_metric import AbstractBaseMetric
from AIToolbox.experiment_save.core_metrics.classification import AccuracyMetric, ROCAUCMetric, \
    PrecisionRecallCurveAUCMetric, F1ScoreMetric
from AIToolbox.experiment_save.core_metrics.regression import MeanSquaredErrorMetric, MeanAbsoluteErrorMetric


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

    def __str__(self):
        return '\n'.join([f'--> {pkg.pkg_name}:\n{str(pkg)}' for pkg in self.result_packages])

    def __len__(self):
        return len(self.result_packages)

    def add_merge_multi_pkg_wrap(self, other_object):
        """

        Args:
            other_object (AIToolbox.experiment_save.result_package.abstract_result_package.AbstractResultPackage or dict):

        Returns:
            AIToolbox.experiment_save.result_package.basic_packages.MultipleResultPackageWrapper:
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
            other (AIToolbox.experiment_save.abstract_result_package.AbstractResultPackage or dict):

        Returns:
            AIToolbox.experiment_save.result_package.basic_packages.MultipleResultPackageWrapper:
        """
        self.warn_if_results_dict_not_defined()
        other_object_pkg = self.create_other_object_pkg(other)

        self.result_packages.append(other_object_pkg)
        self.prepare_result_package(self.y_true, self.y_predicted,
                                    self.hyperparameters, self.training_history, **self.package_metadata)
        return self
