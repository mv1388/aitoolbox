from AIToolbox.experiment_save.result_package.abstract_result_packages import AbstractResultPackage
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
