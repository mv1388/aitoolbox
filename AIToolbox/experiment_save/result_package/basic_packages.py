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
        AbstractResultPackage.__init__(self, pkg_name='GeneralResultPackage', 
                                       strict_content_check=strict_content_check, **kwargs)
        self.metrics_list = metrics_list

    def prepare_results_dict(self):
        """

        Returns:
            dict:
        """
        self.qa_check_hyperparameters_dict()
        results_dict = {}

        for metric in self.metrics_list:
            metric_result = metric(self.y_true, self.y_predicted)
            results_dict = results_dict + metric_result
            
        return results_dict

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
        AbstractResultPackage.__init__(self, pkg_name='BinaryClassificationResult', 
                                       strict_content_check=strict_content_check, **kwargs)

    def prepare_results_dict(self):
        """

        Returns:
            dict:
        """
        accuracy_result = AccuracyMetric(self.y_true, self.y_predicted)
        roc_auc_result = ROCAUCMetric(self.y_true, self.y_predicted)
        pr_auc_result = PrecisionRecallCurveAUCMetric(self.y_true, self.y_predicted)
        f1_score_result = F1ScoreMetric(self.y_true, self.y_predicted)

        return accuracy_result + roc_auc_result + pr_auc_result + f1_score_result


class ClassificationResultPackage(AbstractResultPackage):
    def __init__(self, strict_content_check=False, **kwargs):
        """

        Without Precision-Recall metric which is available only for binary classification problems.

        Args:
            strict_content_check (bool):
            **kwargs (dict):
        """
        AbstractResultPackage.__init__(self, pkg_name='ClassificationResult', 
                                       strict_content_check=strict_content_check, **kwargs)

    def prepare_results_dict(self):
        """

        Returns:
            dict:
        """
        accuracy_result = AccuracyMetric(self.y_true, self.y_predicted).get_metric_dict()

        # Causing problems in default mode. With proper selection of parameters it could work for multiclass
        # roc_auc_result = ROCAUCMetric(self.y_true, self.y_predicted)
        # f1_score_result = F1ScoreMetric(self.y_true, self.y_predicted)

        return accuracy_result

        
class RegressionResultPackage(AbstractResultPackage):
    def __init__(self, strict_content_check=False, **kwargs):
        """

        Args:
            strict_content_check (bool):
            **kwargs (dict):
        """
        AbstractResultPackage.__init__(self, pkg_name='RegressionResult', 
                                       strict_content_check=strict_content_check, **kwargs)

    def prepare_results_dict(self):
        """

        Returns:
            dict:
        """
        mse_result = MeanSquaredErrorMetric(self.y_true, self.y_predicted)
        mae_result = MeanAbsoluteErrorMetric(self.y_true, self.y_predicted)

        return mse_result + mae_result
