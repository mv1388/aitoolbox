from aitoolbox.experiment.result_package.abstract_result_packages import AbstractResultPackage
from aitoolbox.experiment.core_metrics.abstract_metric import AbstractBaseMetric
from aitoolbox.experiment.core_metrics.classification import AccuracyMetric, ROCAUCMetric, \
    PrecisionRecallCurveAUCMetric, F1ScoreMetric
from aitoolbox.experiment.core_metrics.regression import MeanSquaredErrorMetric, MeanAbsoluteErrorMetric


class GeneralResultPackage(AbstractResultPackage):
    def __init__(self, metrics_list, strict_content_check=False, **kwargs):
        """Result package executing given list of metrics

        Args:
            metrics_list (list): List of objects which are inherited from
                aitoolbox.experiment.core_metrics.BaseMetric.AbstractBaseMetric
            strict_content_check (bool): should just print warning or raise the error and crash
            **kwargs (dict): additional package_metadata for the result package
        """
        AbstractResultPackage.__init__(self, pkg_name='GeneralResultPackage', 
                                       strict_content_check=strict_content_check, **kwargs)
        self.metrics_list = metrics_list

    def prepare_results_dict(self):
        self.qa_check_hyperparameters_dict()
        results_dict = {}

        for metric in self.metrics_list:
            metric_result = metric(self.y_true, self.y_predicted)
            results_dict = results_dict + metric_result
            
        return results_dict

    def qa_check_metrics_list(self):
        if len(self.metrics_list) == 0:
            self.warn_about_result_data_problem('Metrics list is empty')

        for metric in self.metrics_list:
            if not isinstance(metric, AbstractBaseMetric):
                self.warn_about_result_data_problem('Metric is not inherited from AbstractBaseMetric class')


class BinaryClassificationResultPackage(AbstractResultPackage):
    def __init__(self, positive_class_thresh=0.5, strict_content_check=False, **kwargs):
        """Binary classification task result package

        Evaluates the following metrics: accuracy, ROC-AUC, PR-AUC and F1 score

        Args:
            positive_class_thresh (float or None): predicted probability positive class threshold
            strict_content_check (bool): should just print warning or raise the error and crash
            **kwargs (dict): additional package_metadata for the result package
        """
        AbstractResultPackage.__init__(self, pkg_name='BinaryClassificationResult',
                                       strict_content_check=strict_content_check, **kwargs)
        self.positive_class_thresh = positive_class_thresh

    def prepare_results_dict(self):
        accuracy_result = AccuracyMetric(self.y_true, self.y_predicted,
                                         positive_class_thresh=self.positive_class_thresh)
        roc_auc_result = ROCAUCMetric(self.y_true, self.y_predicted)
        pr_auc_result = PrecisionRecallCurveAUCMetric(self.y_true, self.y_predicted)
        f1_score_result = F1ScoreMetric(self.y_true, self.y_predicted,
                                        positive_class_thresh=self.positive_class_thresh)

        return accuracy_result + roc_auc_result + pr_auc_result + f1_score_result


class ClassificationResultPackage(AbstractResultPackage):
    def __init__(self, strict_content_check=False, **kwargs):
        """Multi-class classification result package

        Evaluates the accuracy of the predictions.
        Without Precision-Recall metric which is available only for binary classification problems.

        Args:
            strict_content_check (bool): should just print warning or raise the error and crash
            **kwargs (dict): additional package_metadata for the result package
        """
        AbstractResultPackage.__init__(self, pkg_name='ClassificationResult',
                                       strict_content_check=strict_content_check, **kwargs)

    def prepare_results_dict(self):
        accuracy_result = AccuracyMetric(self.y_true, self.y_predicted, positive_class_thresh=None).get_metric_dict()

        return accuracy_result

        
class RegressionResultPackage(AbstractResultPackage):
    def __init__(self, strict_content_check=False, **kwargs):
        """Regression task result package

        Evaluates MSE and MAE metrics.

        Args:
            strict_content_check (bool): should just print warning or raise the error and crash
            **kwargs (dict): additional package_metadata for the result package
        """
        AbstractResultPackage.__init__(self, pkg_name='RegressionResult', 
                                       strict_content_check=strict_content_check, **kwargs)

    def prepare_results_dict(self):
        mse_result = MeanSquaredErrorMetric(self.y_true, self.y_predicted)
        mae_result = MeanAbsoluteErrorMetric(self.y_true, self.y_predicted)

        return mse_result + mae_result
