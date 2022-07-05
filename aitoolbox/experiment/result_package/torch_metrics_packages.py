from aitoolbox.experiment.result_package.abstract_result_packages import AbstractResultPackage


class TorchMetricsPackage(AbstractResultPackage):
    def __init__(self, torch_metrics):
        """Torch Metrics result package wrapper

        https://github.com/Lightning-AI/metrics

        Args:
            torch_metrics (torchmetrics.Metric or torchmetrics.MetricCollection): single torchmetrics metric object or
                a collection of such metrics wrapped inside the MetricCollection
        """
        AbstractResultPackage.__init__(self, pkg_name='Torch Metrics', np_array=False)

        self.metric = torch_metrics

    def prepare_results_dict(self):
        metric_result = self.metric(self.y_predicted, self.y_true)

        if not isinstance(metric_result, dict):
            metric_result = {self.metric.__class__.__name__: metric_result}

        # Add suffix PTLMetrics to indicate that we are using PyTorch Lightning metrics instead of aitoolbox metric
        metric_result = {f'{k}_PTLMetrics': v for k, v in metric_result.items()}

        return metric_result

    def metric_compute(self):
        return self.metric.compute()

    def metric_reset(self):
        self.metric.reset()
