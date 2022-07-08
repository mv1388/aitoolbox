from aitoolbox.experiment.result_package.abstract_result_packages import AbstractResultPackage


class HFEvaluateResultPackage(AbstractResultPackage):
    def __init__(self, hf_evaluate_metric, use_models_additional_results=True, **kwargs):
        """HuggingFace Evaluate Metrics Result Package

        Result package wrapping around the evaluation metrics provided in the HuggingFace Evaluate package.

        All the metric result names will have the '_HFEvaluate' appended at the end to help distinguish them.

        Github: https://github.com/huggingface/evaluate
        More info on how to use the metrics: https://huggingface.co/docs/evaluate/index

        Args:
            hf_evaluate_metric (evaluate.EvaluationModule): HF Evaluate metric to be used by the result package
            use_models_additional_results (bool): Should the additional results from the model
                (in addition to predictions and references) normally returned from the get_predictions() function be
                added as the additional input to the HF Evaluate metric to perform the evaluation calculation.
            **kwargs: additional parameters or inputs to the HF Evaluate metric being calculated. These can be generally
                inputs available already at the start before making model predictions and thus don't need to be gathered
                from the train/prediction loop.
        """
        AbstractResultPackage.__init__(self, pkg_name='HuggingFace Evaluate metrics', **kwargs)

        self.metric = hf_evaluate_metric
        self.use_models_additional_results = use_models_additional_results

    def prepare_results_dict(self):
        additional_metric_inputs = self.package_metadata

        if self.use_models_additional_results:
            model_additional_results = self.additional_results['additional_results']
            additional_metric_inputs = {**additional_metric_inputs, **model_additional_results}

        metric_result = self.metric.compute(
            references=self.y_true, predictions=self.y_predicted,
            **additional_metric_inputs
        )

        if isinstance(metric_result, dict):
            metric_result = {f'{k}_HFEvaluate': v for k, v in metric_result.items()}

        return metric_result
