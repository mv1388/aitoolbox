import unittest

from aitoolbox.experiment.result_package.hf_evaluate_packages import HFEvaluateResultPackage


class DummyHFEvaluateMetric:
    def compute(self, **kwargs):
        return kwargs


class TestHFEvaluateResultPackage(unittest.TestCase):
    def test_models_additional_inputs_parsing(self):
        metric = DummyHFEvaluateMetric()
        result_package = HFEvaluateResultPackage(hf_evaluate_metric=metric, use_models_additional_results=True)
        result_package.y_true = 1
        result_package.y_predicted = 2
        result_package.additional_results = {'additional_results': {'aaa': 3}}
        result_dict = result_package.prepare_results_dict()
        self.assertEqual(result_dict, {'references_HFEvaluate': 1, 'predictions_HFEvaluate': 2, 'aaa_HFEvaluate': 3})

        metric = DummyHFEvaluateMetric()
        result_package = HFEvaluateResultPackage(hf_evaluate_metric=metric, use_models_additional_results=False)
        result_package.y_true = 1
        result_package.y_predicted = 2
        result_package.additional_results = {'additional_results': {'aaa': 3}}
        result_dict = result_package.prepare_results_dict()
        self.assertEqual(result_dict, {'references_HFEvaluate': 1, 'predictions_HFEvaluate': 2})

    def test_additional_inputs_combination(self):
        metric = DummyHFEvaluateMetric()
        result_package = HFEvaluateResultPackage(
            hf_evaluate_metric=metric, use_models_additional_results=True,
            my_additional_input_1=123, my_additional_input_2='ABCD'
        )
        result_package.y_true = 1
        result_package.y_predicted = 2
        result_package.additional_results = {'additional_results': {'aaa': 3}}
        result_dict = result_package.prepare_results_dict()
        self.assertEqual(
            result_dict,
            {'references_HFEvaluate': 1, 'predictions_HFEvaluate': 2,
             'my_additional_input_1_HFEvaluate': 123, 'my_additional_input_2_HFEvaluate': 'ABCD',
             'aaa_HFEvaluate': 3}
        )

        metric = DummyHFEvaluateMetric()
        result_package = HFEvaluateResultPackage(
            hf_evaluate_metric=metric, use_models_additional_results=False,
            my_additional_input_1=123, my_additional_input_2='ABCD'
        )
        result_package.y_true = 1
        result_package.y_predicted = 2
        result_package.additional_results = {'additional_results': {'aaa': 3}}
        result_dict = result_package.prepare_results_dict()
        self.assertEqual(
            result_dict,
            {'references_HFEvaluate': 1, 'predictions_HFEvaluate': 2,
             'my_additional_input_1_HFEvaluate': 123, 'my_additional_input_2_HFEvaluate': 'ABCD'}
        )
