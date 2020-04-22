import os
from transformers.data.metrics.squad_metrics import compute_predictions_logits, squad_evaluate
from transformers.data.processors.squad import SquadResult
from aitoolbox.experiment.result_package.abstract_result_packages import AbstractResultPackage


class DDPBERTQAResultPackage(AbstractResultPackage):
    def __init__(self, examples, features, output_dir, tokenizer,
                 n_best_size=20, max_answer_length=30, do_lower_case=True,
                 strict_content_check=False, **kwargs):
        super().__init__('BERTQAResultPackage', strict_content_check, np_array=False, **kwargs)
        self.examples = examples
        self.features = features
        self.output_dir = os.path.expanduser(output_dir)

        self.tokenizer = tokenizer

        self.n_best_size = n_best_size
        self.max_answer_length = max_answer_length
        self.do_lower_case = do_lower_case

        # Compute predictions
        self.output_prediction_file = os.path.join(self.output_dir, "predictions.json")
        self.output_nbest_file = os.path.join(self.output_dir, "nbest_predictions.json")
        self.output_null_log_odds_file = os.path.join(self.output_dir, "null_odds.json")

    def prepare_results_dict(self):
        start_logits, end_logits = self.y_predicted.unbind(dim=2)
        start_logits = start_logits.detach().cpu().tolist()
        end_logits = end_logits.detach().cpu().tolist()
        example_indices = self.additional_results['additional_results']['example_indices']

        all_results = [SquadResult(int(self.features[int(ex_idx)].unique_id), start_log, end_log)
                       for ex_idx, start_log, end_log in zip(example_indices, start_logits, end_logits)]

        predictions = compute_predictions_logits(
            self.examples,
            self.features,
            all_results,
            self.n_best_size,
            self.max_answer_length,
            self.do_lower_case,
            self.output_prediction_file,
            self.output_nbest_file,
            self.output_null_log_odds_file,
            True,
            True,
            0.,
            self.tokenizer
        )

        # Compute the F1 and exact scores.
        results = squad_evaluate(self.examples, predictions)
        return dict(results)
