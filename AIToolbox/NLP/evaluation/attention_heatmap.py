import seaborn as sns

from AIToolbox.experiment_save.core_metrics.base_metric import AbstractBaseMetric


class AttentionHeatMap(AbstractBaseMetric):
    def __init__(self, attention_matrix, attention_sent_pair):
        """

        Args:
            attention_matrix (numpy.array):
            attention_sent_pair (list):
        """
        self.attention_matrix = attention_matrix
        AbstractBaseMetric.__init__(self, None, None)

        attention_sent = attention_sent_pair[0]
        self.metric_name = 'AttentionHeatMap_' + attention_sent if len(attention_sent) < 20 else attention_sent[:20]

    def calculate_metric(self):
        raise NotImplementedError


class AttentionMultipleHeatMaps(AttentionHeatMap):
    def __init__(self, attention_matrices, attention_sent_pairs):
        """

        Args:
            attention_matrices (numpy.array):
            attention_sent_pairs (list):
        """
        self.attention_matrices = attention_matrices
        self.attention_sent_pairs = attention_sent_pairs

        AbstractBaseMetric.__init__(self, None, None)
        self.metric_name = 'AttentionMultipleHeatMaps'

    def calculate_metric(self):
        raise NotImplementedError

        self.metric_result = {attn_sent: self.attention_matrices[i] for i, attn_sent in enumerate(self.attention_sents)}
