from AIToolbox.experimet_save.core_metrics.base_metric import AbstractBaseMetric


class AttentionHeatMap(AbstractBaseMetric):
    def __init__(self, attention_matrix, attention_sent):
        """

        Args:
            attention_matrix (numpy.array):
            attention_sent (str):
        """
        self.attention_matrix = attention_matrix

        AbstractBaseMetric.__init__(self, None, None)
        self.metric_name = 'AttentionHeatMap_' + attention_sent if len(attention_sent) < 20 else attention_sent[:20]

    def calculate_metric(self):
        raise NotImplementedError


class AttentionMultipleHeatMaps(AttentionHeatMap):
    def __init__(self, attention_matrices, attention_sents):
        """

        Args:
            attention_matrices (numpy.array):
            attention_sents (list):
        """
        self.attention_matrices = attention_matrices
        self.attention_sents = attention_sents

        AbstractBaseMetric.__init__(self, None, None)
        self.metric_name = 'AttentionMultipleHeatMaps'

    def calculate_metric(self):
        raise NotImplementedError

        self.metric_result = {attn_sent: self.attention_matrices[i] for i, attn_sent in enumerate(self.attention_sents)}
