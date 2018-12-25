from AIToolbox.ExperimentSave.MetricsGeneral.BaseMetric import AbstractBaseMetric


class AttentionHeatMap(AbstractBaseMetric):
    def __init__(self, attention_matrix, attention_word):
        """

        Args:
            attention_matrix (numpy.array):
            attention_word (str):
        """
        self.attention_matrix = attention_matrix

        AbstractBaseMetric.__init__(self, None, None)
        self.metric_name = 'AttentionHeatMap_' + attention_word

    def calculate_metric(self):
        raise NotImplementedError


class AttentionMultipleHeatMaps(AttentionHeatMap):
    def __init__(self, attention_matrices, attention_words):
        """

        Args:
            attention_matrices (numpy.array):
            attention_words (list):
        """
        self.attention_matrices = attention_matrices
        self.attention_words = attention_words

        AbstractBaseMetric.__init__(self, None, None)
        self.metric_name = 'AttentionMultipleHeatMaps'

    def calculate_metric(self):
        raise NotImplementedError

        self.metric_result = {attn_word: self.attention_matrices[i] for i, attn_word in enumerate(self.attention_words)}

