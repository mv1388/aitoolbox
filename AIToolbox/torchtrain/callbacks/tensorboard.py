import numpy as np
from tensorboardX import SummaryWriter

from AIToolbox.torchtrain.callbacks.callbacks import AbstractCallback


class TBBatchLossReport(AbstractCallback):
    def __init__(self, tb_logdir=None, comment='', flush_secs=120, filename_suffix=''):
        """

        Args:
            tb_logdir (str):
            comment (str):
            flush_secs (int):
            filename_suffix (str):
        """
        AbstractCallback.__init__(self, 'Tensorboard end of batch report of batch loss')

        self.tb_writer = SummaryWriter(logdir=tb_logdir, comment=comment,
                                       flush_secs=flush_secs, filename_suffix=filename_suffix)

    def on_batch_end(self):
        self.tb_writer.add_scalar('data/last_batch_loss', self.train_loop_obj.loss_batch_accum[-1])
        self.tb_writer.add_scalar('data/accumulated_batch_loss', np.mean(self.train_loop_obj.loss_batch_accum).item())


class TBPerformanceMetricReport(AbstractCallback):
    def __init__(self, metric_names=None, tb_logdir=None, comment='', flush_secs=120, filename_suffix=''):
        """

        Args:
            metric_names (list or None):
            tb_logdir (str):
            comment (str):
            flush_secs (int):
            filename_suffix (str):
        """
        AbstractCallback.__init__(self, 'Tensorboard end of epoch report of all the recorded measures in training hist')

        self.metric_names = metric_names

        self.tb_writer = SummaryWriter(logdir=tb_logdir, comment=comment,
                                       flush_secs=flush_secs, filename_suffix=filename_suffix)

    def on_epoch_end(self):
        metric_names = self.metric_names if self.metric_names is not None else self.train_loop_obj.train_history.keys()

        for metric_name in metric_names:
            metric_results = self.train_loop_obj.train_history[metric_name]
            self.tb_writer.add_scalar(f'data/{metric_name}', metric_results[-1])


class TBAttentionReport(AbstractCallback):
    def __init__(self):
        AbstractCallback.__init__(self, 'Attention heatmap')

    def on_epoch_end(self):
        raise NotImplementedError


class TBEmbeddingReport(AbstractCallback):
    def __init__(self):
        AbstractCallback.__init__(self, 'Neural network embeddings')

    def on_epoch_end(self):
        raise NotImplementedError


class TBHistogramReport(AbstractCallback):
    def __init__(self):
        AbstractCallback.__init__(self, 'Neural network layers histogram')

    def on_batch_end(self):
        raise NotImplementedError

    def on_epoch_end(self):
        raise NotImplementedError


class TBHImageReport(AbstractCallback):
    def __init__(self):
        """

            TODO: use image and images fns

        """
        AbstractCallback.__init__(self, 'Image result')

    def on_epoch_end(self):
        raise NotImplementedError
