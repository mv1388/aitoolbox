import numpy as np
from tensorboardX import SummaryWriter

from AIToolbox.torchtrain.callbacks.callbacks import AbstractCallback


class TBBatchLossReport(AbstractCallback):
    def __init__(self, tb_logdir=None, comment='', flush_secs=120, filename_suffix=''):
        AbstractCallback.__init__(self, 'Tensorboard end of batch report of batch loss')

        self.tb_writer = SummaryWriter(logdir=tb_logdir, comment=comment,
                                       flush_secs=flush_secs, filename_suffix=filename_suffix)

    def on_batch_end(self):
        self.tb_writer.add_scalar('data/last_batch_loss', self.train_loop_obj.loss_batch_accum[-1])
        self.tb_writer.add_scalar('data/accumulated_batch_loss', np.mean(self.train_loop_obj.loss_batch_accum).item())
