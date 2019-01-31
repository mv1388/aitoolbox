from torch.optim.lr_scheduler import ReduceLROnPlateau

from AIToolbox.torchtrain.callbacks.callbacks import AbstractCallback


class BasicLearnRateScheduler(AbstractCallback):
    def __init__(self, lr_decay):
        AbstractCallback.__init__(self, 'Model save at the end of training')
        self.lr_decay = lr_decay
        
    def on_batch_end(self):
        pass
    
    def on_epoch_end(self):
        pass


class ReduceLROnPlateauScheduler(AbstractCallback):
    def __init__(self, mode='min', factor=0.1, patience=10,
                 verbose=False, threshold=1e-4, threshold_mode='rel',
                 cooldown=0, min_lr=0, eps=1e-8):
        """

        Args:
            mode:
            factor:
            patience:
            verbose:
            threshold:
            threshold_mode:
            cooldown:
            min_lr:
            eps:
        """
        AbstractCallback.__init__(self, 'Reduce learn rate if the model hits the plateau')
        self.scheduler = ReduceLROnPlateau(self.train_loop_obj.optimizer,
                                           mode, factor, patience,
                                           verbose, threshold, threshold_mode,
                                           cooldown, min_lr, eps)

    def on_epoch_end(self):
        val_loss_avg = self.train_loop_obj.evaluate_loss_on_validation()
        self.scheduler.step(val_loss_avg)
