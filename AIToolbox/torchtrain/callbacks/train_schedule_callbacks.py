from AIToolbox.torchtrain.callbacks.callbacks import AbstractCallback


class BasicLearnRateScheduler(AbstractCallback):
    def __init__(self, lr_decay):
        AbstractCallback.__init__(self, 'Model save at the end of training')
        self.lr_decay = lr_decay
        
    def on_batch_end(self, train_loop_obj):
        pass
    
    def on_epoch_end(self, train_loop_obj):
        pass
