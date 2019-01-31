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
    def __init__(self, lr_decay):
        AbstractCallback.__init__(self, 'Reduce learn rate if the model hits the plateau')
        self.lr_decay = lr_decay

    def on_epoch_end(self):
        pass
