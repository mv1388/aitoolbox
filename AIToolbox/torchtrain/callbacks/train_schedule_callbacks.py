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


# class ReduceLROnPlateauScheduler(AbstractCallback):
#     def __init__(self, **kwargs):
#         """
#
#         def __init__(self, mode='min', factor=0.1, patience=10,
#                  verbose=False, threshold=1e-4, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-8):
#
#         Args:
#             mode:
#             factor:
#             patience:
#             verbose:
#             threshold:
#             threshold_mode:
#             cooldown:
#             min_lr:
#             eps:
#         """
#         AbstractCallback.__init__(self, 'Reduce learn rate if the model hits the plateau')
#         self.scheduler_args = kwargs
#         self.scheduler = None
#
#     def register_train_loop_obj(self, train_loop_obj):
#         self.train_loop_obj = train_loop_obj
#         self.scheduler = ReduceLROnPlateau(self.train_loop_obj.optimizer, **self.scheduler_args)
#         return self
#
#     def on_epoch_end(self):
#         val_loss_avg = self.train_loop_obj.evaluate_loss_on_validation()
#         self.scheduler.step(val_loss_avg)


# class GeneralLRScheduler(ReduceLROnPlateauScheduler):
#     def __init__(self, scheduler_class, **kwargs):
#         """
#
#         Args:
#             scheduler_class:
#             **kwargs:
#         """
#         ReduceLROnPlateauScheduler.__init__(self, **kwargs)
#         self.scheduler_class = scheduler_class
#
#     def register_train_loop_obj(self, train_loop_obj):
#         self.train_loop_obj = train_loop_obj
#         self.scheduler = self.scheduler_class(self.train_loop_obj.optimizer, **self.scheduler_args)
#         return self


class GeneralLRScheduler(AbstractCallback):
    def __init__(self, scheduler_class, **kwargs):
        """

        Args:
            **kwargs:
        """
        AbstractCallback.__init__(self, 'General learn rate scheduler')
        self.scheduler_args = kwargs
        self.scheduler_class = scheduler_class
        self.scheduler = None

    def register_train_loop_obj(self, train_loop_obj):
        self.train_loop_obj = train_loop_obj
        self.scheduler = self.scheduler_class(self.train_loop_obj.optimizer, **self.scheduler_args)
        return self

    def on_epoch_end(self):
        val_loss_avg = self.train_loop_obj.evaluate_loss_on_validation()
        self.scheduler.step(val_loss_avg)


class ReduceLROnPlateauScheduler(GeneralLRScheduler):
    def __init__(self, **kwargs):
        """

        def __init__(self, mode='min', factor=0.1, patience=10,
                 verbose=False, threshold=1e-4, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-8):

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
        GeneralLRScheduler.__init__(self, ReduceLROnPlateau, **kwargs)
        self.callback_name = 'Reduce learn rate if the model hits the plateau'
