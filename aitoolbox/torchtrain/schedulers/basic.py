from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR, StepLR, MultiStepLR

from aitoolbox.torchtrain.callbacks.abstract import AbstractCallback


class AbstractScheduler:
    def __init__(self):
        """Scheduler (callback) base class

        All the scheduler callbacks should in addition to ``AbstractCallback`` also inherit from this base class.
        This class serves to indicate to torchtrain components which used callbacks are schedulers and which are just
        normal callbacks which have nothing to do with learning rate scheduling.

        In addition to the above, this scheduler base class also implements the interface methods needed for saving
        and loading the scheduler state_dict's when checkpointing and reloading the scheduler.

        When implementing the actual scheduler callback make sure to assign the created learning rate scheduler
        to the ``self.scheduler`` class member.
        """
        self.scheduler = None

    def state_dict(self):
        return self.scheduler.state_dict()

    def load_state_dict(self, state_dict):
        self.scheduler.load_state_dict(state_dict)


class GeneralLRSchedulerCallback(AbstractScheduler, AbstractCallback):
    def __init__(self, scheduler_class, optimizer_idx=None, **kwargs):
        """Learning rate scheduler base class

        Args:
            scheduler_class: PyTorch learning rate scheduler class
            optimizer_idx (int or torch.optim.optimizer.Optimizer or None): index or the actual object reference
                of the paired optimizer when using multiple optimizers
            **kwargs: learning rate scheduler additional parameters
        """
        AbstractScheduler.__init__(self)
        AbstractCallback.__init__(self, 'General learn rate scheduler')
        self.scheduler_args = kwargs
        self.scheduler_class = scheduler_class
        self.optimizer_idx = optimizer_idx

    def register_train_loop_object(self, train_loop_obj):
        """Modified register_train_loop_object method to support scheduler creation

        Args:
            train_loop_obj (aitoolbox.torchtrain.train_loop.TrainLoop): reference to the encapsulating TrainLoop

        Returns:
            AbstractCallback: return the reference to the callback after it is registered
        """
        self.train_loop_obj = train_loop_obj
        self.message_service = train_loop_obj.message_service

        if self.optimizer_idx is None:
            optimizer = self.train_loop_obj.optimizer
        elif type(self.optimizer_idx) == int:
            optimizer = self.train_loop_obj.optimizer[self.optimizer_idx]
        else:
            optimizer = self.optimizer_idx

        self.scheduler = self.scheduler_class(optimizer, **self.scheduler_args)
        self.on_train_loop_registration()
        return self

    def on_epoch_end(self):
        self.scheduler.step()


class ReduceLROnPlateauScheduler(GeneralLRSchedulerCallback):
    def __init__(self, **kwargs):
        """Learning rate scheduler which reduces the rate if the loss performance stops improving

        Args:
            **kwargs: learning rate scheduler additional parameters
        """
        GeneralLRSchedulerCallback.__init__(self, ReduceLROnPlateau, **kwargs)
        self.callback_name = 'Reduce learn rate if the model hits the plateau'

    def on_epoch_end(self):
        val_loss_avg = self.train_loop_obj.evaluate_loss_on_validation_set()
        self.scheduler.step(val_loss_avg)


class ReduceLROnPlateauMetricScheduler(GeneralLRSchedulerCallback):
    def __init__(self, metric_name, **kwargs):
        """Learning rate scheduler which reduces the rate if the performance of the selected metric stops improving

        Needs to be used in combination with ModelPerformanceEvaluation to calculate the metric and fill it in
        the TrainLoop history.

        Args:
            metric_name (str): monitored metric based on which the learning rate scheduler modifies the learning rate
            **kwargs: learning rate scheduler additional parameters
        """
        GeneralLRSchedulerCallback.__init__(self, ReduceLROnPlateau, **kwargs)
        self.metric_name = metric_name
        self.callback_name = 'Reduce learn rate if the model hits the plateau based on metric in TrainLoop history'

    def on_epoch_end(self):
        if self.metric_name not in self.train_loop_obj.train_history:
            raise ValueError(
                f'Metric {self.metric_name} expected for the report missing from TrainLoop.train_history. '
                f'Found only the following: {self.train_loop_obj.train_history.keys()}')

        val_metric_result = self.train_loop_obj.train_history[self.metric_name][-1]
        self.scheduler.step(val_metric_result)


class LambdaLRScheduler(GeneralLRSchedulerCallback):
    def __init__(self, lr_lambda, execute_epoch_end=True, execute_batch_end=False, **kwargs):
        """Sets the learning rate of each parameter group to the initial lr times a given function

        When last_epoch=-1, sets initial lr as lr.

        Args:
            lr_lambda (callable or list): A function or a list of functions which computes a multiplicative factor given
                an integer parameter epoch, or a list of such functions, one for each group in optimizer.param_groups.
            execute_epoch_end (bool): should scheduler step be executed at the end of the epoch
            execute_batch_end (bool): should scheduler step be executed at the end of each batch
            **kwargs: learning rate scheduler additional parameters
        """
        GeneralLRSchedulerCallback.__init__(self, LambdaLR, **dict(kwargs, lr_lambda=lr_lambda))
        self.callback_name = ''
        self.execute_epoch_end = execute_epoch_end
        self.execute_batch_end = execute_batch_end

    def on_epoch_end(self):
        if self.execute_epoch_end:
            self.scheduler.step()

    def on_batch_end(self):
        if self.execute_batch_end:
            if self.train_loop_obj.should_execute_optimizer_update():
                self.scheduler.step()


class StepLRScheduler(GeneralLRSchedulerCallback):
    def __init__(self, step_size, **kwargs):
        """Sets the learning rate of each parameter group to the initial lr decayed by gamma every step_size epochs

        When last_epoch=-1, sets initial lr as lr.

        Args:
            step_size (int): period of learning rate decay
            **kwargs: learning rate scheduler additional parameters
        """
        GeneralLRSchedulerCallback.__init__(self, StepLR, **dict(kwargs, step_size=step_size))
        self.callback_name = ''


class MultiStepLRScheduler(GeneralLRSchedulerCallback):
    def __init__(self, milestones_list, **kwargs):
        """Set the learning rate of each parameter group to the initial lr decayed by gamma once the number of epoch
            reaches one of the milestones.

        When last_epoch=-1, sets initial lr as lr.

        Args:
            milestones_list (list): list of epoch indices. Must be increasing
            **kwargs: learning rate scheduler additional parameters
        """
        GeneralLRSchedulerCallback.__init__(self, MultiStepLR, **dict(kwargs, milestones=milestones_list))
        self.callback_name = ''
