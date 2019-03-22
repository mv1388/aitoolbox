from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR, StepLR, MultiStepLR

from AIToolbox.torchtrain.callbacks.callbacks import AbstractCallback


class GeneralLRScheduler(AbstractCallback):
    def __init__(self, scheduler_class, **kwargs):
        """

        Args:
            scheduler_class:
            **kwargs:
        """
        AbstractCallback.__init__(self, 'General learn rate scheduler')
        self.scheduler_args = kwargs
        self.scheduler_class = scheduler_class
        self.scheduler = None

    def register_train_loop_object(self, train_loop_obj):
        """

        Args:
            train_loop_obj (AIToolbox.torchtrain.train_loop.TrainLoop):

        Returns:

        """
        self.train_loop_obj = train_loop_obj
        self.scheduler = self.scheduler_class(self.train_loop_obj.optimizer, **self.scheduler_args)
        return self

    def on_epoch_end(self):
        val_loss_avg = self.train_loop_obj.evaluate_loss_on_validation_set()
        self.scheduler.step(val_loss_avg)


class ReduceLROnPlateauScheduler(GeneralLRScheduler):
    def __init__(self, **kwargs):
        """

        Args:
            **kwargs:
        """
        GeneralLRScheduler.__init__(self, ReduceLROnPlateau, **kwargs)
        self.callback_name = 'Reduce learn rate if the model hits the plateau'


class ReduceLROnPlateauMetricScheduler(GeneralLRScheduler):
    def __init__(self, metric_name, **kwargs):
        """

        Needs to be used in combination with ModelPerformanceEvaluationCallback to calculate the metric and fill it in
        the TrainLoop history.

        Args:
            metric_name (str):
            **kwargs:
        """
        GeneralLRScheduler.__init__(self, ReduceLROnPlateau, **kwargs)
        self.metric_name = metric_name
        self.callback_name = 'Reduce learn rate if the model hits the plateau based on metric in TrainLoop history'

    def on_epoch_end(self):
        if self.metric_name not in self.train_loop_obj.train_history:
            raise ValueError(
                f'Metric {self.metric_name} expected for the report missing from TrainLoop.train_history. '
                f'Found only the following: {self.train_loop_obj.train_history.keys()}')

        val_metric_result = self.train_loop_obj.train_history[self.metric_name][-1]
        self.scheduler.step(val_metric_result)


class LambdaLRScheduler(GeneralLRScheduler):
    def __init__(self, lr_lambda_list, **kwargs):
        """

        Args:
            lr_lambda_list (list):
            **kwargs:
        """
        GeneralLRScheduler.__init__(self, LambdaLR, **dict(kwargs, lr_lambda=lr_lambda_list))
        self.callback_name = ''


class StepLRScheduler(GeneralLRScheduler):
    def __init__(self, step_size, **kwargs):
        """

        Args:
            step_size (int):
            **kwargs:
        """
        GeneralLRScheduler.__init__(self, StepLR, **dict(kwargs, step_size=step_size))
        self.callback_name = ''


class MultiStepLRScheduler(GeneralLRScheduler):
    def __init__(self, milestones_list, **kwargs):
        """

        Args:
            milestones_list (list):
            **kwargs:
        """
        GeneralLRScheduler.__init__(self, MultiStepLR, **dict(kwargs, milestones=milestones_list))
        self.callback_name = ''
