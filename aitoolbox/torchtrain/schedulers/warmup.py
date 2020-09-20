import math

from aitoolbox.torchtrain.schedulers.basic import LambdaLRScheduler


class ConstantWithWarmupScheduler(LambdaLRScheduler):
    def __init__(self, num_warmup_steps, last_epoch=-1, **kwargs):
        """Constant scheduler with the initial warmup

        Schedule with a constant learning rate preceded by a warmup period during which the learning rate increases
        linearly between 0 and the initial lr set in the optimizer.

        https://huggingface.co/transformers/main_classes/optimizer_schedules.html#transformers.get_constant_schedule_with_warmup

        Args:
            num_warmup_steps (int): The number of steps for the warmup phase
            last_epoch (int): The index of the last epoch when resuming training
            **kwargs: learning rate scheduler additional parameters
        """
        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1.0, num_warmup_steps))
            return 1.0

        super().__init__(lr_lambda=lr_lambda,
                         execute_epoch_end=False, execute_batch_end=True, last_epoch=last_epoch, **kwargs)
        self.callback_name = 'Warmed up constant learning rate scheduler'


class CosineWithWarmupScheduler(LambdaLRScheduler):
    def __init__(self, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1, **kwargs):
        """Cosine decreasing scheduler with the initial warmup

        Schedule with a learning rate that decreases following the values of the cosine function between the initial
        lr set in the optimizer to 0, after a warmup period during which it increases linearly
        between 0 and the initial lr set in the optimizer.

        https://huggingface.co/transformers/main_classes/optimizer_schedules.html#transformers.get_cosine_schedule_with_warmup

        Args:
            num_warmup_steps (int): The number of steps for the warmup phase
            num_training_steps (int): The total number of training steps
            num_cycles (float): The number of waves in the cosine schedule (the defaults is to just decrease
                from the max value to 0 following a half-cosine).
            last_epoch (int): The index of the last epoch when resuming training
            **kwargs: learning rate scheduler additional parameters
        """
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

        super().__init__(lr_lambda=lr_lambda,
                         execute_epoch_end=False, execute_batch_end=True, last_epoch=last_epoch, **kwargs)
        self.callback_name = 'Warmed up cosine decreasing learning rate scheduler'


class HardRestartsCosineWithWarmupScheduler(LambdaLRScheduler):
    def __init__(self, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1, **kwargs):
        """Cosine scheduler with hard restarts and the initial warmup

        Schedule with a learning rate that decreases following the values of the cosine function between
        the initial lr set in the optimizer to 0, with several hard restarts, after a warmup period during which
        it increases linearly between 0 and the initial lr set in the optimizer.

        https://huggingface.co/transformers/main_classes/optimizer_schedules.html#transformers.get_cosine_with_hard_restarts_schedule_with_warmup

        Args:
            num_warmup_steps (int): The number of steps for the warmup phase
            num_training_steps (int): The total number of training steps
            num_cycles (float): The number of waves in the cosine schedule (the defaults is to just decrease
                from the max value to 0 following a half-cosine).
            last_epoch (int): The index of the last epoch when resuming training
            **kwargs: learning rate scheduler additional parameters
        """
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            if progress >= 1.0:
                return 0.0
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * ((float(num_cycles) * progress) % 1.0))))

        super().__init__(lr_lambda=lr_lambda,
                         execute_epoch_end=False, execute_batch_end=True, last_epoch=last_epoch, **kwargs)
        self.callback_name = 'Warmed up hard restarts cosine learning rate scheduler'


class LinearWithWarmupScheduler(LambdaLRScheduler):
    def __init__(self, num_warmup_steps, num_training_steps, last_epoch=-1, **kwargs):
        """Linearly decreasing scheduler with the initial warmup

        Schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0,
        after a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.

        Especially useful in the context of BERT-like models.
        Implementation based on HuggingFace Transformers library's ``get_linear_schedule_with_warmup()`` method.

        https://huggingface.co/transformers/main_classes/optimizer_schedules.html#transformers.get_linear_schedule_with_warmup

        Args:
            num_warmup_steps (int): The number of steps for the warmup phase
            num_training_steps (int): The total number of training steps
            last_epoch (int): The index of the last epoch when resuming training
            **kwargs: learning rate scheduler additional parameters
        """
        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(
                0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
            )

        super().__init__(lr_lambda=lr_lambda,
                         execute_epoch_end=False, execute_batch_end=True, last_epoch=last_epoch, **kwargs)
        self.callback_name = 'Warmed up linearly decreasing learning rate scheduler'
