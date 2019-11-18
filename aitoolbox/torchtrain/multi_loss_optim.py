try:
    from apex import amp
    APEX_AVAILABLE = True
except ImportError:
    APEX_AVAILABLE = False


class MultiLoss:
    def __init__(self, loss_list, amp_optimizer_order=None):
        """Multiple loss wrapper for TrainLoop based training

        Args:
            loss_list (list): list of loss objects which are used to calculate losses in the TrainLoop
            amp_optimizer_order (callable or lambda or function or None): when using Nvidia Apex AMP, construct
                a list of optimizers where the optimizer's position matches the position of the corresponding loss
                in the loss_list. This is useful in cases similar to this:
                https://github.com/NVIDIA/apex/tree/master/examples/dcgan#mixed-precision-dcgan-training-in-pytorch

                When not using Nvidia Apex AMP this can be ignored as losses and optimizers don't need to be
                explicitly paired.
        """
        self.loss_list = loss_list
        self.amp_optimizer_order = amp_optimizer_order

        if self.amp_optimizer_order is not None:
            if not callable(self.amp_optimizer_order):
                raise TypeError('Provided amp_optimizer_order is not callable. When providing amp_optimizer_order '
                                'it should be a callable function taking the optimizer list and returning '
                                'the ordered list of loss-matching optimizers.')

    def backward(self):
        for loss in self.loss_list:
            loss.backward()

    def backward_amp(self, optimizers):
        """Executes backward() over all the losses using the list o optimizers

        Args:
            optimizers (list): list of optimizers all used optimizers

        Returns:
            None
        """
        optimizers_list = optimizers if self.amp_optimizer_order is None else self.amp_optimizer_order(optimizers)

        for i, (loss, optimizer) in enumerate(zip(self.loss_list, optimizers_list)):
            with amp.scale_loss(loss, optimizer, loss_id=i) as scaled_loss:
                scaled_loss.backward()

    def item(self):
        return [loss.item() for loss in self.loss_list]


class MultiOptimizer:
    def __init__(self, optimizer_list):
        """Multiple optimizer wrapper for TrainLoop based training

        Args:
            optimizer_list (list): list of optimizer objects which are used in the TrainLoop
        """
        self.optimizer_list = optimizer_list

    def zero_grad(self):
        for optimizer in self.optimizer_list:
            optimizer.zero_grad()

    def step(self):
        for optimizer in self.optimizer_list:
            optimizer.step()

    def state_dict(self):
        return [optimizer.state_dict() for optimizer in self.optimizer_list]
