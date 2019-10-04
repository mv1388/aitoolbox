try:
    from apex import amp
    APEX_AVAILABLE = True
except ModuleNotFoundError:
    APEX_AVAILABLE = False
except AttributeError:
    APEX_AVAILABLE = False


class MultiLoss:
    def __init__(self, loss_list):
        """Multiple loss wrapper for TrainLoop based training

        Args:
            loss_list (list): list of loss objects which are used to calculate losses in the TrainLoop
        """
        self.loss_list = loss_list

    def backward(self):
        for loss in self.loss_list:
            loss.backward()

    def backward_amp(self, optimizers):
        """Executes backward() over all the losses using the list o optimizers

        Args:
            optimizers (list): list of optimizers. Positions in the list correspond to the ordering of the losses

        Returns:
            None
        """
        for i, (loss, optimizer) in enumerate(zip(self.loss_list, optimizers)):
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
