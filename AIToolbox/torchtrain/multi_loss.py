
class MultiLoss:
    def __init__(self, loss_list):
        """

        Args:
            loss_list (list):
        """
        self.loss_list = loss_list

    def backward(self):
        for loss in self.loss_list:
            loss.backward()

    def item(self):
        return [loss.item() for loss in self.loss_list]


class MultiOptimizer:
    def __init__(self, optimizer_list):
        """

        Args:
            optimizer_list (list):
        """
        self.optimizer_list = optimizer_list

    def zero_grad(self):
        for optimizer in self.optimizer_list:
            optimizer.zero_grad()

    def step(self):
        for optimizer in self.optimizer_list:
            optimizer.step()
