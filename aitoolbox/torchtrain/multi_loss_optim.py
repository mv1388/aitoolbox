
class MultiLoss:
    def __init__(self, loss_dict, loss_optimizer_map=None, retain_graph_until_last=True):
        """Multiple loss wrapper for TrainLoop based training

        Args:
            loss_dict (dict): dict of loss objects which are used to calculate losses in the TrainLoop
            loss_optimizer_map (dict or None): dict mapping the loss name to the corresponding optimizer's index
                in the ``MultiOptimizer``. If this parameter is left to ``None`` the mapping is automatically created
                by assigning values from ``range(len(loss_dict))`` as corresponding optimizer indices.
            retain_graph_until_last (bool): when calling backward should ``retain_graph`` option be enabled for all but
                last loss tensor
        """
        self.loss_dict = loss_dict

        self.loss_backward_remaining = len(self.loss_dict)
        self.retain_graph_until_last = retain_graph_until_last

        if loss_optimizer_map is None:
            self.optimizer_loss_map = {i: k for i, k in enumerate(loss_dict.keys())}
        else:
            if len(loss_optimizer_map) != len(self.loss_dict):
                raise ValueError('loss_optimizer_map length not the same as loss_dict')

            self.optimizer_loss_map = {int(v): str(k) for k, v in loss_optimizer_map.items()}

    def backward(self, optimizer_idx, iteration, amp_grad_scaler):
        """Executes backward() for the specific loss based on provided optimizer_idx

        Args:
            optimizer_idx (int): index of the current optimizer. Mostly useful when using multiple optimizers. When
                only a single optimizer is used this parameter can be ignored.
            iteration (int): Current iteration index. Not used in the most simple setup but provided in case of more
                elaborate loss backward logic is devised.
            amp_grad_scaler (torch.cuda.amp.GradScaler): AMP GradScaler. If scaler ``enabled`` parameter is set to False
                the loss is still passed to it, but it gets returned unscaled so the behaviour is as it is in the case
                of non-AMP training.

        Returns:
            None
        """
        loss = self.loss_dict[self.optimizer_loss_map[optimizer_idx]]

        # Always pass the loss through the scaler
        # Depending on the `enabled` parameter of the scaler
        # the loss gets scaled or just returned unchanged
        loss = amp_grad_scaler.scale(loss)

        if self.retain_graph_until_last and self.loss_backward_remaining > 1:
            loss.backward(retain_graph=True)
        else:
            loss.backward()

        self.loss_backward_remaining -= 1

    def item(self):
        return {k: loss.item() for k, loss in self.loss_dict.items()}

    def __truediv__(self, grad_accumulation):
        self.loss_dict = {k: loss / grad_accumulation for k, loss in self.loss_dict.items()}
        return self


class MultiOptimizer:
    def __init__(self, optimizer_list):
        """Multiple optimizer wrapper for TrainLoop based training

        Args:
            optimizer_list (list): list of optimizer objects which are used in the TrainLoop
        """
        self.optimizer_list = optimizer_list

    def step(self, optimizer_idx, iteration, amp_grad_scaler):
        """Execute step for optimizer at the specified index

        Args:
            optimizer_idx (int): index of the current optimizer. Mostly useful when using multiple optimizers. When
                only a single optimizer is used this parameter can be ignored.
            iteration (int): Current iteration index. Not used in the most simple setup but provided in case of more
                elaborate loss backward logic is devised.
            amp_grad_scaler (torch.cuda.amp.GradScaler): AMP GradScaler. If scaler ``enabled`` parameter is set to False
                the optimizer have it's normal step() method called without applying the AMP mandated unscaling
                beforehand. In this respect the behaviour will be the same as in the non-AMP training.

        Returns:
            None
        """
        amp_grad_scaler.step(self.optimizer_list[optimizer_idx])

    def zero_grad(self, optimizer_idx, iteration):
        """Execute zero_grad for optimizer at the specified index

        Args:
            optimizer_idx (int): index of the current optimizer. Mostly useful when using multiple optimizers. When
                only a single optimizer is used this parameter can be ignored.
            iteration (int): Current iteration index. Not used in the most simple setup but provided in case of more
                elaborate loss backward logic is devised.

        Returns:
            None
        """
        self.optimizer_list[optimizer_idx].zero_grad()

    def state_dict(self):
        return [optimizer.state_dict() for optimizer in self.optimizer_list]

    def load_state_dict(self, state_dict_list):
        if not isinstance(state_dict_list, list):
            raise TypeError("state_dict_list is expected to be a list type.")
        if len(state_dict_list) != len(self.optimizer_list):
            raise ValueError("Provided len(state_dict_list) != len(self.optimizer_list).")

        for state_dict, optimizer in zip(state_dict_list, self.optimizer_list):
            optimizer.load_state_dict(state_dict)

    def __len__(self):
        return len(self.optimizer_list)

    def __getitem__(self, idx):
        return self.optimizer_list[idx]
