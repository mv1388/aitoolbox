import torch

from AIToolbox.torchtrain.callbacks.callbacks import AbstractCallback


class GradientClipCallback(AbstractCallback):
    def __init__(self, grad_clip):
        """

        Args:
            grad_clip (int or float): optional gradient clipping
        """
        AbstractCallback.__init__(self, 'Gradient clipping')
        self.grad_clip = grad_clip

    def on_gradient_update(self):
        torch.nn.utils.clip_grad_norm_(self.train_loop_obj.model.parameters(), self.grad_clip)
