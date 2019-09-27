import numpy as np
import torch

from AIToolbox.torchtrain.callbacks.callbacks import AbstractCallback


class GradNormClipCallback(AbstractCallback):
    def __init__(self, max_norm, **kwargs):
        """Gradient norm clipping

        Args:
            max_norm (int or float): gradient clipping
            **kwargs:
        """
        AbstractCallback.__init__(self, 'Gradient clipping')
        self.max_norm = max_norm
        self.kwargs = kwargs

    def on_train_loop_registration(self):
        self.train_loop_obj.grad_cb_used = True

    def on_after_gradient_update(self):
        torch.nn.utils.clip_grad_norm_(self.train_loop_obj.model.parameters(), self.max_norm, **self.kwargs)


class GradientStatsPrintCallback(AbstractCallback):
    def __init__(self, model_layers_extract_def, on_every_grad_update=False):
        """

        Args:
            model_layers_extract_def:
            on_every_grad_update:
        """
        AbstractCallback.__init__(self, 'Print model gradient stats')
        self.model_layers_extract_def = model_layers_extract_def
        self.on_every_grad_update = on_every_grad_update

    def on_train_loop_registration(self):
        if self.on_every_grad_update:
            self.train_loop_obj.grad_cb_used = True

    def on_after_gradient_update(self):
        if self.on_every_grad_update:
            self.gradients_report()

    def on_epoch_end(self):
        self.gradients_report()

    def gradients_report(self):
        model_layers_list = self.model_layers_extract_def(self.train_loop_obj.model)

        print('---> Model layers gradients stats')
        for i, layer in enumerate(model_layers_list):
            gradients = layer.weight.grad.cpu().numpy()
            mu = np.mean(gradients)
            std = np.std(gradients)

            print(f'Layer {i} grads: Mean: {mu}; Std {std}')
            print(f'\tRatio of zero gradients: {float(np.count_nonzero(gradients == 0)) / gradients.size}')
