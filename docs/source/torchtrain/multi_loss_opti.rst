Multi-Loss and Multi-Optimizer
==============================

TrainLoop supports training using multiple separate losses and/or multiple different
optimizers at the same time.

The multi loss/optimizer functionality is achieved by wrapping multiple loss or
optimizer objects into the ``MultiLoss`` and ``MultiOptimizer`` wrappers respectively
provided in :mod:`aitoolbox.torchtrain.multi_loss_optim`.


Multi-Loss Training
-------------------

To implement training with multiple losses use :class:`aitoolbox.torchtrain.multi_loss_optim.MultiLoss`
to wrap different calculated losses together and return them from model's ``get_loss()`` function.
Train loop will then automatically know to correctly execute backprop through each of the losses.

Multiple losses need to be provided to the MultiLoss as a dict:

.. code-block:: python

    MultiLoss({'main_loss': main_loss, 'aux_loss': aux_loss})


In case of more elaborate backprop logic is needed one can override MultiLoss'
:meth:`aitoolbox.torchtrain.multi_loss_optim.MultiLoss.backward` method with the desired advanced logic.


Multi-Optimizer Training
------------------------

To use multiple optimizers, for example each one optimizing a different part of the model, define multiple
optimizers each with access to different parameters of the model. These separate optimizers need to be provided
in a list to the :class:`aitoolbox.torchtrain.multi_loss_optim.MultiOptimizer` wrapper.
The ``MultiOptimizer`` can subsequently be given to the TrainLoop the same way as the normal single optimizer.

MultiOptimizer definition example:

.. code-block:: python

    MultiOptimizer([optimizer_1, optimizer_2])