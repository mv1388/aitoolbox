Multi-GPU Training
==================

All TrainLoop versions in addition to single GPU also support multi-GPU training to achieve even faster training.
Following the core *PyTorch* setup, two multi-GPU training approaches are available:

* ``DataParallel`` done via :class:`aitoolbox.torchtrain.parallel.TTDataParallel`
* ``DistributedDataParallel`` done via :class:`aitoolbox.torchtrain.parallel.TTDistributedDataParallel`


TTDataParallel
--------------

To use ``DataParallel``-like multiGPU training with TrainLoop just switch the TrainLoop's ``gpu_mode`` parameter to
``'dp'``:

.. code-block:: python

    from aitoolbox.torchtrain.train_loop import *
    from aitoolbox.torchtrain.parallel import TTDataParallel


    model = CNNModel()  # TTModel based neural model

    train_loader = DataLoader(...)
    val_loader = DataLoader(...)
    test_loader = DataLoader(...)

    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    criterion = nn.NLLLoss()

    tl = TrainLoop(model,
                   train_loader, val_loader, test_loader,
                   optimizer, criterion,
                   gpu_mode='dp')

    model = tl.fit(num_epochs=10)


Check out a full
`DataParallel training example <https://github.com/mv1388/aitoolbox/blob/master/examples/dp_ddp_training/dp_training.py#L76>`_.


TTDistributedDataParallel
-------------------------

Distributed training on multiple GPUs via ``DistributedDataParallel`` is enabled by the TrainLoop itself under the hood
by wrapping the :doc:`model`-based model into :class:`aitoolbox.torchtrain.parallel.TTDistributedDataParallel`.
TrainLoop also automatically spawns multiple processes and initializes them. Inside each spawned process the model and
all other necessary training components are moved to the correct GPU belonging to a specific process.
Lastly, TrainLoop also automatically adds the *PyTorch* ``DistributedSampler`` to each of the provided data loaders
in order to ensure different data batches go to different GPUs and there is no overlap.

To enable distributed training via DistributedDataParallel, the user has to set the TrainLoop's ``gpu_mode``
parameter to ``'ddp'``.

.. code-block:: python

    from aitoolbox.torchtrain.train_loop import *


    model = CNNModel()  # TTModel based neural model

    train_loader = DataLoader(...)
    val_loader = DataLoader(...)
    test_loader = DataLoader(...)

    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    criterion = nn.NLLLoss()

    tl = TrainLoop(
        model,
        train_loader, val_loader, test_loader,
        optimizer, criterion,
        gpu_mode='ddp'
    )

    model = tl.fit(num_epochs=10,
                   num_nodes=1, node_rank=0, num_gpus=torch.cuda.device_count())


Check out a full
`DistributedDataParallel training example <https://github.com/mv1388/aitoolbox/blob/master/examples/dp_ddp_training/ddp_training.py#L81>`_.
