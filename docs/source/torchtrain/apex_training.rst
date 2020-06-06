APEX Mixed Precision Training
=============================

All the TrainLoop versions also support training with **Automatic Mixed Precision** (*AMP*) using
the `Nvidia apex <https://github.com/NVIDIA/apex>`_ extension. To use this feature the user first has to install
the Nvidia apex library (`installation instructions <https://github.com/NVIDIA/apex#linux>`_).

The user only has to set the TrainLoop parameter ``use_amp`` to ``use_amp=True`` in order to use the default
AMP initialization and start training the model in the mixed precision mode. If the user wants to specify custom
AMP initialization parameters, these should be provided as a dict parameter ``use_amp={'opt_level': 'O1'}`` to
the TrainLoop. All AMP initializations and training related steps are then handled automatically by the TrainLoop.

You can read more about different AMP optimization levels in the
`official Nvidia apex documentation <https://nvidia.github.io/apex/amp.html#opt-levels-and-properties>`_.


Single-GPU mixed precision training
-----------------------------------

Example of single-GPU APEX setup:

.. code-block:: python

    from aitoolbox.torchtrain.train_loop import *


    train_loader = DataLoader(...)
    val_loader = DataLoader(...)
    test_loader = DataLoader(...)

    model = CNNModel()  # TTModel based neural model

    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    criterion = nn.NLLLoss()

    tl = TrainLoop(model,
                   train_loader, val_loader, test_loader,
                   optimizer, criterion,
                   use_amp={'opt_level': 'O1'})

    model = tl.fit(num_epochs=10)


Check out a full
`Apex AMP training example
<https://github.com/mv1388/aitoolbox/blob/master/examples/apex_amp_training/apex_single_GPU_training.py#L83>`_.


Multi-GPU DDP mixed precision training
--------------------------------------

When training in the multi-GPU setting, the setup is mostly the same as in the single-GPU.
All the user has to do is set accordingly the ``use_amp`` parameter of the TrainLoop and to switch its ``gpu_mode``
parameter to ``'ddp'``.
Under the hood, TrainLoop will initialize the model and the optimizer for AMP and start training using
DistributedDataParallel approach (DDP is currently only multi-GPU training setup supported by Apex AMP).

.. code-block:: python

    from aitoolbox.torchtrain.train_loop import *


    train_loader = DataLoader(...)
    val_loader = DataLoader(...)
    test_loader = DataLoader(...)

    model = CNNModel()  # TTModel based neural model

    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    criterion = nn.NLLLoss()

    tl = TrainLoop(model,
                   train_loader, val_loader, test_loader,
                   optimizer, criterion,
                   gpu_mode='ddp',
                   use_amp={'opt_level': 'O1'})

    model = tl.fit(num_epochs=10,
                   num_nodes=1, node_rank=0, num_gpus=torch.cuda.device_count())


Check out a full
`Apex AMP DistributedDataParallel training example
<https://github.com/mv1388/aitoolbox/blob/master/examples/apex_amp_training/apex_mutli_GPU_training.py#L86>`_.
