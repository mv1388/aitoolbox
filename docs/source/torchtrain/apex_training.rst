APEX Mixed Precision Training
=============================

All the TrainLoop versions also support training with **Automatic Mixed Precision** (*AMP*) using
the `Nvidia apex <https://github.com/NVIDIA/apex>`_ extension. To use this feature the user first has to install
the Nvidia apex library (`installation instructions <https://github.com/NVIDIA/apex#linux>`_).


Single-GPU mixed precision training
-----------------------------------

The user only has to properly amp initialize the model and optimizer and finally set the TrainLoop parameter to
``use_amp=True``. All other training related steps are handled automatically by the TrainLoop.
Example of initialization is shown bellow and more can be read in the
`official Nvidia apex documentation <https://nvidia.github.io/apex/amp.html#opt-levels-and-properties>`_.

.. code-block:: python

    from apex import amp
    from aitoolbox.torchtrain.train_loop import *


    train_loader = DataLoader(...)
    val_loader = DataLoader(...)
    test_loader = DataLoader(...)

    model = CNNModel()  # TTModel based neural model
    model = model.to('cuda')  # model has to be moved to the GPU before amp.initialize

    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    criterion = nn.NLLLoss().to('cuda')

    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    tl = TrainLoop(model,
                   train_loader, val_loader, test_loader,
                   optimizer, criterion,
                   use_amp=True)

    model = tl.fit(num_epochs=10)


Check out a full
`Apex AMP training example
<https://github.com/mv1388/aitoolbox/blob/master/examples/apex_amp_training/apex_single_GPU_training.py#L83>`_.


Multi-GPU DDP mixed precision training
--------------------------------------

When training with automatic mixed precision in the multi-GPU setup TrainLoop automatically handles most of
the AMP initialization. All the user has to do is call :meth:`aitoolbox.torchtrain.train_loop.TrainLoop.fit_distributed`
and provide the AMP initialization parameters as a dict argument ``amp_init_args``.
Under the hood, TrainLoop will initialize model and optimizer for AMP and start training using DistributedDataParallel
approach (DDP is currently only multi-GPU training setup supported by Apex AMP).

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
                   use_amp=True)

    model = tl.fit_distributed(num_epochs=10,
                               amp_init_args={'opt_level': 'O1'},
                               num_nodes=1, node_rank=0, num_gpus=torch.cuda.device_count())


Check out a full
`Apex AMP DistributedDataParallel training example
<https://github.com/mv1388/aitoolbox/blob/master/examples/apex_amp_training/apex_mutli_GPU_training.py#L86>`_.
