Automatic Mixed Precision Training
==================================

All the TrainLoop versions also support training with Automatic Mixed Precision (*AMP*). In the past this required
using the `Nvidia apex <https://github.com/NVIDIA/apex>`_ extension but from *PyTorch 1.6* onwards AMP functionality
is built into core PyTorch and no separate instalation is needed.
Current version of AIToolbox already supports the use of built-in PyTorch AMP.

The user only has to set the TrainLoop parameter ``use_amp`` to ``use_amp=True`` in order to use the default
AMP initialization and start training the model in the mixed precision mode. If the user wants to specify
custom AMP ``GradScaler`` initialization parameters, these should be provided as a dict parameter
``use_amp={'init_scale': 2.**16, 'growth_factor': 2.0, ...}`` to the TrainLoop.
All AMP initializations and training related steps are then handled automatically by the TrainLoop.

You can read more about different AMP details in the
`PyTorch AMP documentation <https://pytorch.org/docs/stable/notes/amp_examples.html>`_.


Single-GPU mixed precision training
-----------------------------------

Example of single-GPU AMP setup:

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

    model = tl.fit(num_epochs=10)


Check out a full
`AMP single-GPU training example
<https://github.com/mv1388/aitoolbox/blob/master/examples/amp_training/single_GPU_training.py>`_.


Multi-GPU DDP mixed precision training
--------------------------------------

When training in the multi-GPU setting, the setup is mostly the same as in the single-GPU.
All the user has to do is set accordingly the ``use_amp`` parameter of the TrainLoop and to switch its ``gpu_mode``
parameter to ``'ddp'``.
Under the hood, TrainLoop will initialize the model and the optimizer for AMP and start training using
DistributedDataParallel approach.

Example of multi-GPU AMP setup:

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
                   use_amp=True)

    model = tl.fit(num_epochs=10,
                   num_nodes=1, node_rank=0, num_gpus=torch.cuda.device_count())


Check out a full
`AMP multi-GPU DistributedDataParallel training example
<https://github.com/mv1388/aitoolbox/blob/master/examples/amp_training/mutli_GPU_training.py>`_.
