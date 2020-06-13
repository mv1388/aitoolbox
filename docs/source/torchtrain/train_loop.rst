Train Loop
==========

``TrainLoop`` and it's module :mod:`aitoolbox.torchtrain.train_loop` is at the core of and probably most important
component of the entire *AIToolbox* package.

Common to all available TrainLoops is the *PyTorch* model training loop engine which automatically handles the
deep learning training process. As part of this it does the batch feeding of data into the model, calculating loss
and updating parameters for a specified number of epochs.

``torchtrain`` and by extension ``TrainLoop``
has been designed with the ease of use in mind. One of the main design principles was to keep as much
training code as possible exactly the same as would be used in normally *PyTorch*. Consequently, the user can define
the dataset, dataloader and models in exactly the same way as it would be done when training directly with core *PyTorch*.
Having no need to modify the definitions of common *PyTorch* training components in order to use torchtrain makes it
very user-friendly and allows the user to apply torchtrain directly to projects which initially weren't even coded with
AIToolbox in mind.

To train the model, all the user has to do is provide the TrainLoop with the model, train / validation / test dataloaders,
loss function and the optimizer. That's it.

Once the TrainLoop with all the necessary components has been created all that's left is to start training the model.
Common to all the available TrainLoops is the :meth:`aitoolbox.torchtrain.train_loop.TrainLoop.fit` method which
initiates the training process. The ``.fit()`` method will train the provided model on the given training dataset in
the training dataloader for the specified number of epochs.

.. note:: In order to use the :mod:`aitoolbox.torchtrain.train_loop` the user has to define their models as a
          :class:`aitoolbox.torchtrain.model.TTModel` which is a slightly modified AIToolbox specific variation of
          the core PyTorch :class:`torch.nn.Module`. Please have a look at the :doc:`model` section of the documentation
          in order to learn how to define your TTModels compatible with TrainLoop supported training.



TrainLoop Variations
--------------------

:mod:`aitoolbox.torchtrain.train_loop` module consists of four different ``TrainLoop`` variations:

* :class:`aitoolbox.torchtrain.train_loop.TrainLoop`
* :class:`aitoolbox.torchtrain.train_loop.TrainLoopCheckpoint`
* :class:`aitoolbox.torchtrain.train_loop.TrainLoopEndSave`
* :class:`aitoolbox.torchtrain.train_loop.TrainLoopCheckpointEndSave`

The above listed TrainLoop options can be distinguished based on the varying extent of the automatic experiment tracking
they do on top of the core training loop functionality. The available TrainLoops follow this naming convention:

* name includes ``Checkpoint`` keyword: the TrainLoop will automatically save the model after each training epoch
* name includes ``EndSave`` keyword: the TrainLoop will automatically evaluate final model performance and
  save the final model at the end of the training


TrainLoop
^^^^^^^^^
The simplest TrainLoop version which only performs the model training and does no experiment tracking and
performance evaluation.

The API can be found in: :class:`aitoolbox.torchtrain.train_loop.TrainLoop`.

Example of the ``TrainLoop`` used to train the model:

.. code-block:: python

    from aitoolbox.torchtrain.train_loop import *


    model = CNNModel()  # TTModel based neural model
    train_loader = DataLoader(...)
    val_loader = DataLoader(...)
    test_loader = None

    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    criterion = nn.NLLLoss()

    tl = TrainLoop(model,
                   train_loader, val_loader, test_loader,
                   optimizer, criterion)

    model = tl.fit(num_epochs=10)


TrainLoopCheckpoint
^^^^^^^^^^^^^^^^^^^
Same training process as in TrainLoop with additional automatic model checkpointing (saving) after every epoch. Model
saving can be done only to the local disk or also to the cloud storage such as AWS S3.

The API can be found in: :class:`aitoolbox.torchtrain.train_loop.TrainLoopCheckpoint`.

.. code-block:: python

    from aitoolbox.torchtrain.train_loop import *
    from aitoolbox.experiment.result_package.basic_packages import ClassificationResultPackage


    hyperparams = {
        'lr': 0.001,
        'betas': (0.9, 0.999)
    }

    model = CNNModel()  # TTModel based neural model
    train_loader = DataLoader(...)
    val_loader = DataLoader(...)
    test_loader = DataLoader(...)

    optimizer = optim.Adam(model.parameters(), lr=hyperparams['lr'], betas=hyperparams['betas'])
    criterion = nn.NLLLoss()

    tl = TrainLoopCheckpoint(
        model,
        train_loader, val_loader, test_loader,
        optimizer, criterion,
        project_name='train_loop_examples', experiment_name='TrainLoopCheckpoint_example',
        local_model_result_folder_path='results_dir',
        hyperparams=hyperparams,
        cloud_save_mode='s3', bucket_name='cloud_results'  # bucket_name should be set to the bucket on your S3
    )

    model = tl.fit(num_epochs=10)



TrainLoopEndSave
^^^^^^^^^^^^^^^^^^^
Same training process as in TrainLoop with additional automatic model checkpointing (saving) and model performance
evaluation at the end of the training process. This way the TrainLoop ensures experiment tracking a the end of
the training. Model and experiment results saving can be done only to the local disk or also to the cloud storage
such as AWS S3.

The API can be found in: :class:`aitoolbox.torchtrain.train_loop.TrainLoopEndSave`.

For information about the ``ResultPackage`` used in this example, have a look at the :doc:`../experiment/result_package`
section.

.. code-block:: python

    from aitoolbox.torchtrain.train_loop import *
    from aitoolbox.experiment.result_package.basic_packages import ClassificationResultPackage


    hyperparams = {
        'lr': 0.001,
        'betas': (0.9, 0.999)
    }

    model = CNNModel()  # TTModel based neural model
    train_loader = DataLoader(...)
    val_loader = DataLoader(...)
    test_loader = DataLoader(...)

    optimizer = optim.Adam(model.parameters(), lr=hyperparams['lr'], betas=hyperparams['betas'])
    criterion = nn.NLLLoss()

    tl = TrainLoopEndSave(
        model,
        train_loader, val_loader, test_loader,
        optimizer, criterion,
        project_name='train_loop_examples', experiment_name='TrainLoopEndSave_example',
        local_model_result_folder_path='results_dir',
        hyperparams=hyperparams,
        val_result_package=ClassificationResultPackage(),
        test_result_package=ClassificationResultPackage(),
        cloud_save_mode='s3', bucket_name='cloud_results'  # bucket_name should be set to the bucket on your S3
    )

    model = tl.fit(num_epochs=10)


TrainLoopCheckpointEndSave
^^^^^^^^^^^^^^^^^^^^^^^^^^
For the most complete experiment tracking it is recommended to use the this TrainLoop option.
At its core it is the same training process as in TrainLoop with additional automatic model checkpointing (saving) after
each epoch as well as automatic model checkpointing and model performance evaluation at the end of the training process.
This way the TrainLoop ensures full experiment tracking with the maximum extent. Model and experiment results saving
can be done only to the local disk or also to the cloud storage such as AWS S3.

The API can be found in: :class:`aitoolbox.torchtrain.train_loop.TrainLoopCheckpointEndSave`.

For information about the ``ResultPackage`` used in this example, have a look at the :doc:`../experiment/result_package`
section.

For a full working example of the ``TrainLoopCheckpointEndSave`` training, check out this
`TrainLoopCheckpointEndSave example training script
<https://github.com/mv1388/aitoolbox/blob/master/examples/TrainLoop_use/trainloop_fully_tracked_experiment.py>`_.

.. code-block:: python

    from aitoolbox.torchtrain.train_loop import *
    from aitoolbox.experiment.result_package.basic_packages import ClassificationResultPackage


    hyperparams = {
        'lr': 0.001,
        'betas': (0.9, 0.999)
    }

    model = CNNModel()  # TTModel based neural model
    train_loader = DataLoader(...)
    val_loader = DataLoader(...)
    test_loader = DataLoader(...)

    optimizer = optim.Adam(model.parameters(), lr=hyperparams['lr'], betas=hyperparams['betas'])
    criterion = nn.NLLLoss()

    tl = TrainLoopCheckpointEndSave(
        model,
        train_loader, val_loader, test_loader,
        optimizer, criterion,
        project_name='train_loop_examples', experiment_name='TrainLoopCheckpointEndSave_example',
        local_model_result_folder_path='results_dir',
        hyperparams=hyperparams,
        val_result_package=ClassificationResultPackage(),
        test_result_package=ClassificationResultPackage(),
        cloud_save_mode='s3', bucket_name='cloud_results'  # bucket_name should be set to the bucket on your S3
    )

    model = tl.fit(num_epochs=10)
