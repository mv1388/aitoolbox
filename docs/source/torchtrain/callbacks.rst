Callbacks
=========

For advanced model training experiments the basic logic offered in available TrainLoops might not be enough.
Additional needed logic can be injected into the training procedure by using *callbacks* and providing them as
a parameter list to :meth:`aitoolbox.torchtrain.train_loop.TrainLoop.fit` function found in all TrainLoops.


Available Callbacks
-------------------

AIToolbox by default already offers a wide selection of different useful callbacks which can be used to augment
the base training procedure. These out of the box callbacks can be found in :mod:`aitoolbox.torchtrain.callbacks`
module. There are several general categories of available callbacks:

* :mod:`aitoolbox.torchtrain.callbacks.basic` - general training augmentation
* :mod:`aitoolbox.torchtrain.callbacks.performance_eval` - model performance evaluation
* :mod:`aitoolbox.torchtrain.callbacks.model_save` - local / cloud based model saving
* :mod:`aitoolbox.torchtrain.callbacks.train_schedule` - learning rate schedulers
* :mod:`aitoolbox.torchtrain.callbacks.gradient` - model gradient reporting
* :mod:`aitoolbox.torchtrain.callbacks.model_load` - existing model loading at train start
* :mod:`aitoolbox.torchtrain.callbacks.tensorboard` - tensorboard training tracking

Example of the several basic callbacks used to infuse additional logic into the model training process:

.. code-block:: python

    from aitoolbox.torchtrain.train_loop import *
    from aitoolbox.torchtrain.callbacks.basic import EarlyStopping, TerminateOnNaN, AllPredictionsSame

    model = CNNModel()  # TTModel based neural model
    train_loader = DataLoader(...)
    val_loader = DataLoader(...)
    test_loader = DataLoader(...)

    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    criterion = nn.NLLLoss()

    callbacks = [
        EarlyStopping(patience=3),
        TerminateOnNaN(),
        AllPredictionsSame(value=0.)
    ]

    tl = TrainLoop(model,
                   train_loader, val_loader, test_loader,
                   optimizer, criterion)

    model = tl.fit(num_epochs=10, callbacks=callbacks)


For a full working example which shows the use of multiple callbacks of various types, check out this
`fully tracked training experiment example
<https://github.com/mv1388/aitoolbox/blob/master/examples/TrainLoop_use/trainloop_fully_tracked_experiment.py#L81>`_.


Developing New Callbacks
------------------------

However when some completely new functionality is desired which is not available out of the box in AIToolbox
the user can also implement their own custom callbacks. These can then be used as any other callback to further
extend the training loop process.

AbstractCallback
^^^^^^^^^^^^^^^^

The new callback can be implemented as a new class which is inheriting from the base callback
:class:`aitoolbox.torchtrain.callbacks.abstract.AbstractCallback`. All that the user has to do is to override and
implement the methods corresponding to positions in the TrainLoop training process at which the newly developed callback
should be executed. If a certain callback method is left unimplemented and thus left to the default from
the parent ``AbstractCallback`` the callback has no effect on the TrainLoop at the corresponding position in
the training process.

Callback execution is currently supported at the following positions in the TrainLoop via the following methods:

* :meth:`aitoolbox.torchtrain.callbacks.abstract.AbstractCallback.on_train_begin`
* :meth:`aitoolbox.torchtrain.callbacks.abstract.AbstractCallback.on_epoch_begin`
* :meth:`aitoolbox.torchtrain.callbacks.abstract.AbstractCallback.on_batch_begin`
* :meth:`aitoolbox.torchtrain.callbacks.abstract.AbstractCallback.on_after_gradient_update`
* :meth:`aitoolbox.torchtrain.callbacks.abstract.AbstractCallback.on_after_optimizer_step`
* :meth:`aitoolbox.torchtrain.callbacks.abstract.AbstractCallback.on_batch_end`
* :meth:`aitoolbox.torchtrain.callbacks.abstract.AbstractCallback.on_epoch_end`
* :meth:`aitoolbox.torchtrain.callbacks.abstract.AbstractCallback.on_train_end`
* :meth:`aitoolbox.torchtrain.callbacks.abstract.AbstractCallback.on_train_loop_registration`
* :meth:`aitoolbox.torchtrain.callbacks.abstract.AbstractCallback.on_multiprocess_start`

train_loop_obj
^^^^^^^^^^^^^^

The most usable and thus important aspect of every callback is its ability to communicate and modify the encapsulating
running TrainLoop. Every callback has a special attribute
:attr:`aitoolbox.torchtrain.callbacks.abstract.AbstractCallback.train_loop_obj` which at the start of the TrainLoop
training process gets assigned the reference (pointer) to the encapsulating TrainLoop object. In AIToolbox the process
is called *TrainLoop registration* and is automatically done under the hood by the TrainLoop by calling the
:meth:`aitoolbox.torchtrain.callbacks.abstract.AbstractCallback.register_train_loop_object`.

Via the ``train_loop_obj`` the callback can thus have a complete access to and control of every aspect of the TrainLoop.
While maybe dangerous for inexperienced users, this extensive low level control is especially welcome for the advanced
research use of AIToolbox. After the train loop object registration inside the callback the reference to
the encapsulating TrainLoop can be simply accessed from any implemented callback method via ``self.train_loop_obj``.

Custom Callback Example
^^^^^^^^^^^^^^^^^^^^^^^

Example of a newly developed callback and its use in the TrainLoop:

.. code-block:: python

    from aitoolbox.torchtrain.train_loop import *
    from aitoolbox.torchtrain.callbacks.abstract import AbstractCallback
    from aitoolbox.torchtrain.callbacks.basic import EarlyStopping, TerminateOnNaN, AllPredictionsSame


    class MyDemoTrainingReportCallback(AbstractCallback):
        def __init__(self):
            super().__init__('simple callback example')

        def on_train_begin(self):
            experiment_start_time = self.train_loop_obj.experiment_timestamp
            print(f'Starting the training! Experiment started at: {experiment_start_time}')

        def on_epoch_begin(self):
            current_epoch = self.train_loop_obj.epoch
            print(f'Starting new epoch num {current_epoch}')

        def on_epoch_end(self):
            val_predictions = self.train_loop_obj.predict_on_validation_set()
            print('Model predictions:')
            print(val_predictions)

        def on_train_end(self):
            print(f'End of training! Stopped at epoch {self.train_loop_obj.epoch}')

            test_predictions = self.train_loop_obj.predict_on_test_set()
            print('Model predictions:')
            print(test_predictions)


    model = CNNModel()  # TTModel based neural model
    train_loader = DataLoader(...)
    val_loader = DataLoader(...)
    test_loader = DataLoader(...)

    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    criterion = nn.NLLLoss()

    callbacks = [
        MyDemoTrainingReportCallback(),
        EarlyStopping(patience=3),
        TerminateOnNaN(),
        AllPredictionsSame(value=0.)
    ]

    tl = TrainLoop(model,
                   train_loader, val_loader, test_loader,
                   optimizer, criterion)

    model = tl.fit(num_epochs=10, callbacks=callbacks)


AbstractExperimentCallback
^^^^^^^^^^^^^^^^^^^^^^^^^^

In case of the developed callback is aimed at experiment tracking where information about the created experiment
details such as project name, experiment name and path of the local experiment folder would be needed there is
available also available the :class:`aitoolbox.torchtrain.callbacks.abstract.AbstractExperimentCallback`.
``AbstractExperimentCallback`` has all the same properties as basic ``AbstractCallback`` and is extended with
the convenience method
:meth:`aitoolbox.torchtrain.callbacks.abstract.AbstractExperimentCallback.try_infer_experiment_details` which extracts
the experiment details from the running ``TrainLoop`` and infuses our callback with this additional needed information.

For the example of the ``try_infer_experiment_details()`` use in practice check this implementation:
:meth:`aitoolbox.torchtrain.callbacks.performance_eval.ModelTrainHistoryPlot.on_train_loop_registration`.
