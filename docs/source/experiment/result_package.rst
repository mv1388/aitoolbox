Result Package
==============

Result Package found in :mod:`aitoolbox.experiment.result_package` defines a set of evaluation metrics that are
used for the performance evaluation of the model on a certain ML task. For example, in the simple classification task,
the corresponding result package would include metrics such as *accuracy*, *F1 score*, *ROC-AUC* and *PR-AUC*.
Result packages can thus be thought of as wrappers around a set of evaluation metrics commonly used for different
ML tasks.

The same as for all other components of :mod:`aitoolbox.experiment` module, when it comes to the usage of result
packages, they can be either used in a standalone manually executed fashion for any kind of ML experiment evaluation.
On the other hand, result packages can also be used in unison with the TrainLoop model training engine from
the :mod:`aitoolbox.torchtrain`. There, the result package assumes the role of the *evaluation recipe* for a certain ML
task. By providing the result package to the TrainLoop the user informs it how to automatically evaluate
the model performance during or at the end of the training process.


Using Result Packages
---------------------

Result Package implementations can be found in the :mod:`aitoolbox.experiment.result_package`. AIToolbox already comes
with result packages for various popular ML tasks included out of the box. These can be found in the
:mod:`aitoolbox.experiment.result_package.basic_packages`.


Result Package with torchtrain TrainLoop
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When using result packages as part of TrainLoop supported training there are two main use-cases: as part
of the :class:`aitoolbox.torchtrain.callbacks.performance_eval.ModelPerformanceEvaluation` callback which optionally
performs the model performance evaluation during the training, e.g. after each epoch, and on the other hand as part of
the *"EndSave"* TrainLoop which automatically evaluates model's performance based on the provided result package at
the end of the training.

.. code-block:: python

    from aitoolbox.torchtrain.train_loop import *
    from aitoolbox.experiment.result_package.basic_packages import ClassificationResultPackage
    from aitoolbox.torchtrain.callbacks.performance_eval import \
        ModelPerformanceEvaluation, ModelPerformancePrintReport

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

    callbacks = [ModelPerformanceEvaluation(ClassificationResultPackage(), hyperparams,
                                            on_train_data=True, on_val_data=True),
                 ModelPerformancePrintReport(['train_Accuracy', 'val_Accuracy'])]

    tl = TrainLoopCheckpointEndSave(
        model,
        train_loader, val_loader, test_loader,
        optimizer, criterion,
        project_name='train_loop_examples',
        experiment_name='result_package_with_trainloop_example',
        local_model_result_folder_path='results_dir',
        hyperparams=hyperparams,
        val_result_package=ClassificationResultPackage(),
        test_result_package=ClassificationResultPackage()
    )

    model = tl.fit(num_epochs=10)


Standalone Result Package Use
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As mentioned above, result packages are completely independent from TrainLoop engine and can thus also be used
for a standalone model performance evaluation, even when not dealing with PyTorch models

.. code-block:: python

    from aitoolbox.experiment.result_package.basic_packages import BinaryClassificationResultPackage

    y_true = ...  # ground truth labels
    y_predicted = ...  # predicted by the model

    result_pkg = BinaryClassificationResultPackage()
    result_pkg.prepare_result_package(y_true, y_predicted)

    # get the results dict with performance results of all the metrics in the result package
    performance_results = result_pkg.get_results()


Implementing New Result Packages
--------------------------------

Although AIToolbox already provides result packages for certain ML tasks sometimes the user wants do define a novel or
unsupported performance evaluation metrics to properly evaluate the ML task at hand. The creation of new result packages
in AIToolbox is supported and can be done very easily.
