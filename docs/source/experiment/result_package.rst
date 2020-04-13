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


.. _implementing-new-result-pkgs:

Implementing New Result Packages
--------------------------------

Although AIToolbox already provides result packages for certain ML tasks sometimes the user wants do define a novel or
unsupported performance evaluation metrics to properly evaluate the ML task at hand. The creation of new result packages
in AIToolbox is supported and can be done very easily.

The new result package can be implemented as a new class which is inheriting from the base abstract result package
:class:`aitoolbox.experiment.result_package.abstract_result_packages.AbstractResultPackage` and implements
the abstract method :meth:`aitoolbox.experiment.result_package.abstract_result_packages.AbstractResultPackage.prepare_results_dict`.

Inside the ``prepare_results_dict()`` the user needs to implement the logic to evaluate the performance on desired
performance metrics forming the result package. In order to perform the evaluation the predicted and ground truth values
are normally needed. These are inserted into the package at run time (via ``prepare_result_package()``) and
exposed inside the result package via: ``self.y_true`` and ``self.y_predicted`` attributes. Logic inside the which
the user needs to define, ``prepare_results_dict()`` should access the values in *y_true* and *y_predicted*,
pass them through the desired performance metrics computations and return the results in the dict form.
Inside the returned dict, keys should represent the evaluated metric names and values the corresponding
evaluated performance metric values.

The performance metric computation as part of the result package can be directly implemented inside the result package class
in the ``prepare_results_dict()`` method. However, especially in the case of more complex performance metric logic
in order to ensure better reusability of the implemented metrics as well as more readable and structured code of
the developed result packages it is common practice in the AIToolbox to implement performance metrics as a separate
specialized metric class. This way the result packages become a lightweight wrappers around the selected performance
metrics while the actual performance metric logic and calculation is done as part of the metric object instead of
being done in the encapsulating result package. To learn more about the AIToolbox performance metric use and
implementations have a look at the :doc:`metrics` documentation section.


Example or Result Package using AIToolbox Result Metric
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from aitoolbox.experiment.result_package.abstract_result_packages import AbstractResultPackage
    from aitoolbox.experiment.core_metrics.classification import \
        AccuracyMetric, ROCAUCMetric, PrecisionRecallCurveAUCMetric


    class ExampleClassificationResultPackage(AbstractResultPackage):
        def __init__(self):
            AbstractResultPackage.__init__(self, pkg_name='ExampleClassificationResult')

        def prepare_results_dict(self):
            accuracy_result = AccuracyMetric(self.y_true, self.y_predicted)
            roc_auc_result = ROCAUCMetric(self.y_true, self.y_predicted)
            pr_auc_result = PrecisionRecallCurveAUCMetric(self.y_true, self.y_predicted)

            return accuracy_result + roc_auc_result + pr_auc_result


Example of Result Package with Direct Performance Metric Calculation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from aitoolbox.experiment.result_package.abstract_result_packages import AbstractResultPackage
    from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve, auc


    class ExampleClassificationResultPackage(AbstractResultPackage):
        def __init__(self):
            AbstractResultPackage.__init__(self, pkg_name='ExampleClassificationResult')

        def prepare_results_dict(self):
            accuracy = accuracy_score(self.y_true, self.y_predicted)
            roc_auc = roc_auc_score(self.y_true, self.y_predicted)

            precision, recall, thresholds = precision_recall_curve(self.y_true, self.y_predicted)
            pr_auc = auc(recall, precision)

            return {'accuracy': accuracy, 'roc_auc': roc_auc, 'pr_auc': pr_auc}
