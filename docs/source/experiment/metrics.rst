Result Metric
=============

Result metric (:mod:`aitoolbox.experiment.core_metrics`) is an abstraction built around the calculation of the single
performance metric. It helps keep the code base more reusable and better structured, especially when used as part of
the encapsulating :doc:`result_package`.

AIToolbox comes out of the box with implemented several commonly used performance evaluation metrics implemented as
result metrics. These can be found in:

* :mod:`aitoolbox.experiment.core_metrics.classification`
* :mod:`aitoolbox.experiment.core_metrics.regression`


Use of Result Metrics inside Result Packages
--------------------------------------------

As it is described in the :ref:`implementing-new-result-pkgs` section, result metrics come in handy when developing
the result packages which are wrapping together multiple metrics needed to evaluate a certain ML task. To support this
chaining together of multiple performance metrics, the result metric abstraction offers a convenient metric
concatenation and result package dictionary creation via the ``+`` operator. To create the dictionary holding all
the performance metric results the user can simply write: ``metric_1 + metric_2 + metric_3 + ...``. This makes the use
of the ``+`` operator very convenient because the produced results dictionary format exactly matches that which is
required when developing an encapsulating result package.

Example of result metric concatenation:

.. code-block:: python

    from aitoolbox.experiment.core_metrics.classification import \
        AccuracyMetric, ROCAUCMetric, PrecisionRecallCurveAUCMetric

    accuracy_result = AccuracyMetric(y_true, y_predicted)
    roc_auc_result = ROCAUCMetric(y_true, y_predicted)
    pr_auc_result = PrecisionRecallCurveAUCMetric(y_true, y_predicted)

    results_dict =  accuracy_result + roc_auc_result + pr_auc_result

    # results_dict will hold:
    # {'Accuracy': 0.95, 'ROC_AUC': 0.88, 'PrecisionRecall_AUC': 0.67}


Implementing New Result Metrics
-------------------------------

When the needed result metric is not available in the AIToolbox, the users can easily implement
their own new metrics. The approach is very similar to that of the new result package development.

In order to implement
a new result metric, the user has to create a new metric class which inherits from the base abstract result metric
:class:`aitoolbox.experiment.core_metrics.abstract_metric.AbstractBaseMetric` and implements the abstract method
:meth:`aitoolbox.experiment.core_metrics.abstract_metric.AbstractBaseMetric.calculate_metric`.

As part of the ``calculate_metric()`` the user has to implement the logic for the performance metric calculation and
return the metric result from the method. Predicted values and ground truth values normally needed for the performance
metric calculations are available inside the metric as object attributes and can thus be accessed as: ``self.y_true``
and ``self.y_predicted`` throughout the metric class, ``calculate_metric()`` included.

Example Result Metric implementation:

.. code-block:: python

    from sklearn.metrics import accuracy_score
    from aitoolbox.experiment.core_metrics.abstract_metric import AbstractBaseMetric


    class ExampleAccuracyMetric(AbstractBaseMetric):
        def __init__(self, y_true, y_predicted, positive_class_thresh=0.5):
            # All additional attributes should be defined before the AbstractBaseMetric.__init__
            self.positive_class_thresh = positive_class_thresh
            AbstractBaseMetric.__init__(self, y_true, y_predicted, metric_name='Accuracy')

        def calculate_metric(self):
            if self.positive_class_thresh is not None:
                self.y_predicted = self.y_predicted >= self.positive_class_thresh

            return accuracy_score(self.y_true, self.y_predicted)
