experiment
==========

:mod:`aitoolbox.experiment` defines the experiment tracking and performance evaluation components. Because all
implemented components are completely independent from the TrainLoop engine they can be used either on their own
in a more manual mode or as part of the TrainLoop functionality available in :mod:`aitoolbox.torchtrain`. Due to the
independence of the components, certain elements, for performance evaluation can even be utilized for evaluation of
non-PyTorch models.

In general, :mod:`aitoolbox.experiment` helps the user with the following:

* Structured and reusable performance evaluation logic definition
   * :mod:`aitoolbox.experiment.result_package`
   * :mod:`aitoolbox.experiment.core_metrics`
* Tracked training performance history primitive
   * :mod:`aitoolbox.experiment.training_history`
* High level experiment tracking API
   * :mod:`aitoolbox.experiment.experiment_saver`
   * :mod:`aitoolbox.experiment.local_experiment_saver`
* Low level experiment tracking primitives for model saving and performance results saving
   * :mod:`aitoolbox.experiment.local_save`
* Saved model re-loading low level primitives
   * :mod:`aitoolbox.experiment.local_load.local_model_load`


.. toctree::
   :maxdepth: 1
   :caption: Guides:

   experiment/result_package
   experiment/metrics
   experiment/training_history
   experiment/experiment_save
