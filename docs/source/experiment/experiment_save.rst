Experiment Saving
=================

One of the main uses of :mod:`aitoolbox.experiment` package is tracking experiments and saving models and result as
the training progresses. This way the executed is well documented and its parameters and details can easily be
determined even a long time after the original training.

The components in this package basically help prevent training the model, then doing
some other experiments and coming back to the original experiment one month later and having no clue what was actually
the experiment setting, what was the performance and how exactly were the results produced.


Experiment Saver
----------------

AIToolbox experiment savers at their core handle the creation of the experiment folder into which all the experiment
results and model are automatically saved in a structured way which helps the experiment traceability. Experiment
savers represent the high-level experiment tracking API and generally fall into two main categories:
cloud experiment savers and local-only experiment savers.

Local-only experiment savers in :mod:`aitoolbox.experiment.local_experiment_saver` are simpler and only save
the the experiment results onto the local drive. Cloud-enabled experiment savers in
:mod:`aitoolbox.experiment.experiment_saver` are an extension in sense that they in addition to tracking the experiment
on the local drive also automatically take care of uploading all the produced results and models to the cloud storage.
This is especially useful when shutting automatically down the GPU instance after the training is finished. By using
cloud experiment saver, all the experiment results are safely and automatically persisted in the cloud storage even
after all the locally produced results are deleted when the instance is terminated.

Cloud enabled experiment savers:

* :class:`~aitoolbox.experiment.experiment_saver.FullPyTorchExperimentS3Saver`
* :class:`~aitoolbox.experiment.experiment_saver.FullKerasExperimentS3Saver`
* :class:`~aitoolbox.experiment.experiment_saver.FullPyTorchExperimentGoogleStorageSaver`
* :class:`~aitoolbox.experiment.experiment_saver.FullKerasExperimentGoogleStorageSaver`

Local-only experiment savers:

* :class:`~aitoolbox.experiment.local_experiment_saver.FullPyTorchExperimentLocalSaver`
* :class:`~aitoolbox.experiment.local_experiment_saver.FullKerasExperimentLocalSaver`

A very convenient property all the experiment savers have is that they all implement the same user facing API
which makes them ideal for easy use as part of the larger system. Due to the unified API different experiment saver
types can be easily dynamically exchanged according to desired training scenarios without any need to modify
the surrounding code. The core API function that is common to all the experiment savers used to initiate the experiment
snapshot saving is :meth:`~aitoolbox.experiment.experiment_saver.AbstractExperimentSaver.save_experiment`.


Local Save
----------

While experiment savers described above serve as the high-level experiment tracking API, the local model and results
savers are the low-level components on top of which the experiment saver API is built. Most users will probably very
often just use the experiment savers, however in certain use cases the use of more low level components could still
be desired.

The local experiment data saving low-level components can be found in the :mod:`aitoolbox.experiment.local_save` subpackage.
They handle all the experiment tracking tasks ranging from experiment folder structuring, neural model weights & optimizer
state saving and all the way to tracked results packaging before finally saving to the local drive.

Local Model Save
^^^^^^^^^^^^^^^^

Implementations of model saving logic to the local drive. Currently available model savers:

* :class:`~aitoolbox.experiment.local_save.local_model_save.PyTorchLocalModelSaver`
* :class:`~aitoolbox.experiment.local_save.local_model_save.KerasLocalModelSaver`

Local Results Save
^^^^^^^^^^^^^^^^^^

Implementation of training results saving logic to the local drive available in
:class:`~aitoolbox.experiment.local_save.local_results_save.LocalResultsSaver`. This class offers two main options to
save produced experiment results:

* saving all results into the single (potentially large) file via
  :class:`~aitoolbox.experiment.local_save.local_results_save.LocalResultsSaver.save_experiment_results`
* saving results into the single multiple separate files via
  :class:`~aitoolbox.experiment.local_save.local_results_save.LocalResultsSaver.save_experiment_results_separate_files`
