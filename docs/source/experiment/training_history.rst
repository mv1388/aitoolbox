Training History
================

:class:`aitoolbox.experiment.training_history.TrainingHistory` is the util class which helps the user with tracking
multiple performance metrics results during the model training process.

Under the hood it is just a simple Python dict and it thus supports many of the standard dict operations and operators
for easier use. However the ``TrainingHistory`` also offers a convenient API on top of the dict geared towards
experiment performance tracking. It is mostly used under the hood of the torchtrain TrainLoops, but it can be
easily also utilized as a standalone results tracker for any kind of ML experiments.
