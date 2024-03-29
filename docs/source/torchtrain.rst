torchtrain
==========

:mod:`aitoolbox.torchtrain` is the main user-facing API of the AIToolbox package. It incorporates the PyTorch model
training via the train loop engine as well as automatic experiment progress and performance tracking. The experiment
results are either stored only locally or if desired also automatically synced to the selected cloud storage
(AWS S3 or Google Cloud Storage).

.. toctree::
   :maxdepth: 1
   :caption: Guides:

   torchtrain/train_loop
   torchtrain/model
   torchtrain/callbacks
   torchtrain/schedulers
   torchtrain/multi_loss_opti
   torchtrain/parallel
   torchtrain/amp_training
   torchtrain/advanced
