Saving to Cloud
===============

One of the most important aspects of the :mod:`aitoolbox.cloud` package is saving of data to the cloud storage.

The data saving components for *AWS S3* are available in:

* :mod:`aitoolbox.cloud.AWS.model_save`
* :mod:`aitoolbox.cloud.AWS.results_save`

The data saving components for *Google Cloud Storage* are available in:

* :mod:`aitoolbox.cloud.GoogleCloud.model_save`
* :mod:`aitoolbox.cloud.GoogleCloud.results_save`

The implementations found here provide an easy to use API to upload the saved models and experiment tracking results
to the cloud storage.


Model Saving
------------

``model_save`` modules provide an API to which the user provides the model they wish to save and the module will
automatically first locally save the model in the easy to track folder structure and then upload it to the selected
cloud storage. Cloud experiment folder structure mirrors that which is created on the local drive.
Currently supported cloud model savers can save PyTorch and Keras models to AWS S3 or Google Cloud Storage.

PyTorch cloud model savers:

* :class:`aitoolbox.cloud.AWS.model_save.PyTorchS3ModelSaver`
* :class:`aitoolbox.cloud.GoogleCloud.model_save.PyTorchGoogleStorageModelSaver`

Keras cloud model savers:

* :class:`aitoolbox.cloud.AWS.model_save.KerasS3ModelSaver`
* :class:`aitoolbox.cloud.GoogleCloud.model_save.KerasGoogleStorageModelSaver`



Results Saving
--------------

