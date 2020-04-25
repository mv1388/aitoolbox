Loading Models from Cloud
=========================

:mod:`aitoolbox.cloud.AWS.model_load` and :mod:`aitoolbox.cloud.GoogleCloud.model_load` modules provide logic to
conveniently download the previously saved model checkpoint from the cloud storage and initialize local model weight.
This in effect automatically jump-starts the local model from the saved checkpoint and makes it ready for inference
use or further training.

Currently available cloud model loading is for the PyTorch and supports *AWS S3* and *Google Cloud Storage*:

* :class:`aitoolbox.cloud.AWS.model_load.PyTorchS3ModelLoader`
* :class:`aitoolbox.cloud.GoogleCloud.model_load.PyTorchGoogleStorageModelLoader`

To load the model from cloud, first, the user needs to initialize the model object and then give it to the cloud
model loader. Model loader will download the saved model from the cloud and use its weights to initialize the provided
local model. Furthermore the model loader can also initialize the optimizer and the AMP in case the local model is
not going to be used just for inference but for further model training.
