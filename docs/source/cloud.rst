cloud
=====

:mod:`aitoolbox.cloud` implements the components for communication and management of different cloud based services.
Most of the functionality is available both for AWS and Google Cloud.

At its core, the package implements data saving and data downloading from the cloud storage. On top of this there
are high level APIs available for conveniently downloading datasets and saving/loading models from the cloud data
storage such as AWS S3.


.. toctree::
   :maxdepth: 1
   :caption: Guides:

   cloud/cloud_save
   cloud/cloud_load
   cloud/data_access
   cloud/aws_ses
