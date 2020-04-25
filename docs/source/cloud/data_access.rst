Data Access
===========

:mod:`aitoolbox.cloud.AWS.data_access` and :mod:`aitoolbox.cloud.GoogleCloud.data_access` implement low-level APIs
to download and upload data to the *AWS S3* and *Google Cloud Storage*.

For *AWS S3* data uploading and downloading use:

* :class:`aitoolbox.cloud.AWS.data_access.BaseDataSaver`
* :class:`aitoolbox.cloud.AWS.data_access.BaseDataLoader`

For *Google Cloud Storage* data uploading and downloading use:

* :class:`aitoolbox.cloud.GoogleCloud.data_access.BaseGoogleStorageDataSaver`
* :class:`aitoolbox.cloud.GoogleCloud.data_access.BaseGoogleStorageDataLoader`
