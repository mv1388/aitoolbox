.. AIToolbox documentation master file, created by
   sphinx-quickstart on Fri Apr 10 11:52:10 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

AIToolbox - Model Training Framework for PyTorch
================================================

.. toctree::
   :maxdepth: 5
   :caption: Components:
   :hidden:

   torchtrain
   experiment
   cloud

.. toctree::
   :maxdepth: 5
   :caption: Examples:
   :hidden:

   examples

.. toctree::
   :maxdepth: 5
   :caption: API:
   :hidden:

   api/api_aitoolbox


AIToolbox is a framework which helps you train deep learning models in PyTorch and quickly iterate experiments.
It hides the repetitive technicalities of training the neural nets and frees you to focus on interesting part of
devising new models. In essence, it offers a keras-style train loop abstraction which can be used for higher
level training process while still allowing the manual control on the lower level when desired.

In addition to orchestrating the model training loop the framework also helps you keep track of different
experiments by automatically saving models in a structured traceable way and creating performance reports.
These can be stored both locally or on AWS S3 (Google Cloud Storage in beta) which makes the library very useful
when training on the GPU instance on AWS. Instance can be automatically shut down when training is finished and all
the results are safely stored on S3.


Main Components
---------------

AIToolbox consists of three main user-facing components:

* :mod:`aitoolbox.torchtrain` - PyTorch train loop engine
* :mod:`aitoolbox.experiment` - experiment tracking
* :mod:`aitoolbox.cloud` - cloud operations for *AWS* and *Google Cloud*

All three AIToolbox components can be used independently when only some subset of functionality is desired in a project.
However, the greatest benefit of AIToolbox comes when all components are used together in unison in order to ease
the process of PyTorch model training and experiment tracking as much as possible.
Most of this top-level API is exposed to the user via the functionality implemented in :mod:`aitoolbox.torchtrain`.

To learn more about each of AIToolbox components have a look at the corresponding documentation sections:

* :doc:`torchtrain`
* :doc:`experiment`
* :doc:`cloud`
