.. AIToolbox documentation master file, created by
   sphinx-quickstart on Fri Apr 10 11:52:10 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

AIToolbox - PyTorch Model Training Framework with Experiment Tracking Support
===================================================================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   :hidden:

   torchtrain
   experiment
   cloud
   examples
   api/aitoolbox


A framework which helps you train deep learning models in PyTorch and quickly iterate experiments.
It hides the repetitive technicalities of training the neural nets and frees you to focus on interesting part of
devising new models. In essence, it offers a keras-style train loop abstraction which can be used for higher
level training process while still allowing the manual control on the lower level when desired.

In addition to orchestrating the model training loop the framework also helps you keep track of different
experiments by automatically saving models in a structured traceable way and creating performance reports.
These can be stored both locally or on AWS S3 (Google Cloud Storage in beta) which makes the library very useful
when training on the GPU instance on AWS. Instance can be automatically shut down when training is finished and all
the results are safely stored on S3.

