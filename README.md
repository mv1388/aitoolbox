# AI TOOLBOX [Work in Progress]

[![Build Status](https://travis-ci.org/mv1388/AIToolbox.svg?branch=master)](https://travis-ci.org/mv1388/AIToolbox)
[![CircleCI](https://circleci.com/gh/mv1388/AIToolbox/tree/master.svg?style=svg)](https://circleci.com/gh/mv1388/AIToolbox/tree/master)
&nbsp;
[![codebeat badge](https://codebeat.co/badges/04217a3f-a838-418f-8f14-66cf6ae1b03d)](https://codebeat.co/projects/github-com-mv1388-aitoolbox-master)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/2a497fd39a0049d19d0749d6dc0beb75)](https://www.codacy.com/manual/mv1388/AIToolbox?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=mv1388/AIToolbox&amp;utm_campaign=Badge_Grade)
[![CodeFactor](https://www.codefactor.io/repository/github/mv1388/aitoolbox/badge)](https://www.codefactor.io/repository/github/mv1388/aitoolbox)


Library that helps you train neural networks and keep track of different 
experiments by automatically saving models and creating performance reports. 
These can be stored both locally or on AWS S3 which makes the library 
very useful when training on the GPU instance on AWS. Instance can be 
automatically shut down when training is finished and all the results 
are safely stored on S3.

Further use of this library is more targeted towards PyTorch training. 
It offers a keras style train loop abstraction which can be used for higher 
level training proces while still allowing to control training on the lower 
level when desired with the use of different callbacks.


## torchtrain

`TrainLoop` abstraction for PyTorch neural net training. Additional logic 
can be injected into the training procedure by using `callbacks`. 
Implement corresponding methods to execute callbacks at the start/end of batch, epoch, training.

By using different implemented derivations of `TrainLoop`, the automatic 
model checkpoint and performance evaluation report saving or end of training 
saving can be achieved. Furthermore, saved models and evaluation reports 
will also be automatically uploaded to AWS S3.  


## experiment

### Result Package

Definition of the model evaluation procedure on the task we are experimenting with. 
Under the hood the result package executes one or more `metrics` objects which actually 
calculate the performance metric calculation. Result package object is thus used as a wrapper 
around potentially multiple performance calculations which are needed for our task.

### Experiment Saver 

Saves the model architecture as well as model performance evaluation results and training history. 
This can be done at the end of each epoch as a model checkpointing or at the end of training.


## cloud

### AWS 

Functionality for saving model architecture and weights to S3 either during 
training or at the training end. At the same time the code here can be also 
used to store model performance reports to S3 in the similar fashion as in the case of model saving.

### Google Cloud

Same functionality as for AWS S3 but for Google Cloud Storage. 
Implemented, however, not yet tested in practice. 


## NLP

Still work in progress... 
Different NLP data processing functions and NLP oriented task performance 
evaluation definitions such as Q&A, summarization, machine translation, ...

## kerastrain

Beta version of TrainLoop framework which is mainly developed for PyTorch but ported here to Keras


# Examples of package usage

Look into the [`/examples`](/examples) folder for starters. 
Will be adding more examples of different training scenarios.
