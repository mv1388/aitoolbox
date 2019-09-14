# AI Toolbox

<!--[![Build Status](https://travis-ci.org/mv1388/AIToolbox.svg?branch=master)](https://travis-ci.org/mv1388/AIToolbox)-->
<!--[![CircleCI](https://circleci.com/gh/mv1388/AIToolbox/tree/master.svg?style=svg)](https://circleci.com/gh/mv1388/AIToolbox/tree/master)-->
<!--&nbsp; &nbsp;-->
<!--[![codebeat badge](https://codebeat.co/badges/04217a3f-a838-418f-8f14-66cf6ae1b03d)](https://codebeat.co/projects/github-com-mv1388-aitoolbox-master)-->
<!--[![Codacy Badge](https://api.codacy.com/project/badge/Grade/2a497fd39a0049d19d0749d6dc0beb75)](https://www.codacy.com/manual/mv1388/AIToolbox?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=mv1388/AIToolbox&amp;utm_campaign=Badge_Grade)-->
<!--[![CodeFactor](https://www.codefactor.io/repository/github/mv1388/aitoolbox/badge)](https://www.codefactor.io/repository/github/mv1388/aitoolbox)-->


Library which helps you train deep learning models in PyTorch and quickly iterate experiments. 
It hides the repetitive technicalities of training the neural nets and 
frees you to focus on interesting part of devising new models. 
In essence, it offers a keras-style train loop abstraction which can be used for higher 
level training process while still allowing the manual control on the lower 
level when desired.

In addition to orchestrating the model training loop the framework also helps you keep track of different 
experiments by automatically saving models in a structured traceable way and creating performance reports. 
These can be stored both locally or on AWS S3 (Google Cloud in beta) which makes the library 
very useful when training on the GPU instance on AWS. Instance can be 
automatically shut down when training is finished and all the results 
are safely stored on S3.


## torchtrain

### TrainLoop

[`TrainLoop`](/AIToolbox/torchtrain/train_loop.py) is the main abstraction for PyTorch neural net training. At it's core
it handles the batch feeding of data into the model, calculating loss and updating parameters for a specified number of epochs.
The simplest way to train a neural net is thus by doing the following:
```
tl = TrainLoop(model,
               train_loader, val_loader, test_loader,
               optimizer, criterion)
model = tl.fit(num_epoch=10)
```

AIToolbox includes a few more advanced derivations of the basic TrainLoop
which automatically handle the experiment tracking by creating model
checkpoints, performance reports, example predictions, etc. All of this can be saved just on the local drive
or can also be automatically also stored on AWS S3.  Currently implemented advanced 
[`TrainLoops`](/AIToolbox/torchtrain/train_loop.py) are `TrainLoopModelCheckpoint`, `TrainLoopModelEndSave` and `TrainLoopModelCheckpointEndSave`.
Here, 'Checkpoint' stands for checkpointing after each epoch, while 'EndSave' will only persist and evaluate at the very end of the training. 

For the most complete experiment tracking it is recommended to use the `TrainLoopModelCheckpointEndSave` option. 
The optional use of the *result packages* needed for the neural net performance evaluation is explained in 
the [experiment section](#experiment) bellow.
```
TrainLoopModelCheckpointEndSave(model,
                                train_loader, validation_loader, test_loader,
                                optimizer, criterion,
                                project_name, experiment_name, local_model_result_folder_path,
                                hyperparams, val_result_package=None, test_result_package=None,
                                cloud_save_mode='s3', bucket_name='models', cloud_dir_prefix='',
                                rm_subopt_local_models=False, num_best_checkpoints_kept=2)
```

### Callbacks

For advanced applications the basic logic offered in different default TrainLoops might not be enough.
Additional needed logic can be injected into the training procedure by using [`callbacks`](/AIToolbox/torchtrain/callbacks). 
AIToolbox by default already offers a wide selection of different useful callbacks. However when
some completely new functionality is desired the user can also implement their own callbacks by 
inheriting from the base callback object [`AbstractCallback`](/AIToolbox/torchtrain/callbacks/callbacks.py). 
All that the user has to do is to implement corresponding methods to execute the new callback 
at the desired point in the train loop, such as: start/end of batch, epoch, training.


## experiment

### Result Package

This is the definition of the model evaluation procedure on the task we are experimenting with.
Result packages available out of the box can be found in the [`result_package` module](/AIToolbox/experiment/result_package/)
where we have implemented several [basic, general result packages](/AIToolbox/experiment/result_package/basic_packages.py). 
Furthermore, for those dealing with NLP, result packages for
several widely researched NLP tasks such as translation, QA can be found as part of the 
[`NLP` module](/AIToolbox/nlp/experiment_evaluation/NLP_result_package.py)
module. Last but not least, as the framework was built with extensibility in mind and thus 
if needed the users can easily define their own result packages with custom evaluations by extending the base
[`AbstractResultPackage`](/AIToolbox/experiment/result_package/abstract_result_packages.py).
 
Under the hood the result package executes one or more [`metrics`](/AIToolbox/experiment/core_metrics) objects which actually 
calculate the performance metric calculation. Result package object is thus used as a wrapper 
around potentially multiple performance calculations which are needed for our task.

### Experiment Saver 

The experiment saver saves the model architecture as well as model performance evaluation results and training history. 
This can be done at the end of each epoch as a model checkpointing or at the end of training.

Normally not really a point of great interest when using the TrainLoop interface as it is hidden under the hood.
However as AIToolbox was designed to be modular one can decide to write their own training loop logic but
just use the provided experiment saver module to help with the experiment tracking and model saving.
For PyTorch users we recommend using the [`FullPyTorchExperimentS3Saver`](/AIToolbox/experiment/experiment_saver.py) 
which has also been most thoroughly tested.


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
