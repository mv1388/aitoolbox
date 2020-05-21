# AI Toolbox

[![PyPI version](https://badge.fury.io/py/aitoolbox.svg)](https://badge.fury.io/py/aitoolbox)
[![Build Status](https://travis-ci.org/mv1388/AIToolbox.svg?branch=master)](https://travis-ci.org/mv1388/AIToolbox)
[![Documentation Status](https://readthedocs.org/projects/aitoolbox/badge/?version=latest)](https://aitoolbox.readthedocs.io/en/latest/?badge=latest)
&nbsp; &nbsp;
[![codebeat badge](https://codebeat.co/badges/04217a3f-a838-418f-8f14-66cf6ae1b03d)](https://codebeat.co/projects/github-com-mv1388-aitoolbox-master)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/8349596c31a948d8916814a2037ffdf3)](https://www.codacy.com/manual/mv1388/AIToolbox?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=mv1388/AIToolbox&amp;utm_campaign=Badge_Grade)
[![CodeFactor](https://www.codefactor.io/repository/github/mv1388/aitoolbox/badge)](https://www.codefactor.io/repository/github/mv1388/aitoolbox)

[**Documentation**](https://aitoolbox.readthedocs.io/en/latest/)


AIToolbox is a framework which helps you train deep learning models in PyTorch and quickly iterate experiments. 
It hides the repetitive technicalities of training the neural nets and 
frees you to focus on interesting part of devising new models. 
In essence, it offers a keras-style train loop abstraction which can be used for higher 
level training process while still allowing the manual control on the lower 
level when desired.

In addition to orchestrating the model training loop the framework also helps you keep track of different 
experiments by automatically saving models in a structured traceable way and creating performance reports. 
These can be stored both locally or on AWS S3 (Google Cloud Storage in beta) which makes the library 
very useful when training on the GPU instance on AWS. Instance can be 
automatically shut down when training is finished and all the results 
are safely stored on S3.


## Installation
To install the AIToolbox package execute:
```bash
pip install aitoolbox
```

If you want to install the most recent version from github repository, first clone the package repository and 
then install via the `pip` command:
```bash
git clone https://github.com/mv1388/aitoolbox.git

pip install ./aitoolbox
```

AIToolbox package can be also provided as a dependency in the `requirements.txt` file. This can be done by 
just specifying the `aitoolbox` dependency. On the other hand, to automatically
download the current master branch from github include the following dependency specification in the requirements.txt:
```bash
git+https://github.com/mv1388/aitoolbox#egg=aitoolbox
```  


## TrainLoop

[`TrainLoop`](/aitoolbox/torchtrain/train_loop.py) is the main abstraction for PyTorch neural net training. At its core
it handles the batch feeding of data into the model, calculating loss and updating parameters for a specified number of epochs.
To learn how to define the TrainLoop supported PyTorch model please look at the [Model](#model) section bellow.

After the model is created, the simplest way to train it via the TrainLoop abstraction is by doing the following:
```python
from aitoolbox.torchtrain.train_loop import *

tl = TrainLoop(model,
               train_loader, val_loader, test_loader,
               optimizer, criterion)

model = tl.fit(num_epochs=10)
```

AIToolbox includes a few more advanced derivations of the basic TrainLoop
which automatically handle the experiment tracking by creating model
checkpoints, performance reports, example predictions, etc. All of this can be saved just on the local drive
or can also be automatically also stored on AWS S3.  Currently implemented advanced 
[`TrainLoops`](/aitoolbox/torchtrain/train_loop.py) are `TrainLoopCheckpoint`, `TrainLoopEndSave` and `TrainLoopCheckpointEndSave`.
Here, 'Checkpoint' stands for checkpointing after each epoch, while 'EndSave' will only persist and evaluate at the very end of the training. 

For the most complete experiment tracking it is recommended to use the `TrainLoopCheckpointEndSave` option. 
The optional use of the *result packages* needed for the neural net performance evaluation is explained in 
the [experiment section](#experiment) bellow.
```python
from aitoolbox.torchtrain.train_loop import *

TrainLoopCheckpointEndSave(
    model,
    train_loader, validation_loader, test_loader,
    optimizer, criterion,
    project_name, experiment_name, local_model_result_folder_path,
    hyperparams, val_result_package=None, test_result_package=None,
    cloud_save_mode='s3', bucket_name='models', cloud_dir_prefix=''
)
```

Check out a full [TrainLoop training & experiment tracking example](https://github.com/mv1388/aitoolbox/blob/master/examples/TrainLoop_use/trainloop_fully_tracked_experiment.py).


## Multi-GPU training

All TrainLoop versions in addition to single GPU also support multi-GPU training to achieve even faster training.
Following the core PyTorch setup, two multi-GPU training approaches are available: 
`DataParallel` implemented via `TTDataParallel` and `DistributedDataParallel` implemented via `TTDistributedDataParallel`.

### DataParallel - via TTDataParallel

To use DataParallel-like multiGPU training with TrainLoop just wrap the model (`TTModel`, [more in Model section](#model))
into the `TTDataParallel` object, the same way it would done in core PyTorch:
```python
from aitoolbox.torchtrain.train_loop import *
from aitoolbox.torchtrain.parallel import TTDataParallel

model = ... # TTModel
model = TTDataParallel(model)

TrainLoop(
    model,
    train_loader, val_loader, test_loader,
    optimizer, criterion
).fit(num_epochs=10)
```

Check out a full [DataParallel training example](https://github.com/mv1388/aitoolbox/blob/master/examples/dp_ddp_training/dp_training.py).

### DistributedDataParallel - via TTDistributedDataParallel

Distributed training on multiple GPUs via DistributedDataParallel is enabled by the TrainLoop itself under
the hood by wrapping the model (`TTModel`, [more in Model section](#model)) into `TTDistributedDataParallel`.
TrainLoop also automatically spawns multiple processes and initializes them. Inside each spawned process
the model and all other necessary training components are moved to the correct GPU belonging to a specific
process. Lastly, TrainLoop also automatically adds the PyTorch `DistributedSampler` to each of the provided
data loaders in order to ensure different data batches go to different GPUs and there is no overlap.

To enable distributed training via DistributedDataParallel, all the user has to do is to initialize
TrainLoop where `TTModel` should be provided and then call train loop's dedicated `fit_distributed()` 
function (instead of `fit()` used otherwise when not training distributed).
```python
from aitoolbox.torchtrain.train_loop import *

model = ... # TTModel

TrainLoop(
    model,
    train_loader, val_loader, test_loader,
    optimizer, criterion
).fit_distributed(num_epochs=10, callbacks=None,
                  train_data_shuffle=True, ddp_model_args=None,
                  num_nodes=1, node_rank=0, num_gpus=torch.cuda.device_count())
```

Check out a full [DistributedDataParallel training example](https://github.com/mv1388/aitoolbox/blob/master/examples/dp_ddp_training/ddp_training.py).

## Automatic Mixed Precision training via Nvidia Apex

All the TrainLoop versions also support training with Automatic Mixed Precision (*AMP*)
using the [Nvidia apex](https://github.com/NVIDIA/apex) extension. To use this feature the user first
has to install the Nvidia apex library ([installation instructions](https://github.com/NVIDIA/apex#linux)).

The user only has to set the TrainLoop parameter `use_amp` to `use_amp=True` in order to use the default 
AMP initialization and start training the model in the mixed precision mode. If the user wants to specify custom 
AMP initialization parameters, these should be provided as a dict parameter `use_amp={'opt_level': 'O1'}` to 
the TrainLoop. All AMP initializations and training related steps are then handled automatically by the TrainLoop. 

You can read more about different AMP optimization levels in the
[Nvidia apex documentation](https://nvidia.github.io/apex/amp.html#opt-levels-and-properties).

### Single-GPU mixed precision training
Example of single-GPU APEX setup:
```python
from aitoolbox.torchtrain.train_loop import *

model = ... # TTModel

TrainLoop(
    model, ...,
    optimizer, criterion, use_amp={'opt_level': 'O1'}
).fit(num_epochs=10)
``` 

Check out a full [Apex AMP training example](https://github.com/mv1388/aitoolbox/blob/master/examples/apex_amp_training/apex_single_GPU_training.py).

### Multi-GPU DDP mixed precision training
When training in the multi-GPU setting, the setup is mostly the same as in the single-GPU. 
All the user has to do is set accordingly the `use_amp` parameter of the TrainLoop and to instead call 
`fit_distributed()` in order to start the distributed training. 
Under the hood, TrainLoop will initialize the model and the optimizer for AMP and start training using 
DistributedDataParallel approach (DDP is currently only multi-GPU training setup supported by Apex AMP).
```python
from aitoolbox.torchtrain.train_loop import *

model = ... # TTModel

TrainLoop(
    model, ...,
    optimizer, criterion, use_amp={'opt_level': 'O1'}
).fit_distributed(num_epochs=10)
``` 

Check out a full [Apex AMP DistributedDataParallel training example](https://github.com/mv1388/aitoolbox/blob/master/examples/apex_amp_training/apex_mutli_GPU_training.py).


## Model

To take advantage of the TrainLoop abstraction the user has to define their model as a class which is a standard way
in core PyTorch as well. The only difference is that for TrainLoop supported training the model class has 
to be inherited from the AIToolbox specific [`TTModel`](/aitoolbox/torchtrain/model.py) base class instead of PyTorch `nn.Module`.

`TTModel` itself inherits from the normally used `nn.Module` class thus our models still
retain all the expected PyTorch enabled functionality. The reason for using the TTModel super class is that
TrainLoop requires users to implement two additional methods which describe how each batch of data
is fed into the model when calculating the loss in the training mode and when making the predictions in the 
evaluation mode.

The code below shows the general skeleton all the TTModels have to follow to enable them to be trained 
with the TrainLoop:
```python
from aitoolbox.torchtrain.model import TTModel

class MyNeuralModel(TTModel):
    def __init__(self):
        # model layers, etc.

    def forward(self, x_data_batch):
        # The same method as required in the base PyTorch nn.Module
        ...
        # return prediction
        
    def get_loss(self, batch_data, criterion, device):
        # Get loss during training stage, called from fit() in TrainLoop
        ...
        # return batch loss

    def get_predictions(self, batch_data, device):
        # Get predictions during evaluation stage 
        # + return any metadata potentially needed for evaluation
        ...
        # return predictions, true_targets, metadata
```

## Callbacks

For advanced applications the basic logic offered in different default TrainLoops might not be enough.
Additional needed logic can be injected into the training procedure by using [`callbacks`](/aitoolbox/torchtrain/callbacks)
and providing them as a parameter list to TrainLoop's `fit(callbacks=[callback_1, callback_2, ...])` function. 

AIToolbox by default already offers a wide selection of different useful callbacks. However when
some completely new functionality is desired the user can also implement their own callbacks by 
inheriting from the base callback object [`AbstractCallback`](/aitoolbox/torchtrain/callbacks/abstract.py). 
All that the user has to do is to implement corresponding methods to execute the new callback 
at the desired point in the train loop, such as: start/end of batch, epoch, training.


## experiment

### Result Package

This is the definition of the model evaluation procedure on the task we are experimenting with.
Result packages available out of the box can be found in the [`result_package` module](/aitoolbox/experiment/result_package/)
where we have implemented several [basic, general result packages](/aitoolbox/experiment/result_package/basic_packages.py). 
Furthermore, for those dealing with NLP, result packages for
several widely researched NLP tasks such as translation, QA can be found as part of the 
[`NLP` module](/aitoolbox/nlp/experiment_evaluation/NLP_result_package.py)
module. Last but not least, as the framework was built with extensibility in mind and thus 
if needed the users can easily define their own result packages with custom evaluations by extending the base
[`AbstractResultPackage`](/aitoolbox/experiment/result_package/abstract_result_packages.py). 
 
Under the hood the result package executes one or more [`metrics`](/aitoolbox/experiment/core_metrics) objects which actually 
calculate the performance metric calculation. Result package object is thus used as a wrapper 
around potentially multiple performance calculations which are needed for our task. The metrics
which are part of the specified result package are calculated by calling the `prepare_result_package()` method 
of the result package which we are using to evaluate model's performance.

### Experiment Saver 

The experiment saver saves the model architecture as well as model performance evaluation results and training history. 
This can be done at the end of each epoch as a model checkpointing or at the end of training.

Normally not really a point of great interest when using the TrainLoop interface as it is hidden under the hood.
However as AIToolbox was designed to be modular one can decide to write their own training loop logic but
just use the provided experiment saver module to help with the experiment tracking and model saving.
For PyTorch users we recommend using the [`FullPyTorchExperimentS3Saver`](/aitoolbox/experiment/experiment_saver.py) 
which has also been most thoroughly tested. 
The experiment is saved by calling the `save_experiment()` function from the selected experiment saver and 
providing the trained model and the evaluated result package containing the calculated performance results.


## cloud

All of these modules are mainly hidden under the hood when using different experiment tracking
abstractions. However, if desired and only the cloud saving functionality is needed it is easy to use them
as standalone modules in some desired downstream application.

### AWS 

Functionality for saving model architecture and training results to S3 either during 
training or at the training end. On the other hand, the module also offers the dataset downloading
from the S3 based dataset store. This is useful when we are experimenting with datasets and have only a slow
local connection, thus scp/FTP is out of the picture.

### Google Cloud

Same functionality as for AWS S3 but for Google Cloud Storage. 
Implemented, however, not yet tested in practice. 


## nlp

Currently, mainly used for the performance evaluation [`result packages`](/aitoolbox/nlp/experiment_evaluation/NLP_result_package.py) 
needed for different NLP tasks, such as Q&A, summarization, machine translation. 

For the case of e.g. NMT the module also provides [attention heatmap plotting](/aitoolbox/nlp/experiment_evaluation/attention_heatmap.py)
which is often helpful for gaining addition insights into the seq2seq model. The heatmap plotter
creates attention heatmap plots for every validation example and saves them as pictures to disk 
(potentially also to cloud).

Lastly, the nlp module also provides several rudimentary NLP data processing functions.


## AWS GPU instance prep and management bash scripts

As some of the tasks when training models on the cloud GPU are quite repetitive, the package
also includes several useful bash scripts to automatize tasks such as instance init preparation,
experiment file updating, remote AIToolbox installation updating, etc.

For further information look into the [`/bin/AWS`](/bin/AWS/) folder and read 
the provided [README](/bin/AWS/README.md).


# Examples of package usage

Look into the [`/examples`](/examples) folder for starters. 
Will be adding more examples of different training scenarios.
