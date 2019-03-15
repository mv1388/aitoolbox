# AI TOOLBOX

Library that helps you train neural networks and keep track of different experiments by automatically saving models and creating performance reports. These can be stored both locally or on AWS S3 which makes the library very useful when training on the GPU instance on AWS. Instance can be automatically shut down when training is finished and all the results are safely stored on S3.

Further use of this library is more targeted towards PyTorch training. It offers a keras style train loop abstraction which can be used for higher level training proces while still allowing to control training on the lower level when desired with the use of different callbacks.


## torchtrain

`TrainLoop` abstraction for PyTorch neural net training. Additional logic can be injected into the training procedure by using `callbacks`. Implement corresponding methods to execute callbacks at the start/end of batch, epoch, training.

By using different implemented derivations of `TrainLoop`, the automatic model checkpoint and performance evaluation report saving or end of training saving can be achieved. Furthermore, saved models and evaluation reports will also be automatically uploaded to AWS S3.  


## experiment_save

### Result Package

Definition of the model evaluation procedure on the task we are experimenting with. Under the hood the result package executes one or more `metrics` objects which actually calculate the performance metric calculation. Result package object is thus used as a wrapper around potentially multiple performance calculations which are needed for our task.

### Experiment Saver 

Saves the model architecture as well as model performance evaluation results and training history. This can be done at the end of each epoch as a model checkpointing or at the end of training.


## AWS 

Functionality for saving model architecture and weights to S3 either during training or at the training end. At the same time the code here can be also used to store model performance reports to S3 in the similar fashion as in the case of model saving.


## NLP

Still work in progress... 
Different NLP data processing functions and NLP oriented task performance evaluation definitions such as Q&A, summarization, machine translation, ...


# Examples of package usage

Look into the `examples` folder for starters. Will be adding more examples of different training scenarios.


## Requesting Spot Instances on AWS

https://chatbotslife.com/aws-setup-for-deep-learning-fda04db6df75


