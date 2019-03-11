# AI TOOLBOX #

# torchtrain

`TrainLoop` abstraction for PyTorch neural net training. Additional logic can be injected into the training procedure by using `callbacks`. Implement corresponding methods to execute callbacks at the start/end of batch, epoch, training.

By using different implemented derivations of `TrainLoop`, the automatic model checkpoint and performance evaluation report saving or end of training saving can be achieved. Furthermore, saved models and evaluation reports will also be automatically uploaded to AWS S3.  


# experiment_save

## Result Package

Definition of the model evaluation procedure on the task we are experimenting with. Under the hood the result package executes one or more `metrics` objects which actually calculate the performance metric calculation. Result package object is thus used as a wrapper around potentially multiple performance calculations which are needed for our task.

## Local experiment saver 

Saves the model architecture as well as model performance evaluation results and training history to local disk. This can be done at the end of each epoch as a model checkpointing or at the end of training.


# AWS 

Similar idea to the above mentioned Local experiment saver, but instead to saving to local disk the final destination is on AWS S3. 


# Requesting Spot Instances on AWS

https://chatbotslife.com/aws-setup-for-deep-learning-fda04db6df75


