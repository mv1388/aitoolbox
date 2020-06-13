Model Wrap and Batch Feed Definition
====================================

The preferred way of defining the model compatible with the TrainLoop is to implement it as the **TTModel** as discussed
in the :doc:`../model` section.

The legacy approach of defining the model for the TrainLoop which still comes in handy in certain specific use cases
was to implement a normal PyTorch nn.Module and define a separate **batch feed definition** for this particular model.
Batch feed definitions are objects which need to be inherited from
:class:`aitoolbox.torchtrain.data.batch_model_feed_defs.AbstractModelFeedDefinition` and the user has to
implement its abstract methods.

It can be seen that the abstract methods requiring the implementation as part of the *AbstractModelFeedDefinition* are
exactly the same as those which need to be implemented as part of the new TTModel definition discussed in further detail
in the :doc:`../model` section. While using TTModel is better for readability and experiment tracking on the other hand
in some rare use cases operating on the core PyTorch *nn.Module* model is required instead of using the TTModel
extension. For such cases the *nn.Module + model feed definition* combination option has been kept in the AIToolbox.

Last step that needs to be done in order to train the nn.Module with it's feed definition as part of the TrainLoop
is to wrap the model and the feed definition into the :class:`aitoolbox.torchtrain.model.ModelWrap`. TrainLoop will
automatically detect the use of separate feed definition instead of the TTModel and execute the training based on the
contents of the provided ``ModelWrap``.

Example of the training with the model feed definition
------------------------------------------------------

For the practical example how the nn.Module can be paired together with its model feed definition and wrapped into the
ModelWrapp for the TrainLoop training have a look at the this `example training script
<https://github.com/mv1388/aitoolbox/blob/master/examples/TrainLoop_use/model_definition_examples/trainloop_model_wrap.py>`_.
