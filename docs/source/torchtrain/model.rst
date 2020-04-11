TTModel
=======

*Torchtrain Model - TTModel for short*

To take advantage of the TrainLoop abstraction the user has to define their model as a class which is a standard way
in core *PyTorch* as well. The only difference is that for TrainLoop supported training the model class has to be
inherited from the AIToolbox specific :class:`aitoolbox.torchtrain.model.TTModel` base class instead of
*PyTorch* :class:`torch.nn.Module`.

``TTModel`` itself inherits from the normally used ``nn.Module`` class thus our models still retain all the expected
*PyTorch* enabled functionality. The reason for using the TTModel super class is that TrainLoop requires users to
implement two additional methods which describe how each batch of data is fed into the model when calculating the loss
in the training mode and when making the predictions in the evaluation mode.

In total the user has to implement the following three methods when building a new model inherited from ``TTModel``:

* :meth:`aitoolbox.torchtrain.model.TTModel.forward` (inherited from :meth:`torch.nn.Module.forward`)
* :meth:`aitoolbox.torchtrain.model.TTModel.get_loss`
* :meth:`aitoolbox.torchtrain.model.TTModel.get_predictions`

The code below shows the general skeleton all the TTModels have to follow to enable them to be trained with
the TrainLoop:

.. code-block:: python

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

        def get_loss_eval(self, batch_data, criterion, device):
            # Get loss during evaluation stage. Normally just calls get_loss()
            return self.get_loss(batch_data, criterion, device)

        def get_predictions(self, batch_data, device):
            # Get predictions during evaluation stage
            # + return any metadata potentially needed for evaluation
            ...
            # return predictions, true_targets, metadata

For a full working example of the ``TTModel`` based model definition, check out this
`model example script
<https://github.com/mv1388/aitoolbox/blob/master/examples/TrainLoop_use/model_definition_examples/trainloop_ttmodel.py#L18>`_.
