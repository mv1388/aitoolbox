Schedulers
==========

TrainLoop-based training supports the use of learning rate schedulers. The built-in common learning rate
schedulers can be found in the :mod:`aitoolbox.torchtrain.schedulers` sub-package.
Currently AIToolbox comes out of the bag with the following scheduler types:

* :mod:`aitoolbox.torchtrain.schedulers.basic` - basic scheduler components and general schedulers
* :mod:`aitoolbox.torchtrain.schedulers.warmup` - schedulers based on HuggingFace Transformers

Schedulers are given to any TrainLoop type the same way as callbacks via the callbacks list provided to
:meth:`~aitoolbox.torchtrain.train_loop.train_loop.TrainLoop.fit`.


Implementing New Schedulers
---------------------------

Under the hood the schedulers are just AIToolbox :doc:`callbacks`. Consequently when desired, new learning rate
schedulers can easily be implemented by just inheriting them from the commonly used
:class:`~aitoolbox.torchtrain.callbacks.abstract.AbstractCallback` base class and implementing the necessary
learning rate scheduling logic.
