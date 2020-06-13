Model Prediction Store
======================

In order to save compute time and prevent repetitive re-computation leading to the same output, TrainLoop utilizes the
:class:`aitoolbox.torchtrain.tl_components.model_prediction_store.ModelPredictionStore` which is used for results
caching.

Especially when using multiple callbacks all executing the same computation, such as making predictions on
the validation set this can get quite time consuming. To speed up training process TrainLoop will calculate the
prediction on particular dataset as part of the current epoch only once and then cache the predictions. If as part
of the same epoch another calculation of predictions on the same data set is requested, the TrainLoop will retrieve
the cached results instead of recomputing them again. Currently the ``ModelPredictionStore`` supports caching the model
loss and model prediction caching on the train, validation and test data sets.

As part of the TrainLoop the model prediction store cache lifecycle ends at the end of the epoch. All the cached model
outputs are removed at the end of the epoch and the new epoch where the weights of the model will change is started
with the clean prediction cache.

To most users this caching is visible as part of the TrainLoop's loss calculation methods:

* :meth:`aitoolbox.torchtrain.train_loop.TrainLoop.evaluate_loss_on_train_set`
* :meth:`aitoolbox.torchtrain.train_loop.TrainLoop.evaluate_loss_on_validation_set`
* :meth:`aitoolbox.torchtrain.train_loop.TrainLoop.evaluate_loss_on_test_set`

and as part of the TrainLoop's model prediction calculation methods:

* :meth:`aitoolbox.torchtrain.train_loop.TrainLoop.predict_on_train_set`
* :meth:`aitoolbox.torchtrain.train_loop.TrainLoop.predict_on_validation_set`
* :meth:`aitoolbox.torchtrain.train_loop.TrainLoop.predict_on_test_set`

Important to note here, is that by default TrainLoop will try to save compute time and cache model outputs when possible
instead of recomputing them. However, if for a particular use case the user wants to get fresh recomputed loss or model
predictions then the ``force_prediction`` parameter in any of the model output computation methods listed above
has to be switched to ``True``. This will cause them to ignore the cached values and recompute them from scratch.
