Message Passing Service
=======================

Most of the time different components in AIToolbox operate either in isolation or communicate over specified APIs.
While this is useful practice for error prevention in some cases less structured form of communication between
components might be desired in order to simplify research development. One such example is the communication between
different callbacks the user might provide to the TrainLoop. To support the convenient and easy development of
callbacks and their communication the TrainLoop provides the *message passing service* implemented in
:mod:`aitoolbox.torchtrain.tl_components.message_passing`.


MessageService Details
----------------------

:class:`aitoolbox.torchtrain.tl_components.message_passing.MessageService` is running as part of the TrainLoop and is
exposed inside every provided callback via the ``self.message_service``.
When we want to pass some information from one callback to another callback (e.g. path where some intermediary results
were saved) the sender callback has to send it into the *MessageService* by calling
:meth:`aitoolbox.torchtrain.tl_components.message_passing.MessageService.write_message` (inside the callback
implementation that would be ``self.message_service.write_message()``). Messages can be considered as a key-value pair
with added message lifecycle setting.

Depending on the **message lifecycle setting**, the messages can be kept in the message service until the end of training,
end of epoch or until first read. As such the message service allows the asynchronous and independent operation of
callbacks enabling the users to add or remove callbacks from the training process as they will without running into
interdependency issues. The message lifecycle settings can be imported from the
:mod:`aitoolbox.torchtrain.tl_components.message_passing`. Currently supported settings are:

* ``KEEP_FOREVER``
* ``UNTIL_END_OF_EPOCH``
* ``UNTIL_READ``
* ``OVERWRITE``

In addition to writing messages, the message service of course also supports the reading of the accumulated messages.
This can be achieved in any TrainLoop component having access to the *MessageService* (callbacks included) by calling
:meth:`aitoolbox.torchtrain.tl_components.message_passing.MessageService.read_messages`. This method will return
all the messages accumulated under the specified key.

In our earlier example of one callback writing a message with the path to the stored intermediary results, the second
callbacks tasked with processing the results or maybe saving them to the cloud would read that message with the data
path and execute it's logic on the data originally provided by the first callback.

Example of MessageService in action
-----------------------------------

An actual example of such message passing between different callbacks can be observed in the implementations of
:class:`aitoolbox.torchtrain.callbacks.performance_eval.ModelTrainHistoryPlot` which sends the message containing the
results path and the :class:`aitoolbox.torchtrain.callbacks.basic.EmailNotification` which reads that message and uses
the sent results path.
