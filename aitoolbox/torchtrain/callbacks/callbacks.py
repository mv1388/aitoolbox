import numpy as np
from typing import Optional

from aitoolbox.utils import util
from aitoolbox.cloud.AWS.simple_email_service import SESSender


class AbstractCallback:
    def __init__(self, callback_name, execution_order=0):
        """Abstract callback class that all actual callback classes have to inherit from

        In the inherited callback classes the callback methods should be overwritten and used to implement desired
        callback functionality at specific points of the train loop.

        Args:
            callback_name (str): name of the callback
            execution_order (int): order of the callback execution. If all the used callbacks have the orders set to 0,
                than the callbacks are executed in the order they were registered.
        """
        from aitoolbox.torchtrain.train_loop import TrainLoop
        from aitoolbox.torchtrain.tl_components.message_passing import MessageService

        self.callback_name = callback_name
        self.execution_order = execution_order
        self.train_loop_obj: Optional[TrainLoop] = None
        self.message_service: Optional[MessageService] = None

    def register_train_loop_object(self, train_loop_obj):
        """Introduce the reference to the encapsulating trainloop so that the callback has access to the
            low level functionality of the trainloop

        The registration is normally handled by the callback handler found inside the train loops. The handler is
        responsible for all the callback orchestration of the callbacks inside the trainloops.

        Args:
            train_loop_obj (aitoolbox.torchtrain.train_loop.TrainLoop): reference to the encapsulating trainloop

        Returns:
            AbstractCallback: return the reference to the callback after it is registered
        """
        self.train_loop_obj = train_loop_obj
        self.message_service = train_loop_obj.message_service
        self.on_train_loop_registration()
        return self

    def on_train_loop_registration(self):
        """Execute callback initialization / preparation after the train_loop_object becomes available

        Returns:

        """
        pass

    def on_epoch_begin(self):
        """Logic executed at the beginning of the epoch

        Returns:

        """
        pass

    def on_epoch_end(self):
        """Logic executed at the end of the epoch

        Returns:

        """
        pass

    def on_train_begin(self):
        """Logic executed at the beginning of the overall training

        Returns:

        """
        pass

    def on_train_end(self):
        """Logic executed at the end of the overall training

        Returns:

        """
        pass

    def on_batch_begin(self):
        """Logic executed before the batch is inserted into the model

        Returns:

        """
        pass

    def on_batch_end(self):
        """Logic executed after the batch is inserted into the model

        Returns:

        """
        pass

    def on_after_gradient_update(self):
        """Logic executed after the model gradients are updated

        To ensure the execution of this callback enable the `self.train_loop_obj.grad_cb_used = True` option in the
        on_train_loop_registration(). Otherwise logic implemented here will not be executed by the TrainLoop.

        Returns:

        """
        pass

    def on_after_optimizer_step(self):
        """Logic executed after the optimizer does a new step and updates the model weights

        To ensure the execution of this callback enable the `self.train_loop_obj.grad_cb_used = True` option in the
        on_train_loop_registration(). Otherwise logic implemented here will not be executed by the TrainLoop.

        Returns:

        """
        pass


class AbstractExperimentCallback(AbstractCallback):
    def __init__(self, callback_name, execution_order=0):
        """Extension of the AbstractCallback implementing the automatic experiment details inference from TrainLoop

        This abstract callback is inherited from when the implemented callbacks intend to save results files into the
        experiment folder and also potentially upload them to AWS S3.

        Args:
            callback_name (str): name of the callback
            execution_order (int): order of the callback execution. If all the used callbacks have the orders set to 0,
                than the callbacks are executed in the order they were registered.
        """
        AbstractCallback.__init__(self, callback_name, execution_order)
        self.project_name = None
        self.experiment_name = None
        self.local_model_result_folder_path = None

        # self.cloud_save_mode = 's3'
        # self.bucket_name = 'model-result'
        # self.cloud_dir_prefix = ''

    def try_infer_experiment_details(self, infer_cloud_details):
        """Infer paths where to save experiment related files from the running TrainLoop.

        This details inference function should only be called after the callback has already been registered in the
        TrainLoop, e.g. in the on_train_loop_registration().

        Args:
            infer_cloud_details (bool): should infer only local project folder details or also cloud project destination

        Raises:
            AttributeError

        Returns:
            None
        """
        try:
            if self.project_name is None:
                self.project_name = self.train_loop_obj.project_name
            if self.experiment_name is None:
                self.experiment_name = self.train_loop_obj.experiment_name
            if self.local_model_result_folder_path is None:
                self.local_model_result_folder_path = self.train_loop_obj.local_model_result_folder_path

            if infer_cloud_details:
                if self.cloud_save_mode == 's3' and \
                        hasattr(self.train_loop_obj,
                                'cloud_save_mode') and self.cloud_save_mode != self.train_loop_obj.cloud_save_mode:
                    self.cloud_save_mode = self.train_loop_obj.cloud_save_mode
                if self.bucket_name == 'model-result' and \
                        hasattr(self.train_loop_obj,
                                'bucket_name') and self.bucket_name != self.train_loop_obj.bucket_name:
                    self.bucket_name = self.train_loop_obj.bucket_name
                if self.cloud_dir_prefix == '' and \
                        hasattr(self.train_loop_obj,
                                'cloud_dir_prefix') and self.cloud_dir_prefix != self.train_loop_obj.cloud_dir_prefix:
                    self.cloud_dir_prefix = self.train_loop_obj.cloud_dir_prefix
        except AttributeError:
            raise AttributeError('Currently used TrainLoop does not support automatic project folder structure '
                                 'creation. Project name, etc. thus can not be automatically deduced. Please provide'
                                 'it in the callback parameters instead of currently used None values.')


class ListRegisteredCallbacks(AbstractCallback):
    def __init__(self):
        """List all the callbacks which are used in the current TrainLoop

        """
        AbstractCallback.__init__(self, 'Print the list of registered callbacks')

    def on_train_begin(self):
        self.train_loop_obj.callbacks_handler.print_registered_callback_names()


class EarlyStopping(AbstractCallback):
    def __init__(self, monitor='val_loss', min_delta=0., patience=0):
        """Early stopping of the training if the performance stops improving

        Args:
            monitor (str): performance measure that is tracked to decide if performance is improving during training
            min_delta (float): by how much the performance has to improve to still keep training the model
            patience (int): how many epochs the early stopper waits after the performance stopped improving
        """
        # execution_order=99 makes sure that any performance calculation callbacks are executed before and the most
        # recent results can already be found in the train_history
        AbstractCallback.__init__(self, 'EarlyStopping', execution_order=99)
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience

        self.patience_count = self.patience
        self.best_performance = None
        self.best_epoch = 0

    def on_epoch_end(self):
        history_data = self.train_loop_obj.train_history[self.monitor]
        current_performance = history_data[-1]

        if self.best_performance is None:
            self.best_performance = current_performance
            self.best_epoch = self.train_loop_obj.epoch
        else:
            if 'loss' in self.monitor.lower() or 'error' in self.monitor.lower():
                if current_performance < self.best_performance - self.min_delta:
                    self.best_performance = current_performance
                    self.best_epoch = self.train_loop_obj.epoch
                    self.patience_count = self.patience
                else:
                    self.patience_count -= 1
            else:
                if current_performance > self.best_performance + self.min_delta:
                    self.best_performance = current_performance
                    self.best_epoch = self.train_loop_obj.epoch
                    self.patience_count = self.patience
                else:
                    self.patience_count -= 1

            if self.patience_count < 0:
                self.train_loop_obj.early_stop = True
                print(f'Early stopping at epoch: {self.train_loop_obj.epoch}. Best recorded epoch: {self.best_epoch}.')


class TerminateOnNaN(AbstractCallback):
    def __init__(self, monitor='loss'):
        """Terminate training if NaNs are predicted, thus metrics are NaN

        Args:
            monitor (str): performance measure that is tracked to decide if performance is improving during training
        """
        AbstractCallback.__init__(self, 'TerminateOnNaN', execution_order=98)
        self.monitor = monitor

    def on_epoch_end(self):
        last_measure = self.train_loop_obj.train_history[self.monitor][-1]

        if last_measure is not None:
            if np.isnan(last_measure) or np.isinf(last_measure):
                self.train_loop_obj.early_stop = True
                print(f'Terminating on {self.monitor} = {last_measure} at epoch: {self.train_loop_obj.epoch}.')


class AllPredictionsSame(AbstractCallback):
    def __init__(self, value=0., stop_training=False, verbose=True):
        """Checks if all the predicted values are the same

        Useful for example when dealing with extremely unbalanced classes.

        Args:
            value (float): all predictions are the same as this value
            stop_training (bool): if all predictions match the specified value, should the training be (early) stopped
            verbose (bool): output messages
        """
        AbstractCallback.__init__(self, 'All predictions have the same value')
        self.value = value
        self.stop_training = stop_training
        self.verbose = verbose

    def on_epoch_end(self):
        predictions, _, _ = self.train_loop_obj.predict_on_validation_set()

        all_values_same = all(el == self.value for el in predictions)

        if all_values_same:
            if self.verbose:
                print(f'All the predicted values are of the same value: {self.value}')

            if self.stop_training:
                print('Executing early stopping')
                self.train_loop_obj.early_stop = True


class EmailNotification(AbstractCallback):
    def __init__(self, sender_name, sender_email, recipient_email,
                 project_name=None, experiment_name=None, aws_region='eu-west-1'):
        """Notify user via email about the training progression

        Args:
            sender_name (str): Name of the email sender
            sender_email (str): Email of the email sender
            recipient_email (str): Email where the email will be sent
            project_name (str or None): root name of the project
            experiment_name (str or None): name of the particular experiment
            aws_region (str): AWS SES region
        """
        AbstractCallback.__init__(self, 'Send email to notify about the state of training', execution_order=98)
        self.project_name = project_name
        self.experiment_name = experiment_name

        self.ses_sender = SESSender(sender_name, sender_email, recipient_email, aws_region)

    def on_epoch_end(self):
        subject = f"End of epoch {self.train_loop_obj.epoch} report: {self.project_name}: {self.experiment_name}"

        performance_list = self.get_metric_list_html()
        plots_file_paths = self.message_service.read_messages('ModelTrainHistoryPlot_results_file_local_paths')
        plots_file_paths += self.message_service.read_messages('ModelTrainHistoryFileWriter_results_file_local_paths')
        plots_file_paths = util.flatten_list_of_lists(plots_file_paths)

        body_text = f"""<h2>End of epoch {self.train_loop_obj.epoch}</h2>
        {performance_list}
        """

        self.ses_sender.send_email(subject, body_text, plots_file_paths)

    def on_train_end(self):
        subject = f"End of training: {self.project_name}: {self.experiment_name}"

        performance_list = self.get_metric_list_html()
        hyperparams = self.get_hyperparams_html()
        plots_file_paths = self.message_service.read_messages('ModelTrainHistoryPlot_results_file_local_paths')
        plots_file_paths += self.message_service.read_messages('ModelTrainHistoryFileWriter_results_file_local_paths')
        plots_file_paths = util.flatten_list_of_lists(plots_file_paths)

        body_text = f"""<h2>End of training at epoch {self.train_loop_obj.epoch}</h2>
                {performance_list}

                <h3>Used hyper parameters:</h3>
                {hyperparams}
                """

        self.ses_sender.send_email(subject, body_text, plots_file_paths)

    def get_metric_list_html(self):
        """

        Returns:
            str:
        """
        performance_list = '<ul>' + \
                           '\n'.join([f'<li><p>{metric_name}: {hist[-1]}</p></li>'
                                      for metric_name, hist in self.train_loop_obj.train_history.items()]) + \
                           '</ul>'

        return performance_list

    def get_hyperparams_html(self):
        """

        Returns:
            str:
        """
        hyperparams = '<ul>' + \
                      '\n'.join([f'<li><p>{param_name}: {val}</p></li>'
                                 for param_name, val in self.train_loop_obj.hyperparams.items()]) + \
                      '</ul>' \
            if hasattr(self.train_loop_obj, 'hyperparams') else 'Not given'

        return hyperparams

    def on_train_loop_registration(self):
        """

        Tries to infer the project description from the running train loop. If the train loop does not build
        the project folder structure (e.g. basic TrainLoop) the descriptions need to be provided manually to
        this callback.

        Returns:
            None
        """
        try:
            if self.project_name is None:
                self.project_name = self.train_loop_obj.project_name
            if self.experiment_name is None:
                self.experiment_name = self.train_loop_obj.experiment_name
        except AttributeError:
            raise AttributeError('Currently used TrainLoop does not support automatic project folder structure '
                                 'creation. Project name, etc. thus can not be automatically deduced. Please provide'
                                 'it in the callback parameters instead of currently used None values.')
