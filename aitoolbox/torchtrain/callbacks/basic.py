import os
import numpy as np

from aitoolbox.torchtrain.callbacks.abstract import AbstractCallback, AbstractExperimentCallback
from aitoolbox.utils import util
from aitoolbox.cloud.AWS.simple_email_service import SESSender
from aitoolbox.cloud.AWS.results_save import BaseResultsSaver
from aitoolbox.cloud.GoogleCloud.results_save import BaseResultsGoogleStorageSaver
from aitoolbox.cloud import s3_available_options, gcs_available_options


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


class ThresholdEarlyStopping(AbstractCallback):
    def __init__(self, monitor, threshold, patience=0):
        """Early stopping of the training if the performance doesn't reach the specified threshold

        Args:
            monitor (str): performance measure that is tracked to decide if performance reached the desired threshold
            threshold (float): performance threshold that needs to be exceeded in order to continue training
            patience (int): how many epochs the early stopper waits for the tracked performance to reach the desired
                threshold
        """
        AbstractCallback.__init__(self, 'Threshold early stopping', execution_order=99)
        self.monitor = monitor
        self.threshold = threshold

        self.patience = patience
        self.patience_count = self.patience

    def on_epoch_end(self):
        history_data = self.train_loop_obj.train_history[self.monitor]
        current_performance = history_data[-1]

        if 'loss' in self.monitor.lower() or 'error' in self.monitor.lower():
            if current_performance > self.threshold:
                self.patience_count -= 1
            else:
                self.patience_count = self.patience
        else:
            if current_performance < self.threshold:
                self.patience_count -= 1
            else:
                self.patience_count = self.patience

        if self.patience_count < 0:
            self.train_loop_obj.early_stop = True
            print('Threshold performance not reached. '
                  f'Early stopping at epoch: {self.train_loop_obj.epoch}.')


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
        AbstractCallback.__init__(self, 'Send email to notify about the state of training',
                                  execution_order=98, device_idx_execution=0)
        self.project_name = project_name
        self.experiment_name = experiment_name

        self.ses_sender = None
        self.sender_name = sender_name
        self.sender_email = sender_email
        self.recipient_email = recipient_email
        self.aws_region = aws_region

    def on_epoch_end(self):
        subject = f"End of epoch {self.train_loop_obj.epoch} report: {self.project_name}: {self.experiment_name}"
        performance_list = self.get_metric_list_html()

        plot_msgs = self.message_service.read_messages('ModelTrainHistoryPlot_results_file_local_paths')
        plots_file_paths = plot_msgs if plot_msgs is not None else []
        file_writer_msgs = self.message_service.read_messages('ModelTrainHistoryFileWriter_results_file_local_paths')
        plots_file_paths += file_writer_msgs if file_writer_msgs is not None else []
        plots_file_paths = util.flatten_list_of_lists(plots_file_paths)

        body_text = f"""<h2>End of epoch {self.train_loop_obj.epoch}</h2>
        {performance_list}
        """

        self.ses_sender.send_email(subject, body_text, plots_file_paths)

    def on_train_end(self):
        subject = f"End of training: {self.project_name}: {self.experiment_name}"
        performance_list = self.get_metric_list_html()
        hyperparams = self.get_hyperparams_html()

        plot_msgs = self.message_service.read_messages('ModelTrainHistoryPlot_results_file_local_paths')
        plots_file_paths = plot_msgs if plot_msgs is not None else []
        file_writer_msgs = self.message_service.read_messages('ModelTrainHistoryFileWriter_results_file_local_paths')
        plots_file_paths += file_writer_msgs if file_writer_msgs is not None else []
        plots_file_paths = util.flatten_list_of_lists(plots_file_paths)

        body_text = f"""<h2>End of training at epoch {self.train_loop_obj.epoch}</h2>
                {performance_list}

                <h3>Used hyper parameters:</h3>
                {hyperparams}
                """

        self.ses_sender.send_email(subject, body_text, plots_file_paths)

    def get_metric_list_html(self):
        """Generate performance metrics list HTML

        Returns:
            str: HTML doc
        """
        performance_list = '<ul>' + \
                           '\n'.join([f'<li><p>{metric_name}: {hist[-1]}</p></li>'
                                      for metric_name, hist in self.train_loop_obj.train_history.items()
                                      if len(hist) > 0]) + \
                           '</ul>'

        return performance_list

    def get_hyperparams_html(self):
        """Generate hyperparameters list HTML

        Returns:
            str: HTML doc
        """
        hyperparams = '<ul>' + \
                      '\n'.join([f'<li><p>{param_name}: {val}</p></li>'
                                 for param_name, val in self.train_loop_obj.hyperparams.items()]) + \
                      '</ul>' \
            if hasattr(self.train_loop_obj, 'hyperparams') else 'Not given'

        return hyperparams

    def on_train_loop_registration(self):
        try:
            if self.project_name is None:
                self.project_name = self.train_loop_obj.project_name
            if self.experiment_name is None:
                self.experiment_name = self.train_loop_obj.experiment_name
        except AttributeError:
            raise AttributeError('Currently used TrainLoop does not support automatic project folder structure '
                                 'creation. Project name, etc. thus can not be automatically deduced. Please provide'
                                 'it in the callback parameters instead of currently used None values.')

        self.ses_sender = SESSender(self.sender_name, self.sender_email, self.recipient_email, self.aws_region)


class LogUpload(AbstractExperimentCallback):
    def __init__(self, log_file_path='~/project/training.log', fail_if_cloud_missing=True,
                 project_name=None, experiment_name=None, local_model_result_folder_path=None,
                 cloud_save_mode=None, bucket_name=None, cloud_dir_prefix=None):
        """Upload logging file to the cloud storage

        Uploading happens after each epoch and at the end of the training process.

        Args:
            log_file_path (str): path to the local logging file
            fail_if_cloud_missing (bool): should throw the exception if cloud saving is not available
            project_name (str or None): root name of the project
            experiment_name (str or None): name of the particular experiment
            local_model_result_folder_path (str or None): root local path where project folder will be created
            cloud_save_mode (str or None): Storage destination selector.
                For AWS S3: 's3' / 'aws_s3' / 'aws'
                For Google Cloud Storage: 'gcs' / 'google_storage' / 'google storage'
                Everything else results just in local storage to disk
            bucket_name (str): name of the bucket in the cloud storage
            cloud_dir_prefix (str): path to the folder inside the bucket where the experiments are going to be saved
        """
        AbstractExperimentCallback.__init__(self, "Upload Tmux logging file to cloud",
                                            project_name, experiment_name, local_model_result_folder_path,
                                            cloud_save_mode, bucket_name, cloud_dir_prefix,
                                            execution_order=1500, device_idx_execution=0)
        self.log_file_path = os.path.expanduser(log_file_path)
        self.log_filename = os.path.basename(self.log_file_path)
        self.fail_if_cloud_missing = fail_if_cloud_missing

        self.cloud_saver = None

    def on_train_loop_registration(self):
        self.try_infer_experiment_details(infer_cloud_details=True)

        if self.cloud_save_mode in s3_available_options:
            self.cloud_saver = BaseResultsSaver(bucket_name=self.bucket_name, cloud_dir_prefix=self.cloud_dir_prefix)

        elif self.cloud_save_mode in gcs_available_options:
            self.cloud_saver = BaseResultsGoogleStorageSaver(bucket_name=self.bucket_name,
                                                             cloud_dir_prefix=self.cloud_dir_prefix)
        else:
            if self.fail_if_cloud_missing:
                raise ValueError("Cloud saving not supported. Produced logs can potentially get los in the case of "
                                 "instance termination.")
            print("Cloud saving not supported. Produced logs can potentially get lost.")

    def on_epoch_end(self):
        self.upload_log_file()

    def on_train_end(self):
        self.upload_log_file()

    def upload_log_file(self):
        experiment_results_cloud_path = \
            self.cloud_saver.create_experiment_cloud_storage_folder_structure(self.project_name,
                                                                              self.experiment_name,
                                                                              self.train_loop_obj.experiment_timestamp)
        experiment_cloud_path = os.path.dirname(experiment_results_cloud_path)

        self.cloud_saver.save_file(local_file_path=self.log_file_path,
                                   cloud_file_path=os.path.join(experiment_cloud_path, self.log_filename))


class DataSubsetTestRun(AbstractCallback):
    def __init__(self, num_train_batches=1, num_val_batches=0, num_test_batches=0):
        """Subset the provided data loaders to execute neural net only on a small dataset subset

        This is especially useful when first developing the neural architectures and debugging them. Subsetting the full
        dataset helps with fast development iterations.

        Args:
            num_train_batches (int): number of the training data batches that are kept in the training dataset
            num_val_batches (int): number of the validation data batches that are kept in the validation dataset
            num_test_batches (int): number of the test data batches that are kept in the test dataset
        """
        AbstractCallback.__init__(self, 'Run model on a small subset of the training dataset')
        self.num_train_batches = num_train_batches
        self.num_val_batches = num_val_batches
        self.num_test_batches = num_test_batches

    def on_train_begin(self):
        self.train_loop_obj.train_loader = self.subset_data_loader(self.train_loop_obj.train_loader,
                                                                   self.num_train_batches)

        if self.num_val_batches > 0:
            self.train_loop_obj.validation_loader = self.subset_data_loader(self.train_loop_obj.validation_loader,
                                                                            self.num_val_batches)
        if self.num_test_batches > 0:
            self.train_loop_obj.test_loader = self.subset_data_loader(self.train_loop_obj.test_loader,
                                                                      self.num_test_batches)

    @staticmethod
    def subset_data_loader(data_loader, num_batches):
        sub_set = []

        for i, batch in enumerate(data_loader):
            sub_set.append(batch)

            if i == num_batches - 1:
                break

        return sub_set

    def on_train_loop_registration(self):
        if self.num_val_batches > 0 and self.train_loop_obj.validation_loader is None:
            raise ValueError("Validation loader was not provided to the TrainLoop, can't subset it")

        if self.num_test_batches > 0 and self.train_loop_obj.test_loader is None:
            raise ValueError("Test loader was not provided to the TrainLoop, can't subset it")


class FunctionOnTrainLoop(AbstractCallback):
    def __init__(self, fn_to_execute,
                 tl_registration=False,
                 epoch_begin=False, epoch_end=False, train_begin=False, train_end=False, batch_begin=False, batch_end=False,
                 after_gradient_update=False, after_optimizer_step=False, execution_order=0, device_idx_execution=None):
        """Execute given function as a callback in the TrainLoop

        Args:
            fn_to_execute (function): function logic to be executed at the desired point of the TrainLoop.
                The function should take a single input as an argument which is the reference to the encapsulating
                TrainLoop object (self.train_loop_obj).
            tl_registration (bool): should execute on TrainLoop registration
            epoch_begin (bool): should execute at the beginning of the epoch
            epoch_end (bool): should execute at the end of the epoch
            train_begin (bool): should execute at the beginning of the training
            train_end (bool): should execute at the end of the training
            batch_begin (bool): should execute at the beginning of the batch
            batch_end (bool): should execute at the end of the batch
            after_gradient_update (bool): should execute after the gradient update
            after_optimizer_step (bool): should execute after the optimizer step
            execution_order (int): order of the callback execution. If all the used callbacks have the orders set to 0,
                then the callbacks are executed in the order they were registered.
        """
        AbstractCallback.__init__(self, 'Execute given function as a callback', execution_order, device_idx_execution)
        self.fn_to_execute = fn_to_execute
        self.tl_registration = tl_registration
        self.epoch_begin = epoch_begin
        self.epoch_end = epoch_end
        self.train_begin = train_begin
        self.train_end = train_end
        self.batch_begin = batch_begin
        self.batch_end = batch_end
        self.after_gradient_update = after_gradient_update
        self.after_optimizer_step = after_optimizer_step

    def execute_callback(self):
        self.fn_to_execute(self.train_loop_obj)

    def on_train_loop_registration(self):
        if self.after_gradient_update or self.after_optimizer_step:
            self.train_loop_obj.grad_cb_used = True

        if self.tl_registration:
            self.execute_callback()

    def on_epoch_begin(self):
        if self.epoch_begin:
            self.execute_callback()

    def on_epoch_end(self):
        if self.epoch_end:
            self.execute_callback()

    def on_train_begin(self):
        if self.train_begin:
            self.execute_callback()

    def on_train_end(self):
        if self.train_end:
            self.execute_callback()

    def on_batch_begin(self):
        if self.batch_begin:
            self.execute_callback()

    def on_batch_end(self):
        if self.batch_end:
            self.execute_callback()

    def on_after_gradient_update(self, optimizer_idx):
        if self.after_gradient_update:
            self.execute_callback()

    def on_after_optimizer_step(self):
        if self.after_optimizer_step:
            self.execute_callback()
