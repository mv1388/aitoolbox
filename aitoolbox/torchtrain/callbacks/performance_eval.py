import copy
import os

from aitoolbox.torchtrain.callbacks.abstract import AbstractCallback, AbstractExperimentCallback
from aitoolbox.torchtrain.train_loop.components.message_passing import MessageHandling
from aitoolbox.cloud.AWS.results_save import BaseResultsSaver as BaseResultsS3Saver
from aitoolbox.cloud.GoogleCloud.results_save import BaseResultsGoogleStorageSaver
from aitoolbox.cloud import s3_available_options, gcs_available_options
from aitoolbox.experiment.local_save.local_results_save import BaseLocalResultsSaver
from aitoolbox.experiment.result_reporting.report_generator import TrainingHistoryPlotter, TrainingHistoryWriter
from aitoolbox.experiment.result_package.torch_metrics_packages import TorchMetricsPackage


class ModelPerformanceEvaluation(AbstractCallback):
    def __init__(self, result_package, args,
                 on_each_epoch=True, on_train_data=False, on_val_data=True, eval_frequency=None,
                 if_available_output_to_project_dir=True):
        """Track performance metrics from result_package and store them into TrainLoop's history

        This callback is different from those for model and experiment saving where performance evaluations are also
        calculated. Here we only want to calculate performance and store it in memory into TrainLoop's history dict.

        It is a more lightweight, on the go performance tracking without the need for the full project folder structure
        construction.

        Args:
            result_package (aitoolbox.experiment.result_package.abstract_result_packages.AbstractResultPackage):
            args (dict): used hyper-parameters
            on_each_epoch (bool): calculate performance results just at the end of training or at the end of each epoch
            on_train_data (bool): should the evaluation be done on the training dataset
            on_val_data (bool): should the evaluation be done on the validation dataset
            eval_frequency (int or None): evaluation is done every specified number of epochs. Useful when predictions
                are quite expensive and are slowing down the overall training
            if_available_output_to_project_dir (bool): if using train loop version which builds project local folder
                structure for saving checkpoints or creation of end of training reports, by setting
                if_available_output_to_project_dir to True the potential additional metadata result outputs from the
                result_package will be saved in the folder inside the main project folder. In this case
                the result_package's output folder shouldn't be full path but just the folder name and the full folder
                path pointing inside the corresponding project folder will be automatically created.
                If such a functionality should to be prevented and manual full additional metadata results dump folder
                is needed potentially outside the project folder, then set this argument to False and
                specify a full folder path.
        """
        AbstractCallback.__init__(self, 'Model performance calculator - evaluator')
        self.result_package = result_package
        self.args = args
        self.on_each_epoch = on_each_epoch
        self.on_train_data = on_train_data
        self.on_val_data = on_val_data
        self.eval_frequency = eval_frequency
        self.if_available_output_to_project_dir = if_available_output_to_project_dir

        if not on_train_data and not on_val_data:
            raise ValueError('Both on_train_data and on_val_data are set to False. At least one of them has to be True')

        if on_train_data:
            self.train_result_package = copy.deepcopy(result_package)

    def on_train_end(self):
        self.evaluate_model_performance(prefix='train_end_')

    def on_epoch_end(self):
        if self.on_each_epoch:
            if self.eval_frequency is None or \
                    (self.eval_frequency is not None and self.train_loop_obj.epoch % self.eval_frequency == 0):
                self.evaluate_model_performance()
            else:
                print(f'Skipping performance evaluation on this epoch ({self.train_loop_obj.epoch}). '
                      f'Evaluating every {self.eval_frequency} epochs.')

        if isinstance(self.result_package, TorchMetricsPackage):
            if self.on_train_data:
                self.train_result_package.metric_reset()

            if self.on_val_data:
                self.result_package.metric_reset()

    def evaluate_model_performance(self, prefix=''):
        """Calculate performance based on the provided result packages

        Args:
            prefix (str): additional prefix for metric names that will get saved into the training history

        Returns:
            None
        """
        if self.on_train_data:
            y_pred, y_test, additional_results = self.train_loop_obj.predict_on_train_set()
            if self.train_result_package.requires_loss:
                additional_results['loss'] = self.train_loop_obj.evaluate_loss_on_train_set()
            self.train_result_package.prepare_result_package(y_test, y_pred,
                                                             hyperparameters=self.args,
                                                             additional_results=additional_results)

        if self.on_val_data:
            y_pred, y_test, additional_results = self.train_loop_obj.predict_on_validation_set()
            if self.result_package.requires_loss:
                additional_results['loss'] = self.train_loop_obj.evaluate_loss_on_validation_set()
            self.result_package.prepare_result_package(y_test, y_pred,
                                                       hyperparameters=self.args,
                                                       additional_results=additional_results)

        self.store_evaluated_metrics_to_history(prefix=prefix)

    def store_evaluated_metrics_to_history(self, prefix=''):
        """Save the calculated performance results into the training history

        The performance results are saved into the training history after they are calculated by the before called
        evaluate_model_performance() function.

        Args:
            prefix (str): additional prefix for metric names that will get saved into the training history

        Returns:
            None
        """
        evaluated_metrics = self.result_package.get_results().keys() if self.on_val_data \
            else self.train_result_package.get_results().keys()

        for m_name in evaluated_metrics:
            if self.on_train_data:
                metric_name = f'{prefix}train_{m_name}'
                self.train_loop_obj.insert_metric_result_into_history(metric_name,
                                                                      self.train_result_package.get_results()[m_name])

            if self.on_val_data:
                metric_name = f'{prefix}val_{m_name}'
                self.train_loop_obj.insert_metric_result_into_history(metric_name,
                                                                      self.result_package.get_results()[m_name])

    def on_train_loop_registration(self):
        if self.if_available_output_to_project_dir and \
            hasattr(self.train_loop_obj, 'project_name') and hasattr(self.train_loop_obj, 'experiment_name') and \
                hasattr(self.train_loop_obj, 'local_model_result_folder_path'):
            self.result_package.set_experiment_dir_path_for_additional_results(self.train_loop_obj.project_name,
                                                                               self.train_loop_obj.experiment_name,
                                                                               self.train_loop_obj.experiment_timestamp,
                                                                               self.train_loop_obj.local_model_result_folder_path)

        if isinstance(self.result_package, TorchMetricsPackage):
            if self.on_train_data:
                self.train_result_package.metric.to(self.train_loop_obj.device)

            if self.on_val_data:
                self.result_package.metric.to(self.train_loop_obj.device)


class ModelPerformancePrintReport(AbstractCallback):
    def __init__(self, metrics, on_each_epoch=True, report_frequency=None,
                 strict_metric_reporting=True, list_tracked_metrics=False):
        """Print the model performance to the console

        Best used in combination with the callback which actually calculates some performance evaluation metrics, such
        as ModelPerformanceEvaluation. Otherwise, we are limited only to automatic loss calculation reporting.

        When listing callbacks for the TrainLoop it is important to list the ModelPerformanceEvaluation before
        this ModelPerformancePrintReport. This ensures that the calculated results are present in the
        TrainLoop.train_history before there is an attempt to print them.

        Args:
            metrics (list): list of string metric names which should be presented in the printed report
            on_each_epoch (bool): present results just at the end of training or at the end of each epoch
            report_frequency (int or None): evaluation is done every specified number of epochs. Useful when predictions
                are quite expensive and are slowing down the overall training
            strict_metric_reporting (bool): if False ignore missing metric in the TrainLoop.train_history, if True, in
                case of missing metric throw and exception and thus interrupt the training loop
            list_tracked_metrics (bool): should all tracked metrics names be listed
        """
        AbstractCallback.__init__(self, 'Model performance print reporter',
                                  execution_order=97, device_idx_execution=0)
        self.metrics = metrics
        self.on_each_epoch = on_each_epoch
        self.report_frequency = report_frequency
        self.strict_metric_reporting = strict_metric_reporting
        self.list_tracked_metrics = list_tracked_metrics

        if len(metrics) == 0:
            raise ValueError('metrics list is empty')

    def on_train_end(self):
        print('-----------------  End of training performance report  -----------------')
        self.print_performance_report(prefix='train_end_')

    def on_epoch_end(self):
        if self.on_each_epoch:
            if self.report_frequency is None or \
                    (self.report_frequency is not None and self.train_loop_obj.epoch % self.report_frequency == 0):
                print('------------------  End of epoch performance report  -------------------')
                self.print_performance_report()

    def print_performance_report(self, prefix=''):
        """Print the model performance

        Args:
            prefix (str): additional prefix for metric names that will get saved into the training history

        Returns:
            None
        """
        if self.list_tracked_metrics:
            print(self.train_loop_obj.train_history.keys())

        for metric_name in self.metrics:
            metric_name = prefix + metric_name

            if metric_name not in self.train_loop_obj.train_history:
                if self.strict_metric_reporting:
                    raise ValueError(
                        f'Metric {metric_name} expected for the report missing from TrainLoop.train_history. '
                        f'Found only the following: {self.train_loop_obj.train_history.keys()}')
                else:
                    print(f'Metric {metric_name} expected for the report missing from TrainLoop.train_history. '
                          f'Found only the following: {self.train_loop_obj.train_history.keys()}')
            else:
                print(f'{metric_name}: {self.train_loop_obj.train_history[metric_name][-1]}')


class TrainHistoryFormatter(AbstractCallback):
    def __init__(self, input_metric_getter, output_metric_setter,
                 epoch_end=True, train_end=False, strict_metric_extract=True):
        """Format stored training history results

        Args:
            input_metric_getter (lambda): extract full history for the desired metric, not just the last history input.
                Return should be represented as a list.
            output_metric_setter (lambda): take the extracted full history of a metric and convert it as desired.
                Return new / transformed metric name and transformed metric result.
            epoch_end (bool): should the formatting be executed at the end of the epoch
            train_end (bool): should the formatting be executed at the end of the training process
            strict_metric_extract (bool): in case of (quality) problems should exception be raised on just the
                notification printed to console
        """
        if epoch_end == train_end:
            raise ValueError(f'Only either epoch_end or train_end have to be set to True. '
                             f'Have set epoch_end to {epoch_end} and train_end to {train_end}')

        AbstractCallback.__init__(self, 'Train history general formatter engine')
        self.input_metric_getter = input_metric_getter
        self.output_metric_setter = output_metric_setter

        self.epoch_end = epoch_end
        self.train_end = train_end
        self.strict_metric_extract = strict_metric_extract

    def on_epoch_end(self):
        if self.epoch_end:
            if self.check_if_history_updated(not self.epoch_end):
                self.format_history()

    def on_train_end(self):
        if self.train_end:
            if self.check_if_history_updated(self.train_end):
                self.format_history()

    def format_history(self):
        input_metric = self.input_metric_getter(self.train_loop_obj.train_history)
        output_metric_name, output_metric = self.output_metric_setter(input_metric)
        self.train_loop_obj.insert_metric_result_into_history(output_metric_name, output_metric)

    def check_if_history_updated(self, train_end_phase):
        if train_end_phase:
            history_elements_expected = 1
        else:
            history_elements_expected = self.train_loop_obj.epoch + 1
        metric_result_list = self.input_metric_getter(self.train_loop_obj.train_history)
        metric_result_len = len(metric_result_list)

        if history_elements_expected != metric_result_len:
            if self.strict_metric_extract:
                raise ValueError(f'Metric found at path specified in input_metric_getter not yet updated. '
                                 f'Expecting {history_elements_expected} history elements, '
                                 f'but got {metric_result_len} elements.')
            else:
                print(f'Metric found at path specified in input_metric_getter not yet updated. '
                      f'Expecting {history_elements_expected} history elements, but got {metric_result_len} elements.')
                return False
        return True


class MetricHistoryRename(TrainHistoryFormatter):
    def __init__(self, input_metric_path, new_metric_name,
                 epoch_end=True, train_end=False, strict_metric_extract=True):
        """Specific interface for TrainHistoryFormatter which renames the metric in the training history

        Args:
            input_metric_path (str or lambda): if using lambda, extract full history for the desired metric,
                not just the last history input. Return should be represented as a list.
            new_metric_name (str): the new metric name
            epoch_end (bool): should the formatting be executed at the end of the epoch
            train_end (bool): should the formatting be executed at the end of the training process
            strict_metric_extract (bool): in case of (quality) problems should exception be raised on just the
                notification printed to console
        """

        # TODO: decide which of these two options is better

        # if callable(input_metric_path):
        #     input_metric_getter = input_metric_path
        # else:
        #     input_metric_getter = lambda train_history: train_history[input_metric_path]

        # input_metric_getter = input_metric_path if callable(input_metric_path) \
        #     else lambda train_history: train_history[input_metric_path]
        # output_metric_setter = lambda input_metric: (new_metric_name, input_metric[-1])

        # TrainHistoryFormatter.__init__(self, input_metric_getter, output_metric_setter,
        #                                epoch_end=True, train_end=True, strict_metric_extract=strict_metric_extract)

        TrainHistoryFormatter.__init__(self,
                                       input_metric_getter=input_metric_path if callable(input_metric_path) else
                                       lambda train_history: train_history[input_metric_path],
                                       output_metric_setter=lambda input_metric: (new_metric_name, input_metric[-1]),
                                       epoch_end=epoch_end, train_end=train_end,
                                       strict_metric_extract=strict_metric_extract)


class ModelTrainHistoryBaseCB(AbstractExperimentCallback):
    def __init__(self, callback_name, execution_order=0,
                 epoch_end=True, train_end=False, file_format='',
                 project_name=None, experiment_name=None, local_model_result_folder_path=None,
                 cloud_save_mode=None, bucket_name=None, cloud_dir_prefix=None):
        """Base callback class to be inherited from when reporting train performance history

        Args:
            callback_name (str): name of the callback
            execution_order (int): order of the callback execution. If all the used callbacks have the orders set to 0,
                then the callbacks are executed in the order they were registered.
            epoch_end (bool): should plot after every epoch
            train_end (bool): should plot at the end of the training
            file_format (str): output file format
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
        AbstractExperimentCallback.__init__(self, callback_name,
                                            project_name, experiment_name, local_model_result_folder_path,
                                            cloud_save_mode, bucket_name, cloud_dir_prefix,
                                            execution_order=execution_order, device_idx_execution=0)
        if epoch_end is False and train_end is False:
            raise ValueError('Both epoch_end and train_end are set to False. At least one of these should be True.')
        self.epoch_end = epoch_end
        self.train_end = train_end
        self.file_format = file_format

        self.cloud_results_saver = None

    def prepare_results_saver(self):
        """Initialize the required results saver

        Returns:
            None
        """
        if self.cloud_save_mode in s3_available_options:
            self.cloud_results_saver = BaseResultsS3Saver(bucket_name=self.bucket_name,
                                                          cloud_dir_prefix=self.cloud_dir_prefix)

        elif self.cloud_save_mode in gcs_available_options:
            self.cloud_results_saver = BaseResultsGoogleStorageSaver(bucket_name=self.bucket_name,
                                                                     cloud_dir_prefix=self.cloud_dir_prefix)
        else:
            self.cloud_results_saver = None


class ModelTrainHistoryPlot(ModelTrainHistoryBaseCB):
    def __init__(self, epoch_end=True, train_end=False, file_format='png',
                 project_name=None, experiment_name=None, local_model_result_folder_path=None,
                 cloud_save_mode=None, bucket_name=None, cloud_dir_prefix=None):
        """Plot the evaluated performance metric history

        Args:
            epoch_end (bool): should plot after every epoch
            train_end (bool): should plot at the end of the training
            file_format (str): output file format. Can be either 'png' for saving separate images or 'pdf' for combining
                all the plots into a single pdf file.
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
        # execution_order=97 makes sure that any performance calculation callbacks are executed before and the most
        # recent results can already be found in the train_history
        ModelTrainHistoryBaseCB.__init__(self, 'Model Train history Plot report', execution_order=97,
                                         epoch_end=epoch_end, train_end=train_end, file_format=file_format,
                                         project_name=project_name, experiment_name=experiment_name,
                                         local_model_result_folder_path=local_model_result_folder_path,
                                         cloud_save_mode=cloud_save_mode, bucket_name=bucket_name,
                                         cloud_dir_prefix=cloud_dir_prefix)
        if self.file_format not in ['png', 'pdf']:
            raise ValueError(f"Output format '{self.file_format}' is not supported. "
                             "Select one of the following: 'png' or 'pdf'.")

    def on_train_loop_registration(self):
        self.try_infer_experiment_details(infer_cloud_details=True)
        self.prepare_results_saver()

    def on_epoch_end(self):
        if self.epoch_end:
            self.plot_current_train_history()

    def on_train_end(self):
        if self.train_end:
            self.plot_current_train_history(prefix='train_end_')

    def plot_current_train_history(self, prefix=''):
        """Plot current training history snapshot in the encapsulating TrainLoop

        Args:
            prefix (str): plots folder name prefix

        Returns:
            None
        """
        experiment_results_local_path = \
            BaseLocalResultsSaver.create_experiment_local_results_folder(self.project_name, self.experiment_name,
                                                                         self.train_loop_obj.experiment_timestamp,
                                                                         self.local_model_result_folder_path)

        plotter = TrainingHistoryPlotter(experiment_results_local_path=experiment_results_local_path)
        saved_local_results_details = \
            plotter.generate_report(training_history=self.train_loop_obj.train_history,
                                    plots_folder_name=f'{prefix}plots_epoch_{self.train_loop_obj.epoch}',
                                    file_format=self.file_format)

        results_file_local_paths = [result_local_path for _, result_local_path in saved_local_results_details]
        self.message_service.write_message('ModelTrainHistoryPlot_results_file_local_paths',
                                           results_file_local_paths,
                                           msg_handling_settings=MessageHandling.UNTIL_END_OF_EPOCH)

        if self.cloud_results_saver is not None:
            experiment_cloud_path = \
                self.cloud_results_saver.create_experiment_cloud_storage_folder_structure(self.project_name,
                                                                                          self.experiment_name,
                                                                                          self.train_loop_obj.experiment_timestamp)

            for results_file_path_in_cloud_results_dir, results_file_local_path in saved_local_results_details:
                results_file_s3_path = os.path.join(experiment_cloud_path, results_file_path_in_cloud_results_dir)
                self.cloud_results_saver.save_file(local_file_path=results_file_local_path,
                                                   cloud_file_path=results_file_s3_path)


class ModelTrainHistoryFileWriter(ModelTrainHistoryBaseCB):
    def __init__(self, epoch_end=True, train_end=False, file_format='txt',
                 project_name=None, experiment_name=None, local_model_result_folder_path=None,
                 cloud_save_mode=None, bucket_name=None, cloud_dir_prefix=None):
        """Write evaluated performance metric history to the text file

        Args:
            epoch_end (bool): should plot after every epoch
            train_end (bool): should plot at the end of the training
            file_format (str): output file format. Can be either 'txt' human-readable output or
                'tsv' for a tabular format or 'csv' for comma separated format.
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
        # execution_order=97 makes sure that any performance calculation callbacks are executed before and the most
        # recent results can already be found in the train_history
        ModelTrainHistoryBaseCB.__init__(self, 'Model Train performance history file writer', execution_order=97,
                                         epoch_end=epoch_end, train_end=train_end, file_format=file_format,
                                         project_name=project_name, experiment_name=experiment_name,
                                         local_model_result_folder_path=local_model_result_folder_path,
                                         cloud_save_mode=cloud_save_mode, bucket_name=bucket_name,
                                         cloud_dir_prefix=cloud_dir_prefix)
        # experiment_results_local_path will be set when callback is executed inside write_current_train_history()
        self.result_writer = TrainingHistoryWriter(experiment_results_local_path=None)
        if self.file_format not in ['txt', 'tsv', 'csv']:
            raise ValueError(f"Output format '{self.file_format}' is not supported. "
                             "Select one of the following: 'txt', 'tsv' or 'csv'.")

    def on_train_loop_registration(self):
        self.try_infer_experiment_details(infer_cloud_details=True)
        self.prepare_results_saver()

    def on_epoch_end(self):
        if self.epoch_end:
            self.write_current_train_history()

    def on_train_end(self):
        if self.train_end:
            self.write_current_train_history(prefix='train_end_')

    def write_current_train_history(self, prefix=''):
        """Write to text file the current training history snapshot in the encapsulating TrainLoop

        Args:
            prefix (str): history text file name prefix

        Returns:
            None
        """
        experiment_results_local_path = \
            BaseLocalResultsSaver.create_experiment_local_results_folder(self.project_name, self.experiment_name,
                                                                         self.train_loop_obj.experiment_timestamp,
                                                                         self.local_model_result_folder_path)
        self.result_writer.experiment_results_local_path = experiment_results_local_path

        results_file_path_in_cloud_results_dir, results_file_local_path = \
            self.result_writer.generate_report(training_history=self.train_loop_obj.train_history,
                                               epoch=self.train_loop_obj.epoch,
                                               file_name=f'{prefix}results.{self.file_format}',
                                               file_format=self.file_format)

        self.message_service.write_message('ModelTrainHistoryFileWriter_results_file_local_paths',
                                           [results_file_local_path],
                                           msg_handling_settings=MessageHandling.UNTIL_END_OF_EPOCH)

        if self.cloud_results_saver is not None:
            experiment_cloud_path = \
                self.cloud_results_saver.create_experiment_cloud_storage_folder_structure(self.project_name,
                                                                                          self.experiment_name,
                                                                                          self.train_loop_obj.experiment_timestamp)

            results_file_s3_path = os.path.join(experiment_cloud_path, results_file_path_in_cloud_results_dir)
            self.cloud_results_saver.save_file(local_file_path=results_file_local_path,
                                               cloud_file_path=results_file_s3_path)
