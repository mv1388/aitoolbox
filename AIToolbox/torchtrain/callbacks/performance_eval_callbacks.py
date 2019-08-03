import copy
import os

from AIToolbox.torchtrain.callbacks.callbacks import AbstractCallback
from AIToolbox.torchtrain.tl_components import message_passing as msg_passing_settings
from AIToolbox.cloud.AWS.results_save import BaseResultsSaver as BaseResultsS3Saver
from AIToolbox.cloud.GoogleCloud.results_save import BaseResultsGoogleStorageSaver
from AIToolbox.experiment.local_save.local_results_save import BaseLocalResultsSaver
from AIToolbox.experiment.result_package.abstract_result_packages import PreCalculatedResultPackage as EmptyResultPackage
from AIToolbox.experiment.result_reporting.report_generator import TrainingHistoryPlotter


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
            result_package (AIToolbox.experiment_save.result_package.abstract_result_packages.AbstractResultPackage):
            args (dict):
            on_each_epoch (bool): calculate performance results just at the end of training or at the end of each epoch
            on_train_data (bool):
            on_val_data (bool):
            eval_frequency (int or None): evaluation is done every specified number of epochs. Useful when predictions
                are quite expensive and are slowing down the overall training
            if_available_output_to_project_dir (bool): if using train loop version which builds project local folder
                structure for saving checkpoints or creation of end of training reports, by setting
                if_available_output_to_project_dir to True the potential additional metadata result outputs from the
                result_package will be saved in the folder inside the main project folder. In this case
                the result_package's output folder shouldn't be full path but just the folder name and the full folder
                path pointing inside the corresponding project folder will be automatically created.
                If such a functionality should to be prevented and manual full additional metadata results dump folder
                is needed potentially outside the project folder, than set this argument to False and
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
        self.evaluate_model_performance()
        self.store_evaluated_metrics_to_history(prefix='train_end_')

    def on_epoch_end(self):
        if self.on_each_epoch:
            if self.eval_frequency is None or \
                    (self.eval_frequency is not None and self.train_loop_obj.epoch % self.eval_frequency == 0):
                self.evaluate_model_performance()
                self.store_evaluated_metrics_to_history()
            else:
                print(f'Skipping performance evaluation on this epoch ({self.train_loop_obj.epoch}). '
                      f'Evaluating every {self.eval_frequency} epochs.')

    def evaluate_model_performance(self):
        """Calculate performance based on the provided result packages

        Returns:
            None
        """
        if self.on_train_data:
            y_test, y_pred, additional_results = self.train_loop_obj.predict_on_train_set()
            self.train_result_package.prepare_result_package(y_test, y_pred,
                                                             hyperparameters=self.args,
                                                             training_history=self.train_loop_obj.train_history,
                                                             additional_results=additional_results)

        if self.on_val_data:
            y_test, y_pred, additional_results = self.train_loop_obj.predict_on_validation_set()
            self.result_package.prepare_result_package(y_test, y_pred,
                                                       hyperparameters=self.args,
                                                       training_history=self.train_loop_obj.train_history,
                                                       additional_results=additional_results)

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


class ModelPerformancePrintReport(AbstractCallback):
    def __init__(self, metrics, on_each_epoch=True, report_frequency=None,
                 strict_metric_reporting=True, list_tracked_metrics=False):
        """Print the model performance to the console

        Best used in combination with the callback which actually calculates some performance evaluation metrics, such
        as ModelPerformanceEvaluation. Otherwise we are limited only to automatic loss calculation reporting.

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
            list_tracked_metrics (bool):
        """
        AbstractCallback.__init__(self, 'Model performance print reporter')
        self.metrics = metrics
        self.on_each_epoch = on_each_epoch
        self.report_frequency = report_frequency
        self.strict_metric_reporting = strict_metric_reporting
        self.list_tracked_metrics = list_tracked_metrics

        if len(metrics) == 0:
            raise ValueError('metrics list is empty')

    def on_train_end(self):
        print('------------  End of training performance report  ------------')
        self.print_performance_report(prefix='train_end_')

    def on_epoch_end(self):
        if self.on_each_epoch:
            if self.report_frequency is None or \
                    (self.report_frequency is not None and self.train_loop_obj.epoch % self.report_frequency == 0):
                print('------------  End of epoch performance report  ------------')
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


class ModelTrainHistoryPlot(AbstractCallback):
    def __init__(self, epoch_end=True, train_end=False,
                 project_name=None, experiment_name=None, local_model_result_folder_path=None,
                 cloud_save_mode='s3', bucket_name='model-result'):
        """

        Args:
            epoch_end (bool):
            train_end (bool):
            project_name (str or None): root name of the project
            experiment_name (str or None): name of the particular experiment
            local_model_result_folder_path (str or None): root local path where project folder will be created
            cloud_save_mode (str or None): Storage destination selector.
                For AWS S3: 's3' / 'aws_s3' / 'aws'
                For Google Cloud Storage: 'gcs' / 'google_storage' / 'google storage'
                Everything else results just in local storage to disk
            bucket_name (str): name of the bucket in the cloud storage
        """
        if epoch_end is False and train_end is False:
            raise ValueError('Both epoch_end and train_end are set to False. At least one of these should be True.')
        # execution_order=98 makes sure that any performance calculation callbacks are executed before and the most
        # recent results can already be found in the train_history
        AbstractCallback.__init__(self, 'Model Train history Plot report', execution_order=97)
        self.epoch_end = epoch_end
        self.train_end = train_end
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.local_model_result_folder_path = os.path.expanduser(local_model_result_folder_path) \
            if local_model_result_folder_path is not None \
            else None
        self.cloud_save_mode = cloud_save_mode
        self.bucket_name = bucket_name

        self.cloud_results_saver = None

    def on_train_loop_registration(self):
        self.try_infer_experiment_details()
        self.prepare_results_saver()

    def try_infer_experiment_details(self):
        """

        Returns:
            None

        Raises:
            AttributeError
        """
        try:
            if self.project_name is None:
                self.project_name = self.train_loop_obj.project_name
            if self.experiment_name is None:
                self.experiment_name = self.train_loop_obj.experiment_name
            if self.local_model_result_folder_path is None:
                self.local_model_result_folder_path = self.train_loop_obj.local_model_result_folder_path
        except AttributeError:
            raise AttributeError('Currently used TrainLoop does not support automatic project folder structure '
                                 'creation. Project name, etc. thus can not be automatically deduced. Please provide'
                                 'it in the callback parameters instead of currently used None values.')

    def prepare_results_saver(self):
        """

        Returns:
            None
        """
        if self.cloud_save_mode == 's3' or self.cloud_save_mode == 'aws_s3' or self.cloud_save_mode == 'aws':
            self.cloud_results_saver = BaseResultsS3Saver(bucket_name=self.bucket_name)

        elif self.cloud_save_mode == 'gcs' or self.cloud_save_mode == 'google_storage' or self.cloud_save_mode == 'google storage':
            self.cloud_results_saver = BaseResultsGoogleStorageSaver(bucket_name=self.bucket_name)
        else:
            self.cloud_results_saver = None

    def on_epoch_end(self):
        if self.epoch_end:
            self.plot_current_train_history()

    def on_train_end(self):
        if self.train_end:
            self.plot_current_train_history(prefix='train_end_')

    def plot_current_train_history(self, prefix=''):
        """

        Args:
            prefix (str):

        Returns:
            None
        """
        experiment_results_local_path = \
            BaseLocalResultsSaver.create_experiment_local_results_folder(self.project_name, self.experiment_name,
                                                                         self.train_loop_obj.experiment_timestamp,
                                                                         self.local_model_result_folder_path)

        # Just a dummy empty result package to wrap the train history as RP is expected in the plotter
        result_pkg_wrapper = EmptyResultPackage(results_dict={})
        result_pkg_wrapper.training_history = self.train_loop_obj.train_history

        plotter = TrainingHistoryPlotter(result_package=result_pkg_wrapper,
                                         experiment_results_local_path=experiment_results_local_path,
                                         plots_folder_name=f'{prefix}plots_epoch_{self.train_loop_obj.epoch}')
        saved_local_results_details = plotter.generate_report()

        results_file_local_paths = [result_local_path for _, result_local_path in saved_local_results_details]
        self.message_service.write_message('ModelTrainHistoryPlot_results_file_local_paths',
                                           results_file_local_paths,
                                           msg_handling_settings=msg_passing_settings.UNTIL_END_OF_EPOCH)

        if self.cloud_results_saver is not None:
            experiment_cloud_path = \
                self.cloud_results_saver.create_experiment_cloud_storage_folder_structure(self.project_name,
                                                                                          self.experiment_name,
                                                                                          self.train_loop_obj.experiment_timestamp)

            for results_file_path_in_cloud_results_dir, results_file_local_path in saved_local_results_details:
                results_file_s3_path = os.path.join(experiment_cloud_path, results_file_path_in_cloud_results_dir)
                self.cloud_results_saver.save_file(local_file_path=results_file_local_path,
                                                   cloud_file_path=results_file_s3_path)


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
            strict_metric_extract (bool):
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
            strict_metric_extract (bool):
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
