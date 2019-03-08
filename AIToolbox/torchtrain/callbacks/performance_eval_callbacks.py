import copy

from AIToolbox.torchtrain.callbacks.callbacks import AbstractCallback
from AIToolbox.experiment_save.training_history import TrainingHistory


class ModelPerformanceEvaluationCallback(AbstractCallback):
    def __init__(self, result_package, args,
                 on_each_epoch=True, on_train_data=False, on_val_data=True):
        """

        Args:
            result_package (AIToolbox.experiment_save.result_package.AbstractResultPackage):
            args (dict):
            on_each_epoch (bool): calculate performance results just at the end of training or at the end of each epoch
            on_train_data (bool):
            on_val_data (bool):
        """
        AbstractCallback.__init__(self, 'Model performance calculator - evaluator')
        self.result_package = result_package
        self.args = args
        self.on_each_epoch = on_each_epoch
        self.on_train_data = on_train_data
        self.on_val_data = on_val_data

        if not on_train_data and not on_val_data:
            raise ValueError('Both on_train_data and on_val_data are set to False. At least one of them has to be True')

        if on_train_data:
            self.train_result_package = copy.deepcopy(result_package)

    def on_train_end(self):
        self.evaluate_model_performance()

    def on_epoch_end(self):
        if self.on_each_epoch:
            self.evaluate_model_performance()

            evaluated_metrics = self.result_package.get_results().keys() if self.on_val_data \
                else self.train_result_package.get_results().keys()

            for m_name in evaluated_metrics:
                if self.on_train_data:
                    metric_name = f'train_{m_name}'
                    # TODO: Test this
                    # if metric_name not in self.train_loop_obj.train_history:
                    #     self.train_loop_obj.train_history[metric_name] = []
                    # self.train_loop_obj.train_history[metric_name].append(self.train_result_package.get_results()[m_name])
                    self.train_loop_obj.insert_metric_result_into_history(metric_name,
                                                                          self.train_result_package.get_results()[m_name])

                if self.on_val_data:
                    metric_name = f'val_{m_name}'
                    # TODO: Test this
                    # if metric_name not in self.train_loop_obj.train_history:
                    #     self.train_loop_obj.train_history[metric_name] = []
                    # self.train_loop_obj.train_history[metric_name].append(self.result_package.get_results()[m_name])
                    self.train_loop_obj.insert_metric_result_into_history(metric_name,
                                                                          self.result_package.get_results()[m_name])

    def evaluate_model_performance(self):
        # TODO: maybe remove these 3 lines to save compute time and don't generate the train history which is not needed
        train_history = self.train_loop_obj.train_history
        epoch_list = list(
            range(len(self.train_loop_obj.train_history[list(self.train_loop_obj.train_history.keys())[0]])))
        train_hist_pkg = TrainingHistory(train_history, epoch_list)

        if self.on_train_data:
            y_test, y_pred = self.train_loop_obj.predict_on_train_set()
            self.train_result_package.prepare_result_package(y_test, y_pred,
                                                             hyperparameters=self.args, training_history=train_hist_pkg)
            print(f'TRAIN: {self.train_result_package.get_results()}')

        if self.on_val_data:
            y_test, y_pred = self.train_loop_obj.predict_on_validation_set()
            self.result_package.prepare_result_package(y_test, y_pred,
                                                       hyperparameters=self.args, training_history=train_hist_pkg)
            print(f'VAL: {self.result_package.get_results()}')


class ModelPerformancePrintReportCallback(AbstractCallback):
    def __init__(self, metrics, on_each_epoch=True, strict_metric_reporting=False):
        """

        Best used in combination with the callback which actually calculates some performance evaluation metrics, such
        as ModelPerformanceEvaluationCallback. Otherwise we are limited only to automatic loss calculation reporting.

        When listing callbacks for the TrainLoop it is important to list the ModelPerformanceEvaluationCallback before
        this ModelPerformancePrintReportCallback. This ensures that the calculated results are present in the
        TrainLoop.train_history before there is an attempt to print them.

        Args:
            metrics (list): list of string metric names which should be presented in the printed report
            on_each_epoch (bool): present results just at the end of training or at the end of each epoch
            strict_metric_reporting (bool): if False ignore missing metric in the TrainLoop.train_history, if True, in
                case of missing metric throw and exception and thus interrupt the training loop
        """
        AbstractCallback.__init__(self, 'Model performance print reporter')
        self.metrics = metrics
        self.on_each_epoch = on_each_epoch
        self.strict_metric_reporting = strict_metric_reporting

        if len(metrics) == 0:
            raise ValueError('metrics list is empty')

    def on_train_end(self):
        print('End of training performance report:')
        self.print_performance_report()

    def on_epoch_end(self):
        if self.on_each_epoch:
            print('End of epoch performance report:')
            self.print_performance_report()

    def print_performance_report(self):
        for metric_name in self.metrics:
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
                 epoch_end=True, train_end=True, strict_metric_extract=False):
        """

        Args:
            input_metric_getter (lambda): extract full history for the desired metric, not just the last history input.
                Return should be represented as a list.
            output_metric_setter (lambda): take the extracted full history of a metric and convert it as desired.
                Return new / transformed metric name and transformed metric result.
            epoch_end (bool):
            train_end (bool):
            strict_metric_extract (bool):
        """
        AbstractCallback.__init__(self, 'Train history general formatter engine')
        self.input_metric_getter = input_metric_getter
        self.output_metric_setter = output_metric_setter

        self.epoch_end = epoch_end
        self.train_end = train_end
        self.strict_metric_extract = strict_metric_extract

    def on_epoch_end(self):
        if self.epoch_end:
            if self.check_if_history_updated():
                self.format_history()

    def on_train_end(self):
        if self.train_end:
            if self.check_if_history_updated():
                self.format_history()

    def format_history(self):
        input_metric = self.input_metric_getter(self.train_loop_obj.train_history)
        output_metric_name, output_metric = self.output_metric_setter(input_metric)
        self.train_loop_obj.insert_metric_result_into_history(output_metric_name, output_metric)

    def check_if_history_updated(self):
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
    def __init__(self, input_metric_path, new_metric_name, strict_metric_extract=False):
        """

        Args:
            input_metric_path (str or lambda): if using lambda, extract full history for the desired metric,
                not just the last history input. Return should be represented as a list.
            new_metric_name (str):
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
                                       epoch_end=True, train_end=True, strict_metric_extract=strict_metric_extract)
