import os
from dataclasses import dataclass

import wandb

from aitoolbox.torchtrain.callbacks.abstract import AbstractExperimentCallback
from aitoolbox.experiment.local_save.folder_create import ExperimentFolder as FolderCreator


@dataclass
class AlertConfig:
    metric_name: str
    threshold_value: float
    objective: str = "maximize"
    wandb_alert_level: wandb.AlertLevel = None


class WandBTracking(AbstractExperimentCallback):
    def __init__(self, metric_names=None, batch_log_frequency=None, hyperparams=None, tags=None, alerts=None,
                 wandb_pre_initialized=False, source_dirs=(), log_dir=None, is_project=True,
                 project_name=None, experiment_name=None, local_model_result_folder_path=None,
                 **kwargs):
        """Weights And Biases Logger

        Find more on: https://wandb.ai

        Before this callback can be used you need to have wandb account and be credentialed on the machine.
        Instructions for this process can be found on wandb GitHub: https://github.com/wandb/client

        Args:
            metric_names (list or None): list of metric names tracked in the training history. If left to ``None``,
                all the metrics in the training history will be logged.
            batch_log_frequency (int or None): frequency of logging. If set to None batch level logging is skipped.
                Instead of also mid-epoch logging only end-of-epoch logging is executed.
            hyperparams (dict or None): dictionary of used hyperparameters. If set to None the callback tries to find
                the hyperparameter dict in the encapsulating TrainLoop running the callback.
            tags (list or None): used for wandb init. From wandb documentation: A list of strings, which will populate
                the list of tags on this run in the UI. Tags are useful for organizing runs together, or applying
                temporary labels like "baseline" or "production". It's easy to add and remove tags in the UI, or filter
                down to just runs with a specific tag.
            alerts (list[AlertConfig] or None): list of alerts where each alert configuration is specified as
                an AlertConfig dataclass. User should provide the ``metric_name`` based on which the alert should be
                triggered. The last calculated value of the metric is then compared with the provided
                ``threshold_value``. The ``objective`` can be either "maximize" or "minimize".
            wandb_pre_initialized (bool): if wandb has been initialized already outside the callback
                (e.g. at the start of the experiment script). If not, the callback initializes the wandb process.
            source_dirs (tuple or list): list of source code directories which will be stored by wandb. If empty list
                is given the callback will try to get this information from the running TrainLoop. If this is also not
                available the callback leaves wandb default code saving operation which saves the execution
                python script.
            log_dir (str or None): save directory location
            is_project (bool): set to ``True`` if the wandb project folder should be placed into the TrainLoop-created
                project folder structure or to ``False`` if you want to save into a specific full path given in
                the log_dir parameter.
            project_name (str or None): root name of the project
            experiment_name (str or None): name of the particular experiment
            local_model_result_folder_path (str or None): root local path where project folder will be created
            **kwargs: additional arguments for ``wandb.init()`` wrapped inside this callback
        """
        AbstractExperimentCallback.__init__(self, 'WeightsAndBiases Experiment Tracking',
                                            project_name, experiment_name, local_model_result_folder_path,
                                            execution_order=97, device_idx_execution=0)
        self.metric_names = metric_names
        self.batch_log_frequency = batch_log_frequency
        self.hyperparams = hyperparams
        self.tags = tags

        self.alerts = alerts
        self.check_alerts()

        self.wandb_pre_initialized = wandb_pre_initialized
        self.source_dirs = source_dirs
        self.wandb_params_kwargs = kwargs

        self.log_dir = log_dir if log_dir is None else os.path.expanduser(log_dir)
        self.is_project = is_project

    def on_epoch_end(self):
        metric_names = self.metric_names if self.metric_names is not None else self.train_loop_obj.train_history.keys()
        metrics_log = self.log_train_history_metrics(metric_names)

        if self.alerts is not None:
            self.send_configured_alerts(self.alerts, metrics_log)

    def on_batch_end(self):
        if self.batch_log_frequency is not None and \
                self.train_loop_obj.total_iteration_idx % self.batch_log_frequency == 0:
            self.log_mid_train_loss()

    def log_mid_train_loss(self):
        """Log the training loss at the batch iteration level

        Logs current batch loss and the accumulated average loss.

        Returns:
            None
        """
        last_batch_loss = self.train_loop_obj.parse_loss(self.train_loop_obj.loss_batch_accum[-1:])
        accum_mean_batch_loss = self.train_loop_obj.parse_loss(self.train_loop_obj.loss_batch_accum)

        if not isinstance(last_batch_loss, dict) and not isinstance(accum_mean_batch_loss, dict):
            last_batch_loss = {'loss': last_batch_loss}
            accum_mean_batch_loss = {'loss': accum_mean_batch_loss}

        assert sorted(last_batch_loss.keys()) == sorted(accum_mean_batch_loss.keys())

        loss_logging = {}

        for loss_name in last_batch_loss.keys():
            loss_logging[f'train_loss/last_batch_{loss_name}'] = last_batch_loss[loss_name]
            loss_logging[f'train_loss/accumulated_batch_{loss_name}'] = accum_mean_batch_loss[loss_name]

        wandb.run.log(loss_logging, step=self.train_loop_obj.total_iteration_idx)

    def log_train_history_metrics(self, metric_names):
        """Log the train history metrics at the end of the epoch

        Args:
            metric_names (list): list of train history tracked metrics to be logged

        Returns:
            None
        """
        metrics_log = {'epoch': self.train_loop_obj.epoch}

        for metric_name in metric_names:
            prefix_name = 'metrics'
            if 'loss' in metric_name:
                prefix_name = 'loss_at_epoch_end'

            metric_results = self.train_loop_obj.train_history[metric_name]
            if len(metric_results) > 0:
                metrics_log[f'{prefix_name}/{metric_name}'] = metric_results[-1]

        wandb.run.log(metrics_log, step=self.train_loop_obj.total_iteration_idx, commit=True)

        return metrics_log

    @staticmethod
    def send_configured_alerts(alerts, metrics_log):
        """Send wandb alerts

        Sending of alerts depends on current metric values in the ``metrics_log``
        satisfying the conditions specified in the alert configuration.

        Args:
            alerts (list[AlertConfig]): list of alerts where each alert configuration is specified as
                an AlertConfig dataclass. User should provide the ``metric_name`` based on which the alert
                should be triggered. The last calculated value of the metric is then compared with
                the provided ``threshold_value``. The ``objective`` can be either "maximize" or "minimize".
            metrics_log (dict): dict of metrics names and their corresponding current values.

        Returns:
            None
        """
        for alert_config in alerts:
            metric_result = metrics_log[alert_config.metric_name]

            if alert_config.objective == 'maximize' and metric_result < alert_config.threshold_value:
                wandb.alert(
                    title=f"{alert_config.metric_name} is too low",
                    text=f"Metric {alert_config.metric_name} is currently at {metric_result} "
                         f"which is below the specified threshold of {alert_config.threshold_value}.",
                    level=alert_config.wandb_alert_level
                )

            elif alert_config.objective == 'minimize' and metric_result > alert_config.threshold_value:
                wandb.alert(
                    title=f"{alert_config.metric_name} is too high",
                    text=f"Metric {alert_config.metric_name} is currently at {metric_result} "
                         f"which is above the specified threshold of {alert_config.threshold_value}.",
                    level=alert_config.wandb_alert_level
                )

    def on_train_loop_registration(self):
        self.try_infer_experiment_details(infer_cloud_details=False)
        self.try_infer_additional_logging_details()

        if not self.wandb_pre_initialized:
            if self.is_project:
                self.log_dir = FolderCreator.create_base_folder(
                    self.project_name, self.experiment_name,
                    self.train_loop_obj.experiment_timestamp,
                    self.local_model_result_folder_path
                )

            wandb.init(
                project=self.project_name, name=self.experiment_name,
                config=self.hyperparams, tags=self.tags,
                save_code=True,
                dir=self.log_dir,
                **self.wandb_params_kwargs
            )

        for source_code_path in self.source_dirs:
            wandb.run.log_code(
                source_code_path, name=os.path.basename(source_code_path), include_fn=lambda _: True
            )

    def try_infer_additional_logging_details(self):
        try:
            if self.hyperparams is None:
                self.hyperparams = self.train_loop_obj.hyperparams
            if len(self.source_dirs) == 0 and self.is_project:
                self.source_dirs = \
                    [self.train_loop_obj.hyperparams['experiment_file_path']] + list(self.train_loop_obj.source_dirs)
        except AttributeError:
            raise AttributeError('Hyperparameters dict and/or source_dirs list not provided to the WandBTracking '
                                 'callback. It also was not possible to retrieve it from the experiment tracking '
                                 'TrainLoop. Possible reason is the use of the basic TrainLoop.')
        except KeyError:
            raise KeyError("'experiment_file_path' not in the TrainingLoop hyperparams dict. If you want to log only "
                           "the single execution python file and don't want to specify it manually consider switching "
                           "'is_project' parameter to False.")

    def check_alerts(self):
        if self.alerts is not None:
            for alert in self.alerts:
                if not isinstance(alert, AlertConfig):
                    raise TypeError("Alerts should be instances of AlertConfig dataclass.")

                if alert.objective not in ['maximize', 'minimize']:
                    raise ValueError("Alert objective can only be 'maximize' or 'minimize'. "
                                     f"Alert {alert} has objective set to: {alert.objective}.")
