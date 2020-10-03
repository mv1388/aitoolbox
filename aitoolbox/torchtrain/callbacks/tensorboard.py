import os
from torch.utils.tensorboard import SummaryWriter

from aitoolbox.torchtrain.callbacks.abstract import AbstractExperimentCallback
from aitoolbox.experiment.local_save.folder_create import ExperimentFolder as FolderCreator
from aitoolbox.cloud import s3_available_options, gcs_available_options
from aitoolbox.cloud.AWS.results_save import BaseResultsSaver as BaseResultsS3Saver
from aitoolbox.cloud.GoogleCloud.results_save import BaseResultsGoogleStorageSaver


class TensorboardReporterBaseCB(AbstractExperimentCallback):
    def __init__(self, callback_name, log_dir=None, is_project=True,
                 project_name=None, experiment_name=None, local_model_result_folder_path=None,
                 cloud_save_mode=None, bucket_name=None, cloud_dir_prefix=None,
                 **kwargs):
        """Base Tensorboard callback wrapping SummaryWriter

        This base callback is intended to be inherited and extended with the more concrete callback geared towards
        a particular use-case. This callback only setups all the folders needed for local and cloud experiment tracking.

        Args:
            callback_name (str): name of the callback
            log_dir (str or None): save directory location
            is_project (bool): set to ``True`` if the results should be saved into the TrainLoop-created project
                folder structure or to ``False`` if you want to save into a specific full path given in the log_dir
                parameter.
            project_name (str or None): root name of the project
            experiment_name (str or None): name of the particular experiment
            local_model_result_folder_path (str or None): root local path where project folder will be created
            cloud_save_mode (str or None): Storage destination selector.
                For AWS S3: 's3' / 'aws_s3' / 'aws'
                For Google Cloud Storage: 'gcs' / 'google_storage' / 'google storage'
                Everything else results just in local storage to disk
            bucket_name (str): name of the bucket in the cloud storage
            cloud_dir_prefix (str): path to the folder inside the bucket where the experiments are going to be saved
            **kwargs: additional arguments for ``torch.utils.tensorboard.SummaryWriter`` wrapped inside this callback
        """
        AbstractExperimentCallback.__init__(self, callback_name,
                                            project_name, experiment_name, local_model_result_folder_path,
                                            cloud_save_mode, bucket_name, cloud_dir_prefix,
                                            execution_order=97, device_idx_execution=0)
        # Default log_dir
        self.log_dir = 'tensorboard'
        if log_dir is not None:
            self.log_dir = os.path.expanduser(log_dir)

        self.is_project = is_project
        self.tb_writer_kwargs = kwargs
        self.tb_writer = None

        self.global_step = 0

        self.cloud_results_saver = None

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

        for loss_name in last_batch_loss.keys():
            self.tb_writer.add_scalar(
                f'train_loss/last_batch_{loss_name}', last_batch_loss[loss_name],
                self.global_step
            )
            self.tb_writer.add_scalar(
                f'train_loss/accumulated_batch_{loss_name}', accum_mean_batch_loss[loss_name],
                self.global_step
            )

    def log_train_history_metrics(self, metric_names):
        """Log the train history metrics at the end of the epoch

        Args:
            metric_names (list): list of train history tracked metrics to be logged

        Returns:
            None
        """
        for metric_name in metric_names:
            prefix_name = 'metrics'
            if 'loss' in metric_name:
                prefix_name = 'loss'

            metric_results = self.train_loop_obj.train_history[metric_name]
            if len(metric_results) > 0:
                self.tb_writer.add_scalar(f'{prefix_name}/{metric_name}', metric_results[-1],
                                          self.train_loop_obj.epoch)

    def on_train_end(self):
        self.tb_writer.close()
        self.upload_to_cloud()

    def on_train_loop_registration(self):
        if self.is_project:
            self.try_infer_experiment_details(infer_cloud_details=True)
            self.prepare_results_saver()

        self.log_dir = self.create_log_dir()

        if 'filename_suffix' not in self.tb_writer_kwargs and not self.is_project:
            self.tb_writer_kwargs['filename_suffix'] = f'_{self.train_loop_obj.experiment_timestamp}'

        self.tb_writer = SummaryWriter(log_dir=self.log_dir, **self.tb_writer_kwargs)

    def create_log_dir(self):
        full_log_dir_path = self.log_dir

        if self.is_project:
            experiment_path = FolderCreator.create_base_folder(self.project_name, self.experiment_name,
                                                               self.train_loop_obj.experiment_timestamp,
                                                               self.local_model_result_folder_path)
            full_log_dir_path = os.path.join(experiment_path, self.log_dir)
            if not os.path.exists(full_log_dir_path):
                os.mkdir(full_log_dir_path)

        if self.project_name is not None and self.experiment_name is not None:
            full_log_dir_path = os.path.join(full_log_dir_path, f'{self.project_name}_{self.experiment_name}')
            if not os.path.exists(full_log_dir_path):
                os.mkdir(full_log_dir_path)

        return full_log_dir_path

    def prepare_results_saver(self):
        if self.cloud_save_mode in s3_available_options:
            self.cloud_results_saver = BaseResultsS3Saver(bucket_name=self.bucket_name,
                                                          cloud_dir_prefix=self.cloud_dir_prefix)
        elif self.cloud_save_mode in gcs_available_options:
            self.cloud_results_saver = BaseResultsGoogleStorageSaver(bucket_name=self.bucket_name,
                                                                     cloud_dir_prefix=self.cloud_dir_prefix)
        else:
            self.cloud_results_saver = None

    def upload_to_cloud(self):
        """Upload sync the local version of tensorboard file to the cloud storage

        Will only upload to cloud if this callback is used as part of the experiment tracking TrainLoop and
        the results are saved in the cloud experiment's folder.

        Returns:
            None
        """
        if self.cloud_results_saver is not None and self.is_project:
            experiment_results_cloud_path = \
                self.cloud_results_saver.create_experiment_cloud_storage_folder_structure(
                    self.project_name, self.experiment_name, self.train_loop_obj.experiment_timestamp
                )
            if self.project_name is not None and self.experiment_name is not None:
                tb_dir_sub_path = '/'.join(self.log_dir.split('/')[-2:])
            else:
                tb_dir_sub_path = os.path.basename(self.log_dir)

            experiment_cloud_path = os.path.join(
                os.path.dirname(experiment_results_cloud_path),
                tb_dir_sub_path
            )
            for root, _, files in os.walk(self.log_dir):
                for file_name in files:
                    local_file_path = os.path.join(root, file_name)
                    cloud_file_path = os.path.join(experiment_cloud_path, file_name)
                    self.cloud_results_saver.save_file(local_file_path=local_file_path,
                                                       cloud_file_path=cloud_file_path)


class TensorboardTrainBatchLoss(TensorboardReporterBaseCB):
    def __init__(self, batch_log_frequency=1,
                 log_dir=None, is_project=True,
                 project_name=None, experiment_name=None, local_model_result_folder_path=None,
                 cloud_save_mode=None, bucket_name=None, cloud_dir_prefix=None,
                 **kwargs):
        """Tensorboard training loss logger

        Args:
            batch_log_frequency (int): frequency of logging
            log_dir (str or None): save directory location
            is_project (bool): set to ``True`` if the results should be saved into the TrainLoop-created project
                folder structure or to ``False`` if you want to save into a specific full path given in the log_dir
                parameter.
            project_name (str or None): root name of the project
            experiment_name (str or None): name of the particular experiment
            local_model_result_folder_path (str or None): root local path where project folder will be created
            cloud_save_mode (str or None): Storage destination selector.
                For AWS S3: 's3' / 'aws_s3' / 'aws'
                For Google Cloud Storage: 'gcs' / 'google_storage' / 'google storage'
                Everything else results just in local storage to disk
            bucket_name (str): name of the bucket in the cloud storage
            cloud_dir_prefix (str): path to the folder inside the bucket where the experiments are going to be saved
            **kwargs: additional arguments for ``torch.utils.tensorboard.SummaryWriter`` wrapped inside this callback
        """
        TensorboardReporterBaseCB.__init__(self, 'Tensorboard end of batch report of batch loss',
                                           log_dir, is_project,
                                           project_name, experiment_name, local_model_result_folder_path,
                                           cloud_save_mode, bucket_name, cloud_dir_prefix,
                                           **kwargs)
        self.batch_log_frequency = batch_log_frequency

    def on_batch_end(self):
        if self.global_step % self.batch_log_frequency == 0:
            self.log_mid_train_loss()

        self.global_step += 1

    def on_epoch_end(self):
        self.upload_to_cloud()


class TensorboardTrainHistoryMetric(TensorboardReporterBaseCB):
    def __init__(self, metric_names=None,
                 log_dir=None, is_project=True,
                 project_name=None, experiment_name=None, local_model_result_folder_path=None,
                 cloud_save_mode=None, bucket_name=None, cloud_dir_prefix=None,
                 **kwargs):
        """Tensorboard training history values logger

        At each end of epoch logs to tensorboard the last value in the training history stored for some tracked metric.

        Args:
            metric_names (list or None): list of metric names tracked in the training history. If left to ``None``,
                all the metrics in the training history will be logged.
            log_dir (str or None): save directory location
            is_project (bool): set to ``True`` if the results should be saved into the TrainLoop-created project
                folder structure or to ``False`` if you want to save into a specific full path given in the log_dir
                parameter.
            project_name (str or None): root name of the project
            experiment_name (str or None): name of the particular experiment
            local_model_result_folder_path (str or None): root local path where project folder will be created
            cloud_save_mode (str or None): Storage destination selector.
                For AWS S3: 's3' / 'aws_s3' / 'aws'
                For Google Cloud Storage: 'gcs' / 'google_storage' / 'google storage'
                Everything else results just in local storage to disk
            bucket_name (str): name of the bucket in the cloud storage
            cloud_dir_prefix (str): path to the folder inside the bucket where the experiments are going to be saved
            **kwargs: additional arguments for ``torch.utils.tensorboard.SummaryWriter`` wrapped inside this callback
        """
        TensorboardReporterBaseCB.__init__(self, 'Tensorboard end of batch report of batch loss',
                                           log_dir, is_project,
                                           project_name, experiment_name, local_model_result_folder_path,
                                           cloud_save_mode, bucket_name, cloud_dir_prefix,
                                           **kwargs)
        self.metric_names = metric_names

    def on_epoch_end(self):
        metric_names = self.metric_names if self.metric_names is not None else self.train_loop_obj.train_history.keys()
        self.log_train_history_metrics(metric_names)

        self.tb_writer.flush()
        self.upload_to_cloud()


class TensorboardFullTracking(TensorboardReporterBaseCB):
    def __init__(self, metric_names=None, batch_log_frequency=1,
                 log_dir=None, is_project=True,
                 project_name=None, experiment_name=None, local_model_result_folder_path=None,
                 cloud_save_mode=None, bucket_name=None, cloud_dir_prefix=None,
                 **kwargs):
        """Full Tensorboard logger

        At each end of epoch logs to tensorboard the last value in the training history stored for some tracked metric.
        Also logs the training loss at the batch iteration level.

        Args:
            metric_names (list or None): list of metric names tracked in the training history. If left to ``None``,
                all the metrics in the training history will be logged.
            batch_log_frequency (int): frequency of logging
            log_dir (str or None): save directory location
            is_project (bool): set to ``True`` if the results should be saved into the TrainLoop-created project
                folder structure or to ``False`` if you want to save into a specific full path given in the log_dir
                parameter.
            project_name (str or None): root name of the project
            experiment_name (str or None): name of the particular experiment
            local_model_result_folder_path (str or None): root local path where project folder will be created
            cloud_save_mode (str or None): Storage destination selector.
                For AWS S3: 's3' / 'aws_s3' / 'aws'
                For Google Cloud Storage: 'gcs' / 'google_storage' / 'google storage'
                Everything else results just in local storage to disk
            bucket_name (str): name of the bucket in the cloud storage
            cloud_dir_prefix (str): path to the folder inside the bucket where the experiments are going to be saved
            **kwargs: additional arguments for ``torch.utils.tensorboard.SummaryWriter`` wrapped inside this callback
        """
        TensorboardReporterBaseCB.__init__(self, 'Tensorboard full tracking',
                                           log_dir, is_project,
                                           project_name, experiment_name, local_model_result_folder_path,
                                           cloud_save_mode, bucket_name, cloud_dir_prefix,
                                           **kwargs)
        self.metric_names = metric_names
        self.batch_log_frequency = batch_log_frequency

    def on_batch_end(self):
        if self.global_step % self.batch_log_frequency == 0:
            self.log_mid_train_loss()

        self.global_step += 1

    def on_epoch_end(self):
        metric_names = self.metric_names if self.metric_names is not None else self.train_loop_obj.train_history.keys()
        self.log_train_history_metrics(metric_names)

        self.tb_writer.flush()
        self.upload_to_cloud()
