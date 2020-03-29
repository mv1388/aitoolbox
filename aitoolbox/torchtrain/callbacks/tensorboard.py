import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from aitoolbox.torchtrain.callbacks.abstract import AbstractExperimentCallback
from aitoolbox.experiment.local_save.folder_create import ExperimentFolder as FolderCreator


class TensorboardReporterBaseCB(AbstractExperimentCallback):
    def __init__(self, callback_name, comment='', flush_secs=120, filename_suffix='',
                 project_name=None, experiment_name=None, local_model_result_folder_path=None, log_dir=None,
                 is_project=True, **kwargs):
        """

        Args:
            callback_name (str or None):
            comment (str):
            flush_secs (int):
            filename_suffix (str):
            project_name (str or None):
            experiment_name (str or None):
            local_model_result_folder_path (str or None):
            log_dir (str or None):
            is_project (bool): if the results should be saved into the TrainLoop created project folder structure or
                into a specific full path given in the log_dir parameter
            **kwargs: additional parameters for tensorboard SummaryWriter
        """
        AbstractExperimentCallback.__init__(self, callback_name,
                                            project_name, experiment_name, local_model_result_folder_path,
                                            device_idx_execution=0)
        self.log_dir = log_dir
        self.is_project = is_project
        self.fallback_log_dir = 'tensorboard'

        if not self.is_project:
            if log_dir is not None:
                self.full_log_dir = os.path.expanduser(log_dir)
            else:
                raise ValueError(f'is_project is set to {is_project}. As it means callback is to be executed outside'
                                 f'of the project folder structure the log_dir parameter must be specified instead.'
                                 f'Currently the log_dir is set to {log_dir}')

        self.tb_writer = SummaryWriter(log_dir=self.full_log_dir, comment=comment,
                                       flush_secs=flush_secs, filename_suffix=filename_suffix, **kwargs)

    def on_train_loop_registration(self):
        if self.is_project:
            self.try_infer_experiment_details(infer_cloud_details=False)
            self.full_log_dir = self.create_log_dir()

    def on_train_end(self):
        self.tb_writer.close()

    def create_log_dir(self):
        """Crate log folder

        Returns:
            str: log dir path
        """
        if self.log_dir is None:
            self.log_dir = self.fallback_log_dir

        experiment_path = FolderCreator.create_base_folder(self.project_name, self.experiment_name,
                                                           self.train_loop_obj.experiment_timestamp,
                                                           self.local_model_result_folder_path)
        full_log_dir = os.path.join(experiment_path, self.log_dir)
        if not os.path.exists(full_log_dir):
            os.mkdir(full_log_dir)

        return full_log_dir


class TBBatchLossReport(TensorboardReporterBaseCB):
    def __init__(self, log_dir=None, comment='', flush_secs=120, filename_suffix='',
                 project_name=None, experiment_name=None, local_model_result_folder_path=None,
                 is_project=True, **kwargs):
        """

        Args:
            log_dir (str or None):
            comment (str):
            flush_secs (int):
            filename_suffix (str):
            project_name (str or None):
            experiment_name (str or None):
            local_model_result_folder_path (str or None):
            is_project (bool): if the results should be saved into the TrainLoop created project folder structure or
                into a specific full path given in the log_dir parameter
            **kwargs: additional parameters for tensorboard SummaryWriter
        """
        TensorboardReporterBaseCB.__init__(self, 'Tensorboard end of batch report of batch loss',
                                           comment, flush_secs, filename_suffix,
                                           project_name, experiment_name, local_model_result_folder_path, log_dir,
                                           is_project, **kwargs)

    def on_batch_end(self):
        self.tb_writer.add_scalar('train/last_batch_loss', self.train_loop_obj.loss_batch_accum[-1])
        self.tb_writer.add_scalar('train/accumulated_batch_loss', np.mean(self.train_loop_obj.loss_batch_accum).item())


class TBPerformanceMetricReport(TensorboardReporterBaseCB):
    def __init__(self, metric_names=None, log_dir=None, comment='', flush_secs=120, filename_suffix='',
                 project_name=None, experiment_name=None, local_model_result_folder_path=None,
                 is_project=True, **kwargs):
        """

        Args:
            metric_names (list or None):
            log_dir (str or None):
            comment (str):
            flush_secs (int):
            filename_suffix (str):
            project_name (str or None):
            experiment_name (str or None):
            local_model_result_folder_path (str or None):
            is_project (bool):
            **kwargs: additional parameters for tensorboard SummaryWriter
        """
        TensorboardReporterBaseCB.__init__(self, 'Tensorboard end of batch report of batch loss',
                                           comment, flush_secs, filename_suffix,
                                           project_name, experiment_name, local_model_result_folder_path, log_dir,
                                           is_project, **kwargs)
        self.metric_names = metric_names

    def on_epoch_end(self):
        metric_names = self.metric_names if self.metric_names is not None else self.train_loop_obj.train_history.keys()

        for metric_name in metric_names:
            metric_results = self.train_loop_obj.train_history[metric_name]
            self.tb_writer.add_scalar(f'data/{metric_name}', metric_results[-1])


class TBAttentionReport(TensorboardReporterBaseCB):
    def __init__(self, log_dir=None, comment='', flush_secs=120, filename_suffix='',
                 project_name=None, experiment_name=None, local_model_result_folder_path=None,
                 is_project=True, **kwargs):
        TensorboardReporterBaseCB.__init__(self, 'Attention heatmap',
                                           comment, flush_secs, filename_suffix,
                                           project_name, experiment_name, local_model_result_folder_path, log_dir,
                                           is_project, **kwargs)

        raise NotImplementedError

    def on_epoch_end(self):
        raise NotImplementedError


class TBEmbeddingReport(TensorboardReporterBaseCB):
    def __init__(self, log_dir=None, comment='', flush_secs=120, filename_suffix='',
                 project_name=None, experiment_name=None, local_model_result_folder_path=None,
                 is_project=True, **kwargs):
        TensorboardReporterBaseCB.__init__(self, 'Neural network embeddings',
                                           comment, flush_secs, filename_suffix,
                                           project_name, experiment_name, local_model_result_folder_path, log_dir,
                                           is_project, **kwargs)

        raise NotImplementedError

    def on_epoch_end(self):
        raise NotImplementedError


class TBHistogramReport(TensorboardReporterBaseCB):
    def __init__(self, log_dir=None, comment='', flush_secs=120, filename_suffix='',
                 project_name=None, experiment_name=None, local_model_result_folder_path=None,
                 is_project=True, **kwargs):
        TensorboardReporterBaseCB.__init__(self, 'Neural network layers histogram',
                                           comment, flush_secs, filename_suffix,
                                           project_name, experiment_name, local_model_result_folder_path, log_dir,
                                           is_project, **kwargs)

        raise NotImplementedError

    def on_batch_end(self):
        raise NotImplementedError

    def on_epoch_end(self):
        raise NotImplementedError


class TBHImageReport(TensorboardReporterBaseCB):
    def __init__(self, log_dir=None, comment='', flush_secs=120, filename_suffix='',
                 project_name=None, experiment_name=None, local_model_result_folder_path=None,
                 is_project=True, **kwargs):
        """

            TODO: use image and images fns

        """
        TensorboardReporterBaseCB.__init__(self, 'Image result',
                                           comment, flush_secs, filename_suffix,
                                           project_name, experiment_name, local_model_result_folder_path, log_dir,
                                           is_project, **kwargs)

        raise NotImplementedError

    def on_epoch_end(self):
        raise NotImplementedError
