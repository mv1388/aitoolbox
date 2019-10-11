import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from aitoolbox.torchtrain.callbacks.callbacks import AbstractCallback
from aitoolbox.experiment.local_save.folder_create import ExperimentFolderCreator as FolderCreator


class TensorboardReporterBaseCB(AbstractCallback):
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
        AbstractCallback.__init__(self, callback_name)
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.local_model_result_folder_path = os.path.expanduser(local_model_result_folder_path) \
            if local_model_result_folder_path is not None \
            else None
        self.log_dir = log_dir
        self.fallback_log_dir = 'tensorboard'

        if is_project:
            self.full_log_dir = self.try_infer_experiment_details()
        else:
            if log_dir is not None:
                self.full_log_dir = os.path.expanduser(log_dir)
            else:
                raise ValueError(f'is_project is set to {is_project}. As it means callback is to be executed outside'
                                 f'of the project folder structure the log_dir parameter must be specified instead.'
                                 f'Currently the log_dir is set to {log_dir}')

        self.tb_writer = SummaryWriter(log_dir=self.full_log_dir, comment=comment,
                                       flush_secs=flush_secs, filename_suffix=filename_suffix, **kwargs)

    def on_train_end(self):
        self.tb_writer.close()

    def try_infer_experiment_details(self):
        """

        Returns:
            str: tensorboard folder path

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
            if self.log_dir is None:
                self.log_dir = self.fallback_log_dir

            experiment_path = FolderCreator.create_experiment_base_folder(self.project_name, self.experiment_name,
                                                                          self.train_loop_obj.experiment_timestamp,
                                                                          self.local_model_result_folder_path)
            full_log_dir = os.path.join(experiment_path, self.log_dir)
            if not os.path.exists(full_log_dir):
                os.mkdir(full_log_dir)

            return full_log_dir

        except AttributeError:
            raise AttributeError('Currently used TrainLoop does not support automatic project folder structure '
                                 'creation. Project log_dir thus can not be automatically deduced. Please provide'
                                 'it in the callback parameter instead of currently used None value.')


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
