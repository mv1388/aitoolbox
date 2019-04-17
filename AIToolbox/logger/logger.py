import logging
import os

from AIToolbox.experiment_save.local_save.local_results_save import BaseLocalResultsSaver


class Logger:
    def __init__(self, summary_logger='aitoolbox_summary', full_logger='aitoolbox_full', name=None):
        """

        Args:
            summary_logger:
            full_logger:
            name:
        """
        prefix = f'{name}_' if name else ''
        
        self.summary_logger_name = f'{prefix}{summary_logger}'
        self.full_logger_name = f'{prefix}{full_logger}'

        self.summary_logger = logging.getLogger(self.summary_logger_name)
        self.full_logger = logging.getLogger(self.full_logger_name)

        self.summary_logger.propagate = False
        self.full_logger.propagate = False
        
    def get_summary_log_name(self):
        return self.summary_logger_name
    
    def get_full_log_name(self):
        return self.full_logger_name

    def info(self, msg, for_summary=True):
        """

        Args:
            msg:
            for_summary:

        Returns:

        """
        if for_summary:
            self.summary_logger.info(msg)
        self.full_logger.info(msg)

    def debug(self, msg, for_summary=True):
        """

        Args:
            msg:
            for_summary:

        Returns:

        """
        if for_summary:
            self.summary_logger.debug(msg)
        self.full_logger.debug(msg)

    def warning(self, msg, for_summary=True):
        """

        Args:
            msg:
            for_summary:

        Returns:

        """
        if for_summary:
            self.summary_logger.warning(msg)
        self.full_logger.warning(msg)

    def error(self, msg, for_summary=True):
        """

        Args:
            msg:
            for_summary:

        Returns:

        """
        if for_summary:
            self.summary_logger.error(msg)
        self.full_logger.error(msg)

    def setup_logger(self, logs_folder_path, summary_logger_name=None, full_logger_name=None):
        """

        Args:
            logs_folder_path:
            summary_logger_name:
            full_logger_name:

        Returns:

        """
        summary_log_path = os.path.join(logs_folder_path, 'summary.log')
        full_log_path = os.path.join(logs_folder_path, 'full.log')

        summary_logger_name = summary_logger_name if summary_logger_name is not None else self.get_summary_log_name()
        full_logger_name = full_logger_name if full_logger_name is not None else self.get_full_log_name()

        summary_logger = self._setup_logger(summary_logger_name, summary_log_path, logging.INFO, console_format=True)
        full_logger = self._setup_logger(full_logger_name, full_log_path, logging.DEBUG)

        return summary_logger, full_logger

    @staticmethod
    def _setup_logger(logger_name, logger_output_file_path, level, console_format=False):
        """

        Args:
            logger_name (str):
            logger_output_file_path (str):
            level:
            console_format (bool):

        Returns:

        """
        logger = logging.getLogger(logger_name)

        handler = logging.FileHandler(logger_output_file_path, "w", encoding=None, delay=True)
        handler.setLevel(level)

        # formatter = logging.Formatter("%(asctime)s — %(name)s — %(levelname)s — %(message)s")
        formatter = logging.Formatter("%(asctime)s — %(levelname)s:  %(message)s")

        handler.setFormatter(formatter)
        # remove all old handlers
        for hdlr in logger.handlers[:]:
            logger.removeHandler(hdlr)
        logger.addHandler(handler)

        if console_format:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(level)
            console_formatter = logging.Formatter("%(asctime)s — %(levelname)s:  %(message)s")
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)

        return logger

    @staticmethod
    def create_experiment_logs_local_folder_structure(local_model_result_folder_path,
                                                      project_name, experiment_name, experiment_timestamp):
        """

        Args:
            local_model_result_folder_path (str):
            project_name (str):
            experiment_name (str):
            experiment_timestamp (str):

        Returns:
            str:

        """
        paths = BaseLocalResultsSaver.form_experiment_local_folders_paths(project_name, experiment_name,
                                                                          experiment_timestamp,
                                                                          local_model_result_folder_path)
        project_path, experiment_path, _ = paths
        logs_folder_path = os.path.join(experiment_path, 'logs')

        if not os.path.exists(project_path):
            os.mkdir(project_path)
        if not os.path.exists(experiment_path):
            os.mkdir(experiment_path)
        if not os.path.exists(logs_folder_path):
            os.mkdir(logs_folder_path)

        return logs_folder_path
