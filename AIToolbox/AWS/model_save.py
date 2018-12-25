from abc import ABC, abstractmethod
import boto3
import os
import time
import datetime

from AIToolbox.experimet_save.local_model_save import KerasLocalModelSaver, TensorFlowLocalModelSaver, PyTorchLocalModelSaver


class AbstractModelSaver(ABC):
    @abstractmethod
    def save_model(self, model, project_name, experiment_name, experiment_timestamp=None, epoch=None, protect_existing_folder=True):
        pass


class BaseModelSaver:
    def __init__(self, bucket_name='model-result', local_model_result_folder_path='~/project/model_result',
                 checkpoint_model=False):
        """

        Args:
            bucket_name (str):
            local_model_result_folder_path (str):
            checkpoint_model (bool):
        """
        self.bucket_name = bucket_name
        self.s3_client = boto3.client('s3')
        self.s3_resource = boto3.resource('s3')

        self.local_model_result_folder_path = os.path.expanduser(local_model_result_folder_path)
        self.checkpoint_model = checkpoint_model

    def save_file(self, local_file_path, s3_file_path):
        """

        Args:
            local_file_path (str):
            s3_file_path (str):

        Returns:
            None
        """
        self.s3_client.upload_file(os.path.expanduser(local_file_path),
                                   self.bucket_name, s3_file_path)

    def create_experiment_S3_folder_structure(self, project_name, experiment_name, experiment_timestamp):
        """

        Args:
            project_name (str):
            experiment_name (str):
            experiment_timestamp (str):

        Returns:
            str:
        """
        experiment_s3_path = os.path.join(project_name,
                                          experiment_name + '_' + experiment_timestamp,
                                          'model' if not self.checkpoint_model else 'checkpoint_model')
        return experiment_s3_path


class KerasS3ModelSaver(AbstractModelSaver, BaseModelSaver):
    def __init__(self, bucket_name='model-result', local_model_result_folder_path='~/project/model_result',
                 checkpoint_model=False):
        """

        Args:
            bucket_name (str):
            local_model_result_folder_path (str):
            checkpoint_model (bool):
        """
        BaseModelSaver.__init__(self, bucket_name, local_model_result_folder_path, checkpoint_model)
        self.keras_local_saver = KerasLocalModelSaver(local_model_result_folder_path, checkpoint_model)

    def save_model(self, model, project_name, experiment_name, experiment_timestamp=None, epoch=None, protect_existing_folder=True):
        """

        Args:
            model (keras.engine.training.Model):
            project_name (str):
            experiment_name (str):
            experiment_timestamp (str or None):
            epoch (int or None):
            protect_existing_folder (bool):

        Returns:
            (str, str):

        Examples:
            local_model_result_folder_path = '~/project/model_results'
            # local_model_result_folder_path = '~/PycharmProjects/MemoryNet/model_results'
            m_saver = KerasS3ModelSaver(local_model_result_folder_path=local_model_result_folder_path)
            m_saver.save_model(model=model,
                               project_name='QA_QAngaroo', experiment_name='FastQA_RNN_concat_model_GLOVE',
                               protect_existing_folder=False)

        """
        if experiment_timestamp is None:
            experiment_timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')

        saved_local_model_details = self.keras_local_saver.save_model(model, project_name, experiment_name,
                                                                      experiment_timestamp, epoch, protect_existing_folder)

        model_name, model_weights_name, model_local_path, model_weights_local_path = saved_local_model_details

        experiment_s3_path = self.create_experiment_S3_folder_structure(project_name, experiment_name, experiment_timestamp)
        model_s3_path = os.path.join(experiment_s3_path, model_name)
        model_weights_s3_path = os.path.join(experiment_s3_path, model_weights_name)

        self.save_file(local_file_path=model_local_path, s3_file_path=model_s3_path)
        self.save_file(local_file_path=model_weights_local_path, s3_file_path=model_weights_s3_path)

        return model_s3_path, experiment_timestamp


class TensorFlowS3ModelSaver(AbstractModelSaver, BaseModelSaver):
    def __init__(self, bucket_name='model-result', local_model_result_folder_path='~/project/model_result',
                 checkpoint_model=False):
        """

        Args:
            bucket_name (str):
            local_model_result_folder_path (str):
            checkpoint_model (bool):
        """
        BaseModelSaver.__init__(self, bucket_name, local_model_result_folder_path, checkpoint_model)
        self.tf_local_saver = TensorFlowLocalModelSaver(local_model_result_folder_path, checkpoint_model)

    def save_model(self, model, project_name, experiment_name, experiment_timestamp=None, epoch=None, protect_existing_folder=True):
        raise NotImplementedError


class PyTorchS3ModelSaver(AbstractModelSaver, BaseModelSaver):
    def __init__(self, bucket_name='model-result', local_model_result_folder_path='~/project/model_result',
                 checkpoint_model=False):
        """

        Args:
            bucket_name (str):
            local_model_result_folder_path (str):
            checkpoint_model (bool):
        """
        BaseModelSaver.__init__(self, bucket_name, local_model_result_folder_path, checkpoint_model)
        self.pytorch_local_saver = PyTorchLocalModelSaver(local_model_result_folder_path, checkpoint_model)

    def save_model(self, model, project_name, experiment_name, experiment_timestamp=None, epoch=None, protect_existing_folder=True):
        """

        Args:
            model (torch.nn.modules.Module):
            project_name (str):
            experiment_name (str):
            experiment_timestamp (str or None):
            epoch (int or None):
            protect_existing_folder (bool):

        Returns:
            (str, str):

        Examples:
            local_model_result_folder_path = '~/project/model_results'
            # local_model_result_folder_path = '~/PycharmProjects/MemoryNet/model_results'
            m_saver = PyTorchLocalModelSaver(local_model_result_folder_path=local_model_result_folder_path)
            m_saver.save_model(model=model,
                               project_name='QA_QAngaroo', experiment_name='FastQA_RNN_concat_model_GLOVE',
                               protect_existing_folder=False)

        """
        if experiment_timestamp is None:
            experiment_timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')

        saved_local_model_details = self.pytorch_local_saver.save_model(model, project_name, experiment_name,
                                                                        experiment_timestamp, epoch, protect_existing_folder)

        model_name, model_weights_name, model_local_path, model_weights_local_path = saved_local_model_details

        experiment_s3_path = self.create_experiment_S3_folder_structure(project_name, experiment_name, experiment_timestamp)
        model_s3_path = os.path.join(experiment_s3_path, model_name)
        model_weights_s3_path = os.path.join(experiment_s3_path, model_weights_name)

        self.save_file(local_file_path=model_local_path, s3_file_path=model_s3_path)
        self.save_file(local_file_path=model_weights_local_path, s3_file_path=model_weights_s3_path)

        return model_s3_path, experiment_timestamp
