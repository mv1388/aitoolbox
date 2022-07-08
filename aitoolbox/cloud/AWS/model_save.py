from abc import ABC, abstractmethod
import os
import time
import datetime

from aitoolbox.cloud.AWS.data_access import BaseDataSaver
from aitoolbox.experiment.local_save.local_model_save import PyTorchLocalModelSaver, KerasLocalModelSaver


class AbstractModelSaver(ABC):
    @abstractmethod
    def save_model(self, model, project_name, experiment_name, experiment_timestamp=None,
                   epoch=None, iteration_idx=None,
                   protect_existing_folder=True):
        """

        Args:
            model:
            project_name (str):
            experiment_name (str):
            experiment_timestamp (str or None):
            epoch (int or None):
            iteration_idx (int or None):
            protect_existing_folder (bool):

        Returns:
            (str, str, str): model_s3_path, experiment_timestamp, model_local_path
        """
        pass


class BaseModelSaver(BaseDataSaver):
    def __init__(self, bucket_name='model-result', cloud_dir_prefix='', checkpoint_model=False):
        """Base model saving to AWS S3 functionality

        Args:
            bucket_name (str): S3 bucket into which the files will be saved
            cloud_dir_prefix (str): destination folder path inside selected bucket
            checkpoint_model (bool): if the model that is going to be saved is final model or mid-training checkpoint
        """
        BaseDataSaver.__init__(self, bucket_name)
        self.cloud_dir_prefix = cloud_dir_prefix
        self.checkpoint_model = checkpoint_model

    def create_experiment_cloud_storage_folder_structure(self, project_name, experiment_name, experiment_timestamp):
        """

        Args:
            project_name (str): root name of the project
            experiment_name (str): name of the particular experiment
            experiment_timestamp (str): time stamp at the start of training

        Returns:
            str: experiment cloud path
        """
        experiment_cloud_path = os.path.join(self.cloud_dir_prefix,
                                             project_name,
                                             experiment_name + '_' + experiment_timestamp,
                                             'model' if not self.checkpoint_model else 'checkpoint_model')
        return experiment_cloud_path


class PyTorchS3ModelSaver(AbstractModelSaver, BaseModelSaver):
    def __init__(self, bucket_name='model-result', cloud_dir_prefix='',
                 local_model_result_folder_path='~/project/model_result', checkpoint_model=False):
        """PyTorch AWS S3 model saving

        Args:
            bucket_name (str): name of the bucket in the S3 to which the models will be saved
            cloud_dir_prefix (str): destination folder path inside selected bucket
            local_model_result_folder_path (str): root local path where project folder will be created
            checkpoint_model (bool): if the model being saved is checkpoint model or final end of training model
        """
        BaseModelSaver.__init__(self, bucket_name, cloud_dir_prefix, checkpoint_model)
        self.pytorch_local_saver = PyTorchLocalModelSaver(local_model_result_folder_path, checkpoint_model)

    def save_model(self, model, project_name, experiment_name, experiment_timestamp=None,
                   epoch=None, iteration_idx=None,
                   protect_existing_folder=True):
        """Save PyTorch model representation to AWS S3

        Args:
            model (dict): PyTorch model representation dict
            project_name (str): root name of the project
            experiment_name (str): name of the particular experiment
            experiment_timestamp (str or None): time stamp at the start of training
            epoch (int or None): epoch number
            iteration_idx (int or None): at which training iteration the model is being saved
            protect_existing_folder (bool): can override potentially already existing folder or not

        Returns:
            (str, str, str): model_s3_path, experiment_timestamp, model_local_path

        Examples:
            .. code-block:: python

                local_model_result_folder_path = '~/project/model_results'
                m_saver = PyTorchLocalModelSaver(local_model_result_folder_path=local_model_result_folder_path)
                m_saver.save_model(model=model,
                                   project_name='QA_QAngaroo',
                                   experiment_name='FastQA_RNN_concat_model_GLOVE',
                                   protect_existing_folder=False)
        """
        PyTorchLocalModelSaver.check_model_dict_contents(model)

        if experiment_timestamp is None:
            experiment_timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S')

        model_name, model_local_path = self.pytorch_local_saver.save_model(model, project_name, experiment_name,
                                                                           experiment_timestamp, epoch, iteration_idx,
                                                                           protect_existing_folder)

        experiment_s3_path = self.create_experiment_cloud_storage_folder_structure(project_name, experiment_name,
                                                                                   experiment_timestamp)
        model_s3_path = os.path.join(experiment_s3_path, model_name)

        self.save_file(local_file_path=model_local_path, cloud_file_path=model_s3_path)

        full_model_s3_path = os.path.join(self.bucket_name, model_s3_path)

        return full_model_s3_path, experiment_timestamp, model_local_path


class KerasS3ModelSaver(AbstractModelSaver, BaseModelSaver):
    def __init__(self, bucket_name='model-result', cloud_dir_prefix='',
                 local_model_result_folder_path='~/project/model_result', checkpoint_model=False):
        """Keras AWS S3 model saving

        Args:
            bucket_name (str): name of the bucket in the S3 to which the models will be saved
            cloud_dir_prefix (str): destination folder path inside selected bucket
            local_model_result_folder_path (str): root local path where project folder will be created
            checkpoint_model (bool): if the model being saved is checkpoint model or final end of training model
        """
        BaseModelSaver.__init__(self, bucket_name, cloud_dir_prefix, checkpoint_model)
        self.keras_local_saver = KerasLocalModelSaver(local_model_result_folder_path, checkpoint_model)

    def save_model(self, model, project_name, experiment_name, experiment_timestamp=None,
                   epoch=None, iteration_idx=None,
                   protect_existing_folder=True):
        """Save Keras model to AWS S3

        Args:
            model (keras.Model):
            project_name (str): root name of the project
            experiment_name (str): name of the particular experiment
            experiment_timestamp (str or None): time stamp at the start of training
            epoch (int or None): epoch number
            iteration_idx (int or None): at which training iteration the model is being saved
            protect_existing_folder (bool): can override potentially already existing folder or not

        Returns:
            (str, str, str): model_s3_path, experiment_timestamp, model_local_path

        Examples:
            .. code-block:: python

                local_model_result_folder_path = '~/project/model_results'
                m_saver = KerasS3ModelSaver(local_model_result_folder_path=local_model_result_folder_path)
                m_saver.save_model(model=model,
                                   project_name='QA_QAngaroo',
                                   experiment_name='FastQA_RNN_concat_model_GLOVE',
                                   protect_existing_folder=False)
        """
        if experiment_timestamp is None:
            experiment_timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S')

        model_name, model_local_path = self.keras_local_saver.save_model(model, project_name, experiment_name,
                                                                         experiment_timestamp, epoch, iteration_idx,
                                                                         protect_existing_folder)

        experiment_s3_path = self.create_experiment_cloud_storage_folder_structure(project_name, experiment_name,
                                                                                   experiment_timestamp)
        model_s3_path = os.path.join(experiment_s3_path, model_name)

        self.save_file(local_file_path=model_local_path, cloud_file_path=model_s3_path)

        full_model_s3_path = os.path.join(self.bucket_name, model_s3_path)

        return full_model_s3_path, experiment_timestamp, model_local_path
