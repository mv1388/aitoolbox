from abc import ABC, abstractmethod
import boto3
import os
import time
import datetime


class AbstractModelSaver(ABC):
    @abstractmethod
    def save_model(self, model, project_name, experiment_name, protect_existing_folder=True):
        pass


class SmartModelSaver:
    def __init__(self, bucket_name='model-result', local_model_result_folder_path='~/project/model_result'):
        """

        Args:
            bucket_name (str):
            local_model_result_folder_path (str):
        """
        self.bucket_name = bucket_name
        self.s3_client = boto3.client('s3')
        self.s3_resource = boto3.resource('s3')

        self.local_model_result_folder_path = os.path.expanduser(local_model_result_folder_path)

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

    def exists_model_experiment_S3_folder(self, s3_project_folder_path, model_experiment_name, protect_local_folder=True):
        """

        Args:
            s3_project_folder_path (str):
            model_experiment_name (str):
            protect_local_folder (bool):

        Returns:
            bool:
        """
        s3_experiment_path = os.path.join(s3_project_folder_path, model_experiment_name)
        s3_experiment_dir_list = list(self.s3_resource.Bucket(self.bucket_name).objects.filter(Prefix=s3_experiment_path))

        if len(s3_experiment_dir_list) > 0 and protect_local_folder:
            print('S3 experiment folder exists and protect_local_folder in True: leaving S3 experiment folder as it is.')
            return True

        if len(s3_experiment_dir_list) > 0 and not protect_local_folder:
            print('S3 experiment folder exists and protect_local_folder in False: deleting S3 experiment folder.')
            for s3_object in s3_experiment_dir_list:
                s3_object.delete()

        return False

    def create_experiment_local_folder_structure(self, project_name, experiment_name, experiment_timestamp):
        """

        Args:
            project_name (str):
            experiment_name (str):
            experiment_timestamp (str):

        Returns:
            str:
        """
        if not os.path.exists(os.path.join(self.local_model_result_folder_path, project_name)):
            os.mkdir(os.path.join(self.local_model_result_folder_path, project_name))

        experiment_path = os.path.join(self.local_model_result_folder_path, project_name,
                                       experiment_name + '_' + experiment_timestamp)
        os.mkdir(experiment_path)
        return experiment_path


class KerasS3ModelSaver(AbstractModelSaver, SmartModelSaver):
    def __init__(self, bucket_name='model-result', local_model_result_folder_path='~/project/model_result'):
        """

        Args:
            bucket_name (str):
            local_model_result_folder_path (str):
        """
        SmartModelSaver.__init__(self, bucket_name, local_model_result_folder_path)

    def save_model(self, model, project_name, experiment_name, protect_existing_folder=True):
        """

        Args:
            model (keras.engine.training.Model):
            project_name (str):
            experiment_name (str):
            protect_existing_folder (bool):

        Returns:
            None

        Examples:
            local_model_result_folder_path = '~/project/model_results'
            # local_model_result_folder_path = '/Users/markovidoni/PycharmProjects/MemoryNet/model_results'
            m_saver = KerasS3ModelSaver(local_model_result_folder_path=local_model_result_folder_path)
            m_saver.save_model(model=model,
                               project_name='QA_QAngaroo', experiment_name='AAAAAFastQA_RNN_concat_model_GLOVE',
                               protect_existing_folder=False)

        """
        experiment_timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')
        experiment_local_path = self.create_experiment_local_folder_structure(project_name, experiment_name, experiment_timestamp)

        model_name = 'model_{}_{}.h5'.format(experiment_name, experiment_timestamp)
        model_local_path = os.path.join(experiment_local_path, model_name)
        model_weights_name = 'modelWeights_{}_{}.h5'.format(experiment_name, experiment_timestamp)
        model_weights_local_path = os.path.join(experiment_local_path, model_weights_name)

        model.save(model_local_path)
        model.save_weights(model_weights_local_path)

        experiment_s3_path = os.path.join(project_name, experiment_name)
        if self.exists_model_experiment_S3_folder(project_name, experiment_name, protect_existing_folder):
            experiment_s3_path += '_' + experiment_timestamp
        model_s3_path = os.path.join(experiment_s3_path, model_name)
        model_weights_s3_path = os.path.join(experiment_s3_path, model_weights_name)

        self.save_file(local_file_path=model_local_path, s3_file_path=model_s3_path)
        self.save_file(local_file_path=model_weights_local_path, s3_file_path=model_weights_s3_path)


class TensorFlowS3ModelSaver(AbstractModelSaver, SmartModelSaver):
    def __init__(self, bucket_name='model-result', local_model_result_folder_path='~/project/model_result'):
        """

        Args:
            bucket_name (str):
            local_model_result_folder_path (str):
        """
        SmartModelSaver.__init__(self, bucket_name, local_model_result_folder_path)

    def save_model(self, model, project_name, experiment_name, protect_existing_folder=True):
        raise NotImplementedError


class PyTorchS3ModelSaver(AbstractModelSaver, SmartModelSaver):
    def __init__(self, bucket_name='model-result', local_model_result_folder_path='~/project/model_result'):
        """

        Args:
            bucket_name (str):
            local_model_result_folder_path (str):
        """
        SmartModelSaver.__init__(self, bucket_name, local_model_result_folder_path)

    def save_model(self, model, project_name, experiment_name, protect_existing_folder=True):
        raise NotImplementedError
