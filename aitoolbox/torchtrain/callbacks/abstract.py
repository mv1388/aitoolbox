import os
from typing import Optional


class AbstractCallback:
    def __init__(self, callback_name, execution_order=0, device_idx_execution=None):
        """Abstract callback class that all actual callback classes have to inherit from

        In the inherited callback classes the callback methods should be overwritten and used to implement desired
        callback functionality at specific points of the train loop.

        Args:
            callback_name (str): name of the callback
            execution_order (int): order of the callback execution. If all the used callbacks have the orders set to 0,
                then the callbacks are executed in the order they were registered.
            device_idx_execution (int or None): index of the (CUDA GPU) device DDP process inside which the callback
                should be executed
        """
        from aitoolbox.torchtrain.train_loop import TrainLoop
        from aitoolbox.torchtrain.train_loop.components.message_passing import MessageService

        self.callback_name = callback_name
        self.execution_order = execution_order
        self.train_loop_obj: Optional[TrainLoop] = None
        self.message_service: Optional[MessageService] = None
        self.device_idx_execution = device_idx_execution

    def register_train_loop_object(self, train_loop_obj):
        """Introduce the reference to the encapsulating trainloop so that the callback has access to the
            low level functionality of the trainloop

        The registration is normally handled by the callback handler found inside the train loops. The handler is
        responsible for all the callback orchestration of the callbacks inside the trainloops.

        Args:
            train_loop_obj (aitoolbox.torchtrain.train_loop.TrainLoop): reference to the encapsulating trainloop

        Returns:
            AbstractCallback: return the reference to the callback after it is registered
        """
        self.train_loop_obj = train_loop_obj
        self.message_service = train_loop_obj.message_service
        self.on_train_loop_registration()
        return self

    def on_train_loop_registration(self):
        """Execute callback initialization / preparation after the train_loop_object becomes available

        Returns:
            None
        """
        pass

    def on_epoch_begin(self):
        """Logic executed at the beginning of the epoch

        Returns:
            None
        """
        pass

    def on_epoch_end(self):
        """Logic executed at the end of the epoch

        Returns:
            None
        """
        pass

    def on_train_begin(self):
        """Logic executed at the beginning of the overall training

        Returns:
            None
        """
        pass

    def on_train_end(self):
        """Logic executed at the end of the overall training

        Returns:
            None
        """
        pass

    def on_batch_begin(self):
        """Logic executed before the batch is inserted into the model

        Returns:
            None
        """
        pass

    def on_batch_end(self):
        """Logic executed after the batch is inserted into the model

        Returns:
            None
        """
        pass

    def on_after_gradient_update(self, optimizer_idx):
        """Logic executed after the model gradients are updated

        To ensure the execution of this callback enable the `self.train_loop_obj.grad_cb_used = True` option in the
        on_train_loop_registration(). Otherwise, logic implemented here will not be executed by the TrainLoop.

        Args:
            optimizer_idx (int): index of the current optimizer. Mostly useful when using multiple optimizers.
                When only a single optimizer is used this parameter can be ignored.

        Returns:
            None
        """
        pass

    def on_after_optimizer_step(self):
        """Logic executed after the optimizer does a new step and updates the model weights

        To ensure the execution of this callback enable the `self.train_loop_obj.grad_cb_used = True` option in the
        on_train_loop_registration(). Otherwise, logic implemented here will not be executed by the TrainLoop.

        Returns:
            None
        """
        pass

    def on_multiprocess_start(self):
        """Logic executed after a new multiprocessing process is spawned at the beginning of every child process

        Returns:
            None
        """
        pass

    def on_after_batch_prediction(self, y_pred_batch, y_test_batch, metadata_batch, dataset_info):
        """Logic executed in the prediction loop after the predictions for the single batch are made

        IMPORTANT: Take care to not unintentionally modify the (predicted) input data when it's passed inside
                   this function of a callback (you have a reference to the original).
                   If the data is modified the subsequent steps or evaluations that are executed by the TrainLoop
                   might get broken or corrupted. With more access/power there needs to be more care!

        All the inputs into this function are the outputs from the model's ``get_predictions()`` method.

        Args:
            y_pred_batch: model's predictions for the current batch
            y_test_batch: reference ground truth targets for the current batch
            metadata_batch: additional results/metadata returned by the model for the current batch
            dataset_info (dict or None): additional information describing the dataset inside the provided dataloader.
                One such dataset info is the dataset ``type`` (train, validation, or test) set by TrainLoop's
                predict_on_train_set(), predict_on_validation_set() and predict_on_test_set() methods.

        Returns:
            None
        """
        pass


class AbstractExperimentCallback(AbstractCallback):
    def __init__(self, callback_name,
                 project_name=None, experiment_name=None, local_model_result_folder_path=None,
                 cloud_save_mode=None, bucket_name=None, cloud_dir_prefix=None,
                 execution_order=0, device_idx_execution=None):
        """Extension of the AbstractCallback implementing the automatic experiment details inference from TrainLoop

        This abstract callback is inherited from when the implemented callbacks intend to save results files into the
        experiment folder and also potentially upload them to AWS S3.

        Args:
            callback_name (str): name of the callback
            project_name (str or None): root name of the project
            experiment_name (str or None): name of the particular experiment
            local_model_result_folder_path (str or None): root local path where project folder will be created
            cloud_save_mode (str or None): Storage destination selector.
                For AWS S3: 's3' / 'aws_s3' / 'aws'
                For Google Cloud Storage: 'gcs' / 'google_storage' / 'google storage'
                Everything else results just in local storage to disk
            bucket_name (str): name of the bucket in the cloud storage
            cloud_dir_prefix (str): path to the folder inside the bucket where the experiments are going to be saved
            execution_order (int): order of the callback execution. If all the used callbacks have the orders set to 0,
                then the callbacks are executed in the order they were registered.
            device_idx_execution (int or None): index of the (CUDA GPU) device DDP process inside which the callback
                should be executed
        """
        AbstractCallback.__init__(self, callback_name, execution_order=execution_order,
                                  device_idx_execution=device_idx_execution)
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.local_model_result_folder_path = os.path.expanduser(local_model_result_folder_path) \
            if local_model_result_folder_path is not None \
            else None

        self.cloud_save_mode = cloud_save_mode
        self.bucket_name = bucket_name
        self.cloud_dir_prefix = cloud_dir_prefix

    def try_infer_experiment_details(self, infer_cloud_details):
        """Infer paths where to save experiment related files from the running TrainLoop.

        This details inference function should only be called after the callback has already been registered in the
        TrainLoop, e.g. in the on_train_loop_registration().

        General rule:
            take details from the TrainLoop -> for this option where experiment details are inferred from TrainLoop
                            all of the cloud_save_mode, bucket_name and cloud_dir_prefix should be set to None

            Based on `self.cloud_save_mode` the inference decision is made as follows:
                - ['s3', 'aws_s3', 'aws'] --> AWS S3
                - ['gcs', 'google_storage', 'google storage'] --> Google Cloud Storage
                - 'local' or whatever value -> local only

        Args:
            infer_cloud_details (bool): should infer only local project folder details or also cloud project destination

        Raises:
            AttributeError

        Returns:
            None
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
                                 'creation. Project name, etc. thus can not be automatically deduced. Please provide '
                                 'them in the callback parameters instead of currently used None values.')

        try:
            if infer_cloud_details and \
                    self.cloud_save_mode is None and self.bucket_name is None and self.cloud_dir_prefix is None:
                # infer from train loop
                self.cloud_save_mode = self.train_loop_obj.cloud_save_mode
                self.bucket_name = self.train_loop_obj.bucket_name
                self.cloud_dir_prefix = self.train_loop_obj.cloud_dir_prefix
        except AttributeError:
            raise AttributeError('Currently used TrainLoop does not support automatic project cloud storage details '
                                 'inference. Please provide them in the callback parameters instead of '
                                 'currently used None values.')
