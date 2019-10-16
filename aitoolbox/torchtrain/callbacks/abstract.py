from typing import Optional


class AbstractCallback:
    def __init__(self, callback_name, execution_order=0):
        """Abstract callback class that all actual callback classes have to inherit from

        In the inherited callback classes the callback methods should be overwritten and used to implement desired
        callback functionality at specific points of the train loop.

        Args:
            callback_name (str): name of the callback
            execution_order (int): order of the callback execution. If all the used callbacks have the orders set to 0,
                than the callbacks are executed in the order they were registered.
        """
        from aitoolbox.torchtrain.train_loop import TrainLoop
        from aitoolbox.torchtrain.tl_components.message_passing import MessageService

        self.callback_name = callback_name
        self.execution_order = execution_order
        self.train_loop_obj: Optional[TrainLoop] = None
        self.message_service: Optional[MessageService] = None

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

        """
        pass

    def on_epoch_begin(self):
        """Logic executed at the beginning of the epoch

        Returns:

        """
        pass

    def on_epoch_end(self):
        """Logic executed at the end of the epoch

        Returns:

        """
        pass

    def on_train_begin(self):
        """Logic executed at the beginning of the overall training

        Returns:

        """
        pass

    def on_train_end(self):
        """Logic executed at the end of the overall training

        Returns:

        """
        pass

    def on_batch_begin(self):
        """Logic executed before the batch is inserted into the model

        Returns:

        """
        pass

    def on_batch_end(self):
        """Logic executed after the batch is inserted into the model

        Returns:

        """
        pass

    def on_after_gradient_update(self):
        """Logic executed after the model gradients are updated

        To ensure the execution of this callback enable the `self.train_loop_obj.grad_cb_used = True` option in the
        on_train_loop_registration(). Otherwise logic implemented here will not be executed by the TrainLoop.

        Returns:

        """
        pass

    def on_after_optimizer_step(self):
        """Logic executed after the optimizer does a new step and updates the model weights

        To ensure the execution of this callback enable the `self.train_loop_obj.grad_cb_used = True` option in the
        on_train_loop_registration(). Otherwise logic implemented here will not be executed by the TrainLoop.

        Returns:

        """
        pass


class AbstractExperimentCallback(AbstractCallback):
    def __init__(self, callback_name, execution_order=0):
        """Extension of the AbstractCallback implementing the automatic experiment details inference from TrainLoop

        This abstract callback is inherited from when the implemented callbacks intend to save results files into the
        experiment folder and also potentially upload them to AWS S3.

        Args:
            callback_name (str): name of the callback
            execution_order (int): order of the callback execution. If all the used callbacks have the orders set to 0,
                than the callbacks are executed in the order they were registered.
        """
        AbstractCallback.__init__(self, callback_name, execution_order)
        self.project_name = None
        self.experiment_name = None
        self.local_model_result_folder_path = None

        # self.cloud_save_mode = 's3'
        # self.bucket_name = 'model-result'
        # self.cloud_dir_prefix = ''

    def try_infer_experiment_details(self, infer_cloud_details):
        """Infer paths where to save experiment related files from the running TrainLoop.

        This details inference function should only be called after the callback has already been registered in the
        TrainLoop, e.g. in the on_train_loop_registration().

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

            if infer_cloud_details:
                if self.cloud_save_mode == 's3' and \
                        hasattr(self.train_loop_obj,
                                'cloud_save_mode') and self.cloud_save_mode != self.train_loop_obj.cloud_save_mode:
                    self.cloud_save_mode = self.train_loop_obj.cloud_save_mode
                if self.bucket_name == 'model-result' and \
                        hasattr(self.train_loop_obj,
                                'bucket_name') and self.bucket_name != self.train_loop_obj.bucket_name:
                    self.bucket_name = self.train_loop_obj.bucket_name
                if self.cloud_dir_prefix == '' and \
                        hasattr(self.train_loop_obj,
                                'cloud_dir_prefix') and self.cloud_dir_prefix != self.train_loop_obj.cloud_dir_prefix:
                    self.cloud_dir_prefix = self.train_loop_obj.cloud_dir_prefix
        except AttributeError:
            raise AttributeError('Currently used TrainLoop does not support automatic project folder structure '
                                 'creation. Project name, etc. thus can not be automatically deduced. Please provide'
                                 'it in the callback parameters instead of currently used None values.')