import os
import numpy as np
import torch

from aitoolbox.torchtrain.callbacks.abstract import AbstractCallback, AbstractExperimentCallback
from aitoolbox.torchtrain.multi_loss_optim import MultiOptimizer
from aitoolbox.experiment.local_save.local_results_save import BaseLocalResultsSaver
from aitoolbox.experiment.result_reporting.report_generator import GradientPlotter
from aitoolbox.cloud.AWS.results_save import BaseResultsSaver as BaseResultsS3Saver
from aitoolbox.cloud.GoogleCloud.results_save import BaseResultsGoogleStorageSaver
from aitoolbox.cloud import s3_available_options, gcs_available_options


class GradientCallbackBase(AbstractCallback):
    def __init__(self, callback_name, execution_order=0):
        """Base abstract class for gradient related callbacks

        It has not implemented logic except for the turning enabling of the grad_cb_used inside TrainLoop as part of
        the on_train_loop_registration(). Consequently, this potentially repeated task in every gradient calculation
        callback doesn't need to be done for every implemented callback.

        Args:
            callback_name (str): name of the callback
            execution_order (int): order of the callback execution. If all the used callbacks have the orders set to 0,
                then the callbacks are executed in the order they were registered.
        """
        AbstractCallback.__init__(self, callback_name, execution_order)

    def on_train_loop_registration(self):
        self.train_loop_obj.grad_cb_used = True


class GradValueClip(GradientCallbackBase):
    def __init__(self, max_grad_value):
        """Gradient value clipping

        Args:
            max_grad_value (int or float): maximum allowed value of the gradients
        """
        GradientCallbackBase.__init__(self, 'Gradient value clipping')
        self.max_grad_value = max_grad_value

    def on_after_gradient_update(self, optimizer_idx):
        if self.train_loop_obj.should_execute_optimizer_update():
            optimizer = self.train_loop_obj.optimizer
            if isinstance(optimizer, MultiOptimizer):
                optimizer = optimizer[optimizer_idx]

            # Unscales the gradients of optimizer's assigned params in-place
            # Following: https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-clipping
            self.train_loop_obj.amp_scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_value_(self.train_loop_obj.model.parameters(), self.max_grad_value)


class GradNormClip(GradientCallbackBase):
    def __init__(self, max_grad_norm, **kwargs):
        """Gradient norm clipping

        Args:
            max_grad_norm (int or float): max norm of the gradients
            **kwargs: torch.nn.utils.clip_grad_norm_ additional arguments
        """
        GradientCallbackBase.__init__(self, 'Gradient norm clipping')
        self.max_grad_norm = max_grad_norm
        self.kwargs = kwargs

    def on_after_gradient_update(self, optimizer_idx):
        if self.train_loop_obj.should_execute_optimizer_update():
            optimizer = self.train_loop_obj.optimizer
            if isinstance(optimizer, MultiOptimizer):
                optimizer = optimizer[optimizer_idx]

            # Unscales the gradients of optimizer's assigned params in-place
            # Following: https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-clipping
            self.train_loop_obj.amp_scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(self.train_loop_obj.model.parameters(), self.max_grad_norm, **self.kwargs)


class GradientStatsPrint(AbstractCallback):
    def __init__(self, model_layers_extract_def, on_every_grad_update=False):
        """Model gradients statistics reporting

        Args:
            model_layers_extract_def (lambda or function): lambda/function accepting model as the input and returning
                a list of all the layers in the model for which the gradient stats should be calculated
            on_every_grad_update (bool): should the gradient stats be calculated on every gradient update, e.g. after
                every batch or only at the end of the epoch
        """
        AbstractCallback.__init__(self, 'Print model gradient stats', device_idx_execution=0)
        self.model_layers_extract_def = model_layers_extract_def
        self.on_every_grad_update = on_every_grad_update

    def on_train_loop_registration(self):
        if self.on_every_grad_update:
            self.train_loop_obj.grad_cb_used = True

    def on_after_gradient_update(self, optimizer_idx):
        if self.on_every_grad_update and optimizer_idx == 0:
            self.gradients_report()

    def on_epoch_end(self):
        self.gradients_report()

    def gradients_report(self):
        model_layers_list = self.model_layers_extract_def(self.train_loop_obj.model)

        print('---> Model layers gradients stats')
        for i, layer in enumerate(model_layers_list):
            gradients = layer.weight.grad

            if gradients is not None:
                gradients = gradients.cpu().numpy()

                mu = np.mean(gradients)
                std = np.std(gradients)

                print(f'Layer {i} grads: Mean: {mu}; Std {std}')
                print(f'\tRatio of zero gradients: {float(np.count_nonzero(gradients == 0)) / gradients.size}')
            else:
                print(f'Layer {i} grad are None')


class GradDistributionPlot(AbstractExperimentCallback):
    def __init__(self, model_layers_extract_def, grad_plots_dir_name='grad_distribution', file_format='png',
                 project_name=None, experiment_name=None, local_model_result_folder_path=None,
                 cloud_save_mode=None, bucket_name=None, cloud_dir_prefix=None):
        """Plot layers' gradient distributions after every epoch

        Args:
            model_layers_extract_def (lambda or function): lambda/function accepting model as the input and returning
                a list of all the layers in the model for which the gradient stats should be calculated
            grad_plots_dir_name (str): name of the folder where gradient distribution plots are saved after every epoch
            file_format (str): output file format. Can be either 'png' for saving separate images or 'pdf' for combining
                all the plots into a single pdf file.
            project_name (str or None): root name of the project
            experiment_name (str or None): name of the particular experiment
            local_model_result_folder_path (str or None): root local path where project folder will be created
            cloud_save_mode (str or None): Storage destination selector.
                For AWS S3: 's3' / 'aws_s3' / 'aws'
                For Google Cloud Storage: 'gcs' / 'google_storage' / 'google storage'
                Everything else results just in local storage to disk
            bucket_name (str): name of the bucket in the cloud storage
            cloud_dir_prefix (str): path to the folder inside the bucket where the experiments are going to be saved
        """
        AbstractExperimentCallback.__init__(self, 'Gradient distribution plotter',
                                            project_name, experiment_name, local_model_result_folder_path,
                                            cloud_save_mode, bucket_name, cloud_dir_prefix,
                                            device_idx_execution=0)
        self.grad_plots_dir_name = grad_plots_dir_name
        self.file_format = file_format
        if self.file_format not in ['png', 'pdf']:
            raise ValueError(f"Output format '{self.file_format}' is not supported. "
                             "Select one of the following: 'png' or 'pdf'.")

        self.model_layers_extract_def = model_layers_extract_def
        self.cloud_results_saver = None

        self.gradient_plotter = None

    def on_train_loop_registration(self):
        self.try_infer_experiment_details(infer_cloud_details=True)
        self.prepare_results_saver()

    def on_epoch_end(self):
        self.gradient_plot()

    def gradient_plot(self):
        grad_plot_dir_path = self.create_plot_dirs()
        if self.gradient_plotter is None:
            self.gradient_plotter = GradientPlotter(experiment_grad_results_local_path=grad_plot_dir_path)

        model_layers_list = self.model_layers_extract_def(self.train_loop_obj.model)
        model_layer_gradients = [layer.weight.grad.reshape(-1).cpu().numpy() if layer.weight.grad is not None else None
                                 for layer in model_layers_list]

        saved_plot_paths = self.gradient_plotter.generate_report(model_layer_gradients, f'epoch_{self.train_loop_obj.epoch}',
                                                                 file_format=self.file_format)

        if self.cloud_results_saver is not None:
            self.save_to_cloud(saved_plot_paths)

    def save_to_cloud(self, saved_plot_paths):
        experiment_cloud_path = \
            self.cloud_results_saver.create_experiment_cloud_storage_folder_structure(self.project_name,
                                                                                      self.experiment_name,
                                                                                      self.train_loop_obj.experiment_timestamp)
        grad_plots_dir_path = os.path.join(experiment_cloud_path, self.grad_plots_dir_name)

        for file_path_in_cloud_grad_results_dir, local_file_path in saved_plot_paths:
            plot_file_cloud_path = os.path.join(grad_plots_dir_path, file_path_in_cloud_grad_results_dir)
            self.cloud_results_saver.save_file(local_file_path=local_file_path,
                                               cloud_file_path=plot_file_cloud_path)

    def create_plot_dirs(self):
        experiment_results_local_path = \
            BaseLocalResultsSaver.create_experiment_local_results_folder(self.project_name, self.experiment_name,
                                                                         self.train_loop_obj.experiment_timestamp,
                                                                         self.local_model_result_folder_path)
        grad_plots_dir_path = os.path.join(experiment_results_local_path, self.grad_plots_dir_name)
        if not os.path.exists(grad_plots_dir_path):
            os.mkdir(grad_plots_dir_path)

        return grad_plots_dir_path

    def prepare_results_saver(self):
        if self.cloud_save_mode in s3_available_options:
            self.cloud_results_saver = BaseResultsS3Saver(bucket_name=self.bucket_name,
                                                          cloud_dir_prefix=self.cloud_dir_prefix)

        elif self.cloud_save_mode in gcs_available_options:
            self.cloud_results_saver = BaseResultsGoogleStorageSaver(bucket_name=self.bucket_name,
                                                                     cloud_dir_prefix=self.cloud_dir_prefix)
        else:
            self.cloud_results_saver = None
