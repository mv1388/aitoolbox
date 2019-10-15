import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.style as style
import torch

from aitoolbox.torchtrain.callbacks.callbacks import AbstractCallback
from aitoolbox.experiment.local_save.folder_create import ExperimentFolderCreator
from aitoolbox.cloud.AWS.results_save import BaseResultsSaver as BaseResultsS3Saver
from aitoolbox.cloud.GoogleCloud.results_save import BaseResultsGoogleStorageSaver

style.use('ggplot')


class GradientCallbackBase(AbstractCallback):
    def __init__(self, callback_name, execution_order=0):
        """Base abstract class for gradient related callbacks

        It has not implemented logic except for the the turning enabling of the grad_cb_used inside TrainLoop as part of
        the on_train_loop_registration(). Consequently, this potentially repeated task in every gradient calculation
        callback doesn't need to be done for every implemented callback.

        Args:
            callback_name (str): name of the callback
            execution_order (int): order of the callback execution. If all the used callbacks have the orders set to 0,
                than the callbacks are executed in the order they were registered.
        """
        AbstractCallback.__init__(self, callback_name, execution_order)

    def on_train_loop_registration(self):
        self.train_loop_obj.grad_cb_used = True


class GradNormClip(GradientCallbackBase):
    def __init__(self, max_norm, **kwargs):
        """Gradient norm clipping

        Args:
            max_norm (int or float): gradient clipping
            **kwargs:
        """
        GradientCallbackBase.__init__(self, 'Gradient clipping')
        self.max_norm = max_norm
        self.kwargs = kwargs

    def on_after_gradient_update(self):
        torch.nn.utils.clip_grad_norm_(self.train_loop_obj.model.parameters(), self.max_norm, **self.kwargs)


class GradientStatsPrint(AbstractCallback):
    def __init__(self, model_layers_extract_def, on_every_grad_update=False):
        """Model gradients statistics reporting

        Args:
            model_layers_extract_def: function/lambda accepting model as the input and returning a list of all
                the layers in the model for which the gradient stats should be calculated
            on_every_grad_update (bool): should the gradient stats be calculated on every gradient update, e.g. after
                every batch or only at the end of the epoch
        """
        AbstractCallback.__init__(self, 'Print model gradient stats')
        self.model_layers_extract_def = model_layers_extract_def
        self.on_every_grad_update = on_every_grad_update

    def on_train_loop_registration(self):
        if self.on_every_grad_update:
            self.train_loop_obj.grad_cb_used = True

    def on_after_gradient_update(self):
        if self.on_every_grad_update:
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


class GradDistributionPlot(AbstractCallback):
    def __init__(self, model_layers_extract_def,
                 project_name=None, experiment_name=None, local_model_result_folder_path=None,
                 cloud_save_mode='s3', bucket_name='model-result', cloud_dir_prefix=''):
        AbstractCallback.__init__(self, 'Gradient distribution plotter')
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.local_model_result_folder_path = os.path.expanduser(local_model_result_folder_path) \
            if local_model_result_folder_path is not None \
            else None
        self.cloud_save_mode = cloud_save_mode
        self.bucket_name = bucket_name
        self.cloud_dir_prefix = cloud_dir_prefix

        self.model_layers_extract_def = model_layers_extract_def
        self.cloud_results_saver = None

    def on_train_loop_registration(self):
        self.try_infer_experiment_details()

    def on_epoch_end(self):
        self.gradient_plot()

    def gradient_plot(self):
        grad_plot_dir_path = self.create_plot_dirs()
        grad_plot_dir_epoch_path = os.path.join(grad_plot_dir_path, f'epoch_{self.train_loop_obj.epoch}')
        os.mkdir(grad_plot_dir_epoch_path)

        model_layers_list = self.model_layers_extract_def(self.train_loop_obj.model)

        for i, layer in enumerate(model_layers_list):
            gradients = layer.weight.grad

            if gradients is not None:
                file_name = f'Layer_{i}.png'
                file_path = os.path.join(grad_plot_dir_epoch_path, file_name)

                gradients_flat = gradients.reshape(-1).cpu().numpy()

                fig = plt.figure()
                fig.set_size_inches(10, 8)

                ax = sns.distplot(gradients_flat)

                ax.set_xlabel("Gradient magnitude", size=10)
                ax.set_title(f'Gradient distribution for layer {i}', size=10)

                fig.savefig(file_path)
                plt.close()
            else:
                print(f'Layer {i} grad are None')

    def create_plot_dirs(self):
        experiment_path = \
            ExperimentFolderCreator.create_experiment_base_folder(self.project_name, self.experiment_name,
                                                                  self.train_loop_obj.experiment_timestamp,
                                                                  self.local_model_result_folder_path)
        results_dir_path = os.path.join(experiment_path, 'results')

        if not os.path.exists(results_dir_path):
            os.mkdir(results_dir_path)
        grad_plot_dir_path = os.path.join(results_dir_path, 'grad_distribution')
        if not os.path.exists(grad_plot_dir_path):
            os.mkdir(grad_plot_dir_path)

        return grad_plot_dir_path

    def prepare_results_saver(self):
        """

        Returns:
            None
        """
        if self.cloud_save_mode == 's3' or self.cloud_save_mode == 'aws_s3' or self.cloud_save_mode == 'aws':
            self.cloud_results_saver = BaseResultsS3Saver(bucket_name=self.bucket_name,
                                                          cloud_dir_prefix=self.cloud_dir_prefix)

        elif self.cloud_save_mode == 'gcs' or self.cloud_save_mode == 'google_storage' or self.cloud_save_mode == 'google storage':
            self.cloud_results_saver = BaseResultsGoogleStorageSaver(bucket_name=self.bucket_name,
                                                                     cloud_dir_prefix=self.cloud_dir_prefix)
        else:
            self.cloud_results_saver = None

    def try_infer_experiment_details(self):
        try:
            if self.project_name is None:
                self.project_name = self.train_loop_obj.project_name
            if self.experiment_name is None:
                self.experiment_name = self.train_loop_obj.experiment_name
            if self.local_model_result_folder_path is None:
                self.local_model_result_folder_path = self.train_loop_obj.local_model_result_folder_path

            if self.cloud_save_mode == 's3' and \
                    hasattr(self.train_loop_obj,
                            'cloud_save_mode') and self.cloud_save_mode != self.train_loop_obj.cloud_save_mode:
                self.cloud_save_mode = self.train_loop_obj.cloud_save_mode
            if self.bucket_name == 'model-result' and \
                    hasattr(self.train_loop_obj, 'bucket_name') and self.bucket_name != self.train_loop_obj.bucket_name:
                self.bucket_name = self.train_loop_obj.bucket_name
            if self.cloud_dir_prefix == '' and \
                    hasattr(self.train_loop_obj,
                            'cloud_dir_prefix') and self.cloud_dir_prefix != self.train_loop_obj.cloud_dir_prefix:
                self.cloud_dir_prefix = self.train_loop_obj.cloud_dir_prefix
        except AttributeError:
            raise AttributeError('Currently used TrainLoop does not support automatic project folder structure '
                                 'creation. Project name, etc. thus can not be automatically deduced. Please provide'
                                 'it in the callback parameters instead of currently used None values.')
