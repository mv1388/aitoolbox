import os
import csv
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.style as style
style.use('ggplot')


class TrainingHistoryPlotter:
    def __init__(self, experiment_results_local_path):
        """Plot the calculated performance metrics in the training history

        Args:
            experiment_results_local_path (str): path to the main experiment results folder on the local drive
        """
        self.experiment_results_local_path = experiment_results_local_path

    def generate_report(self, training_history, plots_folder_name='plots'):
        """Plot all the currently present performance result in the training history

        Every plot shows the progression of a single performance metric over the epochs.

        Args:
            training_history (aitoolbox.experiment.training_history.TrainingHistory): TrainLoop training history
            plots_folder_name (str): local dir name where the plots should be saved

        Returns:
            list: list of saved plot paths
        """
        plots_local_folder_path = os.path.join(self.experiment_results_local_path, plots_folder_name)
        if not os.path.exists(plots_local_folder_path):
            os.mkdir(plots_local_folder_path)

        plots_paths = []

        for metric_name, result_history in training_history.get_train_history_dict(flatten_dict=True).items():
            if len(result_history) > 1:
                file_name, file_path = self.plot_performance_curve(metric_name, result_history, plots_local_folder_path)
                plots_paths.append([os.path.join(plots_folder_name, file_name), file_path])

        return plots_paths

    @staticmethod
    def plot_performance_curve(metric_name, result_history, results_local_folder_path):
        """Plot the performance of a selected calculated metric over the epochs

        Args:
            metric_name (str or int): name of plotted metric
            result_history (list or np.array): results history for the selected metric
            results_local_folder_path (str): path to the folder where the plot should be saved

        Returns:
            (str, str): file_name, file_path
        """
        file_name = f'{metric_name}.png'
        file_path = os.path.join(results_local_folder_path, file_name)

        fig = plt.figure()
        fig.set_size_inches(10, 8)
        
        ax = sns.lineplot(x=list(range(len(result_history))), y=result_history,
                          markers='o')

        ax.set_xlabel("Epoch", size=10)
        ax.set_ylabel(metric_name, size=10)

        # Adding plot title and subtitles
        ax.text(s=metric_name, x=0.5, y=1.07, fontsize=16, weight='bold', ha='center', va='bottom',
                transform=ax.transAxes)
        ax.text(s=f'Max result: {max(result_history)} {metric_name} (at epoch {np.argmax(result_history)})',
                x=0.5, y=1.035, fontsize=8, alpha=0.75,
                ha='center', va='bottom', transform=ax.transAxes)
        ax.text(s=f'Min result: {min(result_history)} {metric_name} (at epoch {np.argmin(result_history)})',
                x=0.5, y=1.01, fontsize=8, alpha=0.75,
                ha='center', va='bottom', transform=ax.transAxes)

        fig.savefig(file_path)
        plt.close()
        return file_name, file_path


class TrainingHistoryWriter:
    def __init__(self, experiment_results_local_path):
        """Write the calculated performance metrics in the training history into human-readable text file

        Args:
            experiment_results_local_path (str or None): path to the main experiment results folder on the local drive
        """
        self.experiment_results_local_path = experiment_results_local_path
        self.metric_name_cols = None

    def generate_report(self, training_history, epoch, file_name, results_folder_name='', file_format='txt'):
        """Write all the currently present performance result in the training history into the text file

        Args:
            training_history (aitoolbox.experiment.training_history.TrainingHistory):
            epoch (int): current epoch
            file_name (str): output text file name
            results_folder_name (str): results folder path where the report file will be located
            file_format (str): output file format. Can be either 'txt' human readable output or
                'tsv' for a tabular format or 'csv' for comma separated format.

        Returns:
            str, str: file name/path inside the experiment folder, local file_path
        """
        results_write_local_folder_path = os.path.join(self.experiment_results_local_path, results_folder_name)
        if not os.path.exists(results_write_local_folder_path):
            os.mkdir(results_write_local_folder_path)

        file_path = os.path.join(results_write_local_folder_path, file_name)

        if file_format == 'txt':
            self.write_txt(training_history, epoch, file_path)
        elif file_format == 'tsv':
            self.write_csv_tsv(training_history, epoch, file_path, delimiter='\t')
        elif file_format == 'csv':
            self.write_csv_tsv(training_history, epoch, file_path, delimiter=',')
        else:
            raise ValueError(f"Output format '{file_format}' is not supported. "
                             "Select one of the following: txt, tsv, csv.")

        return os.path.join(results_folder_name if results_folder_name is not None else '',
                            file_name), file_path

    @staticmethod
    def write_txt(training_history, epoch, file_path):
        with open(file_path, 'a') as f:
            f.write('============================\n')
            f.write(f'Epoch: {epoch}\n')
            f.write('============================\n')
            for metric_name, result_history in training_history.get_train_history_dict(flatten_dict=True).items():
                f.write(f'{metric_name}:\t{result_history[-1]}\n')
            f.write('\n\n')

    def write_csv_tsv(self, training_history, epoch, file_path, delimiter):
        with open(file_path, 'a') as f:
            tsv_writer = csv.writer(f, delimiter=delimiter)
            current_metric_names = list(training_history.get_train_history_dict(flatten_dict=True).keys())

            if self.metric_name_cols is None:
                self.metric_name_cols = current_metric_names
                tsv_writer.writerow(['Epoch'] + self.metric_name_cols)

            if sorted(current_metric_names) != sorted(self.metric_name_cols):
                self.metric_name_cols = current_metric_names
                tsv_writer.writerow(['NEW_METRICS_DETECTED'])
                tsv_writer.writerow(['Epoch'] + self.metric_name_cols)

            training_history_dict = training_history.get_train_history_dict(flatten_dict=True)
            tsv_writer.writerow([epoch] + [training_history_dict[metric_name][-1] for metric_name in self.metric_name_cols])


class GradientPlotter:
    def __init__(self, experiment_grad_results_local_path):
        """Plot the gradient distributions for model's layers

        Args:
            experiment_grad_results_local_path (str): path to the main experiment results folder on the local drive
        """
        self.experiment_grad_results_local_path = experiment_grad_results_local_path

    def generate_report(self, model_layer_gradients, grad_plots_folder_name='grad_plots'):
        """Plot all the gradient distributions for the layers in the model

        Args:
            model_layer_gradients (list): list of model's gradients
            grad_plots_folder_name (str): name of the folder where gradient distribution plots will be saved

        Returns:
            list: list of saved plot paths: [file_path_in_cloud_grad_results_dir, local_file_path]
        """
        grad_plots_local_folder_path = os.path.join(self.experiment_grad_results_local_path, grad_plots_folder_name)
        if not os.path.exists(grad_plots_local_folder_path):
            os.mkdir(grad_plots_local_folder_path)

        saved_plot_paths = []

        for i, gradients in enumerate(model_layer_gradients):
            if gradients is not None:
                file_name, file_path = self.plot_gradient_distribution(gradients, i, grad_plots_local_folder_path)

                saved_plot_paths.append([os.path.join(grad_plots_folder_name, file_name), file_path])
            else:
                print(f'Layer {i} grad are None')

        return saved_plot_paths

    @staticmethod
    def plot_gradient_distribution(gradients, layer_name, grad_plots_local_folder_path):
        """Plot and save to file the distribution of the single layer's gradients

        Args:
            gradients (list or np.array): a flattened list  of gradients from a single layer
            layer_name (str or int): name or index of the layer
            grad_plots_local_folder_path (str): path to the folder where the plot should be saved

        Returns:
            (str, str): file_name, file_path
        """
        file_name = f'layer_{layer_name}.png'
        file_path = os.path.join(grad_plots_local_folder_path, file_name)

        fig = plt.figure()
        fig.set_size_inches(10, 8)

        ax = sns.distplot(gradients)
        ax.set_xlabel("Gradient magnitude", size=10)
        ax.set_title(f'Gradient distribution for layer {layer_name}', size=10)

        fig.savefig(file_path)
        plt.close()

        return file_name, file_path
