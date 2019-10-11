import os
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
            training_history (AIToolbox.experiment.training_history.TrainingHistory): TrainLoop training history
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
            metric_name: name of plotted metric
            result_history: results history for the selected metric
            results_local_folder_path: folder where the plot should be saved

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
        ax.set_title(metric_name, size=10)

        fig.savefig(file_path)
        plt.close()
        return file_name, file_path


class TrainingHistoryWriter:
    def __init__(self, experiment_results_local_path):
        """Write the calculated performance metrics in the training history into human-readable text file

        Args:
            experiment_results_local_path (str): path to the main experiment results folder on the local drive
        """
        self.experiment_results_local_path = experiment_results_local_path
        self.plots_local_folder_path = experiment_results_local_path

    def generate_report(self, training_history, epoch, file_name, results_folder_name=None):
        """Write all the currently present performance result in the training history into the text file

        Args:
            training_history (AIToolbox.experiment.training_history.TrainingHistory):
            epoch (int): current epoch
            file_name (str): output text file name
            results_folder_name (str or None): results folder path where the report file will be located

        Returns:
            str, str: file name/path inside the experiment folder, local file_path
        """
        if results_folder_name is not None:
            self.plots_local_folder_path = os.path.join(self.experiment_results_local_path, results_folder_name)
            if not os.path.exists(self.plots_local_folder_path):
                os.mkdir(self.plots_local_folder_path)

        file_path = os.path.join(self.plots_local_folder_path, file_name)

        with open(file_path, 'a') as f:
            f.write('============================\n')
            f.write(f'Epoch: {epoch}\n')
            f.write('============================\n')
            for metric_name, result_history in training_history.get_train_history_dict(flatten_dict=True).items():
                f.write(f'{metric_name}:\t{result_history[-1]}\n')
            f.write('\n\n')

        return os.path.join(results_folder_name if results_folder_name is not None else '',
                            file_name), file_path
