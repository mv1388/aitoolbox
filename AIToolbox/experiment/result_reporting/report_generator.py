import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.style as style
style.use('ggplot')


class TrainingHistoryPlotter:
    def __init__(self, experiment_results_local_path):
        """

        Args:
            experiment_results_local_path (str):
        """
        self.experiment_results_local_path = experiment_results_local_path

    def generate_report(self, training_history, plots_folder_name='plots'):
        """

        Args:
            training_history (AIToolbox.experiment.training_history.TrainingHistory):
            plots_folder_name (str):

        Returns:
            list:
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
        """

        Args:
            metric_name:
            result_history:
            results_local_folder_path:

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
        """

        Args:
            experiment_results_local_path (str):
        """
        self.experiment_results_local_path = experiment_results_local_path
        self.plots_local_folder_path = None

    def generate_report(self, training_history, epoch, file_name, results_folder_name='results_txt'):
        """

        Args:
            training_history (AIToolbox.experiment.training_history.TrainingHistory):
            epoch (int):
            file_name (str):
            results_folder_name (str):

        Returns:
            str, str: file name/path inside the experiment folder, local file_path
        """
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

        return os.path.join(results_folder_name, file_name), file_path
