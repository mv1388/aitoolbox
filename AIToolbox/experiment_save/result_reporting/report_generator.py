import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.style as style
style.use('ggplot')


class TrainingHistoryPlotter:
    def __init__(self, result_package, experiment_results_local_path, plots_folder_name='plots'):
        """

        Args:
            result_package (AIToolbox.experiment_save.result_package.abstract_result_packages.AbstractResultPackage):
            experiment_results_local_path (str):
            plots_folder_name (str):
        """
        self.result_package = result_package
        self.training_history = result_package.get_training_history_object()

        self.plots_folder_name = plots_folder_name
        self.plots_local_folder_path = os.path.join(experiment_results_local_path, plots_folder_name)
        if not os.path.exists(self.plots_local_folder_path):
            os.mkdir(self.plots_local_folder_path)

    def generate_report(self):
        """

        Returns:
            list:
        """
        plots_paths = []

        for metric_name, result_history in self.training_history.get_train_history_dict(flatten_dict=True).items():
            if len(result_history) > 1:
                file_name, file_path = self.plot_performance_curve(metric_name, result_history, self.plots_local_folder_path)
                plots_paths.append([os.path.join(self.plots_folder_name, file_name), file_path])

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
                          markers=True)

        ax.set_xlabel("Epoch", size=10)
        ax.set_ylabel(metric_name, size=10)
        ax.set_title(metric_name, size=10)

        fig.savefig(file_path)
        return file_name, file_path
