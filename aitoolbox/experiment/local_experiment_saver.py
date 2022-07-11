import datetime
import time

from aitoolbox.experiment.experiment_saver import AbstractExperimentSaver
from aitoolbox.experiment.local_save.local_model_save import PyTorchLocalModelSaver, KerasLocalModelSaver, AbstractLocalModelSaver
from aitoolbox.experiment.local_save.local_results_save import LocalResultsSaver


class BaseFullExperimentLocalSaver(AbstractExperimentSaver):
    def __init__(self, model_saver, project_name, experiment_name, local_model_result_folder_path='~/project/model_result'):
        """Base functionality class common to all the full experiment local saver derivations

        Args:
            model_saver (aitoolbox.experiment.local_save.local_model_save.AbstractLocalModelSaver): selected model
                saver implementing the saving logic for the desired framework
            project_name (str): root name of the project
            experiment_name (str): name of the particular experiment
            local_model_result_folder_path (str): root local path where project folder will be created
        """
        if not isinstance(model_saver, AbstractLocalModelSaver):
            raise TypeError(f'model_saver must be inherited from AbstractLocalModelSaver. '
                            f'model_saver type is: {type(model_saver)}')

        self.project_name = project_name
        self.experiment_name = experiment_name

        self.model_saver = model_saver
        self.results_saver = LocalResultsSaver(local_model_result_folder_path)

    def save_experiment(self, model, result_package, training_history, experiment_timestamp=None,
                        save_true_pred_labels=False, separate_files=False,
                        protect_existing_folder=True):
        """Save the experiment with the provided model saver

        Args:
            model (dict or keras.Model): model representation.
                If used with PyTorch it is a simple dict under the hood.
                In the case of Keras training this would be the keras Model.
            result_package (aitoolbox.experiment.result_package.abstract_result_packages.AbstractResultPackage):
                selected result package which will be evaluated to produce the performance results
            training_history (aitoolbox.experiment.training_history.TrainingHistory):
            experiment_timestamp (str): time stamp at the start of training
            save_true_pred_labels (bool): should ground truth labels also be saved
            separate_files (bool): should the results be saved in separate pickle files or should all the results
                be batched together in a single results file
            protect_existing_folder (bool): can override potentially already existing folder or not

        Returns:
            list: local model and results paths
        """
        if experiment_timestamp is None:
            experiment_timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S')

        _, model_local_path = self.model_saver.save_model(model=model,
                                                          project_name=self.project_name,
                                                          experiment_name=self.experiment_name,
                                                          experiment_timestamp=experiment_timestamp,
                                                          protect_existing_folder=protect_existing_folder)

        saved_paths = [model_local_path]

        if not separate_files:
            saved_local_results_details = \
                self.results_saver.save_experiment_results(result_package=result_package,
                                                           training_history=training_history,
                                                           project_name=self.project_name,
                                                           experiment_name=self.experiment_name,
                                                           experiment_timestamp=experiment_timestamp,
                                                           save_true_pred_labels=save_true_pred_labels,
                                                           protect_existing_folder=protect_existing_folder)
        else:
            saved_local_results_details = \
                self.results_saver.save_experiment_results_separate_files(result_package=result_package,
                                                                          training_history=training_history,
                                                                          project_name=self.project_name,
                                                                          experiment_name=self.experiment_name,
                                                                          experiment_timestamp=experiment_timestamp,
                                                                          save_true_pred_labels=save_true_pred_labels,
                                                                          protect_existing_folder=protect_existing_folder)

        saved_paths += [path for _, path in saved_local_results_details]

        return saved_paths


class FullPyTorchExperimentLocalSaver(BaseFullExperimentLocalSaver):
    def __init__(self, project_name, experiment_name, local_model_result_folder_path='~/project/model_result'):
        """PyTorch local experiment saver

        Args:
            project_name (str): root name of the project
            experiment_name (str): name of the particular experiment
            local_model_result_folder_path (str): root local path where project folder will be created
        """
        BaseFullExperimentLocalSaver.__init__(self, PyTorchLocalModelSaver(local_model_result_folder_path),
                                              project_name, experiment_name,
                                              local_model_result_folder_path=local_model_result_folder_path)


class FullKerasExperimentLocalSaver(BaseFullExperimentLocalSaver):
    def __init__(self, project_name, experiment_name, local_model_result_folder_path='~/project/model_result'):
        """Keras local experiment saver

        Args:
            project_name (str): root name of the project
            experiment_name (str): name of the particular experiment
            local_model_result_folder_path (str): root local path where project folder will be created
        """
        BaseFullExperimentLocalSaver.__init__(self, KerasLocalModelSaver(local_model_result_folder_path),
                                              project_name, experiment_name,
                                              local_model_result_folder_path=local_model_result_folder_path)
