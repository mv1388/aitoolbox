import unittest
import random
import os
import datetime
import time
import shutil
import pickle
import json

from aitoolbox.experiment.local_save.local_results_save import *
from aitoolbox.experiment.result_package.abstract_result_packages import AbstractResultPackage
from aitoolbox.experiment.training_history import TrainingHistory


THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class DummyTrainingHistory(TrainingHistory):
    def __init__(self):
        TrainingHistory.__init__(self)

    def _build_epoch_list(self):
        return []


class DummyFullResultPackage(AbstractResultPackage):
    def __init__(self, result_dict, hyper_params, additional_results=None):
        AbstractResultPackage.__init__(self, 'dummyFullPkg')
        self.result_dict = result_dict
        self.hyper_params = hyper_params
        self.y_true = [10.0] * 100
        self.y_predicted = [123.4] * 100

        self.additional_results = additional_results

    def prepare_results_dict(self):
        return self.result_dict

    def get_results(self):
        return self.prepare_results_dict()

    def get_hyperparameters(self):
        return self.hyper_params

    def list_additional_results_dump_paths(self):
        return self.additional_results


class TestBaseLocalResultsSaver(unittest.TestCase):
    def test_init(self):
        short_path = '~/dadada/dfeefe'
        saver = BaseLocalResultsSaver(local_model_result_folder_path=short_path)
        self.assertEqual(saver.local_model_result_folder_path, os.path.expanduser(short_path))

        saver_pickle = BaseLocalResultsSaver(THIS_DIR, 'pickle')
        self.assertEqual(saver_pickle.file_format, 'pickle')

        saver_json = BaseLocalResultsSaver(THIS_DIR, 'json')
        self.assertEqual(saver_json.file_format, 'json')

        saver_unsupported_format = BaseLocalResultsSaver(THIS_DIR, 'my_fancy_format')
        self.assertEqual(saver_unsupported_format.file_format, 'pickle')

    def test_create_experiment_local_folder_structure(self):
        project_dir_name = 'projectDir'
        exp_dir_name = 'experimentSubDir'
        current_time = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S')

        base_final = BaseLocalResultsSaver(local_model_result_folder_path=THIS_DIR)
        path = base_final.create_experiment_local_folder_structure(project_dir_name, exp_dir_name, current_time)

        project_path = os.path.join(THIS_DIR, project_dir_name)
        exp_path = os.path.join(project_path, f'{exp_dir_name}_{current_time}')
        model_path = os.path.join(exp_path, 'results')

        self.assertTrue(os.path.exists(project_path))
        self.assertTrue(os.path.exists(exp_path))
        self.assertTrue(os.path.exists(model_path))
        self.assertTrue(os.path.exists(path))
        self.assertEqual(model_path, path)

        if os.path.exists(project_path):
            shutil.rmtree(project_path)

    def test_create_experiment_local_folder_structure_calls_paths_sub_fn(self):
        project_dir_name = 'projectDir'
        exp_dir_name = 'experimentSubDir'
        current_time = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S')
        project_path = os.path.join(THIS_DIR, project_dir_name)

        base_final = BaseLocalResultsSaver(local_model_result_folder_path=THIS_DIR)
        path = base_final.create_experiment_local_folder_structure(project_dir_name, exp_dir_name, current_time)

        paths_list_obj = base_final.get_experiment_local_results_folder_paths(project_path, exp_dir_name, current_time, THIS_DIR)
        paths_list_class = BaseLocalResultsSaver.get_experiment_local_results_folder_paths(project_dir_name, exp_dir_name,
                                                                                           current_time, THIS_DIR)
        self.assertEqual(paths_list_obj, paths_list_class)

        expected_path = os.path.join(THIS_DIR, project_dir_name, f'{exp_dir_name}_{current_time}', 'results')
        self.assertEqual(path, expected_path)

        if os.path.exists(project_path):
            shutil.rmtree(project_path)

    def test_form_experiment_local_folders_paths(self):
        project_dir_name = 'projectDir'
        exp_dir_name = 'experimentSubDir'
        current_time = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S')

        project_dir_path_true = os.path.join(THIS_DIR, project_dir_name)
        experiment_dir_path_true = os.path.join(project_dir_path_true, f'{exp_dir_name}_{current_time}')
        experiment_results_dir_path_true = os.path.join(experiment_dir_path_true, 'results')

        paths = BaseLocalResultsSaver.get_experiment_local_results_folder_paths(project_dir_name, exp_dir_name, current_time,
                                                                                THIS_DIR)
        project_dir_path, experiment_dir_path, experiment_results_dir_path = paths

        self.assertEqual(len(paths), 3)
        self.assertEqual(project_dir_path, project_dir_path_true)
        self.assertEqual(experiment_dir_path, experiment_dir_path_true)
        self.assertEqual(experiment_results_dir_path, experiment_results_dir_path_true)

    def test_save_file_pickle(self):
        self.save_file_result('pickle', '.p')

    def test_save_file_json(self):
        self.save_file_result('json', '.json')

    def test_save_file_unsupported_format(self):
        self.save_file_result('my_fancy_format', '.p')

    def save_file_result(self, file_format, expected_extension):
        project_dir_name = 'projectDir'
        exp_dir_name = 'experimentSubDir'
        current_time = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S')
        file_name = 'test_dump'

        project_dir_path_true = os.path.join(THIS_DIR, project_dir_name)
        experiment_dir_path_true = os.path.join(project_dir_path_true, f'{exp_dir_name}_{current_time}')
        experiment_results_dir_path_true = os.path.join(experiment_dir_path_true, 'results')

        result_dict = {'acc': 10, 'loss': 101010.2, 'rogue': 4445.5}

        saver = BaseLocalResultsSaver(local_model_result_folder_path=THIS_DIR, file_format=file_format)
        saver.create_experiment_local_folder_structure(project_dir_name, exp_dir_name, current_time)
        saver.save_file(result_dict, file_name, f'{experiment_results_dir_path_true}/{file_name}')

        self.assertTrue(os.path.exists(f'{experiment_results_dir_path_true}/{file_name}{expected_extension}'))

        with open(f'{experiment_results_dir_path_true}/{file_name}{expected_extension}', 'rb') as f:
            if expected_extension == '.p':
                read_result_dict = pickle.load(f)
            elif expected_extension == '.json':
                read_result_dict = json.load(f)
            else:
                read_result_dict = pickle.load(f)

        self.assertEqual(result_dict, read_result_dict)

        if os.path.exists(project_dir_path_true):
            shutil.rmtree(project_dir_path_true)

    def test_forced_unsupported_file_format_error(self):
        project_dir_name = 'projectDir'
        exp_dir_name = 'experimentSubDir'
        current_time = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S')
        file_name = 'test_dump'

        project_dir_path_true = os.path.join(THIS_DIR, project_dir_name)
        experiment_dir_path_true = os.path.join(project_dir_path_true, f'{exp_dir_name}_{current_time}')
        experiment_results_dir_path_true = os.path.join(experiment_dir_path_true, 'results')

        result_dict = {'acc': 10, 'loss': 101010.2, 'rogue': 4445.5}

        saver = BaseLocalResultsSaver(local_model_result_folder_path=THIS_DIR, file_format='my_fancy_format')

        self.assertEqual(saver.file_format, 'pickle')

        saver.create_experiment_local_folder_structure(project_dir_name, exp_dir_name, current_time)

        # Force format re-set to unsupported my_fancy_format
        saver.file_format = 'my_fancy_format'

        with self.assertRaises(ValueError):
            saver.save_file(result_dict, file_name, f'{experiment_results_dir_path_true}/{file_name}')

        if os.path.exists(project_dir_path_true):
            shutil.rmtree(project_dir_path_true)


class TestLocalResultsSaverSingleFile(unittest.TestCase):
    def test_save_experiment_results_pickle(self):
        self.save_experiment_results(file_format='pickle', expected_extension='.p', save_true_pred_labels=True)
        self.save_experiment_results(file_format='pickle', expected_extension='.p', save_true_pred_labels=False)

    def test_save_experiment_results_json(self):
        self.save_experiment_results(file_format='json', expected_extension='.json', save_true_pred_labels=True)
        self.save_experiment_results(file_format='json', expected_extension='.json', save_true_pred_labels=False)

    def test_save_experiment_results_unsupported_format(self):
        self.save_experiment_results(file_format='my_fancy_format', expected_extension='.p', save_true_pred_labels=True)
        self.save_experiment_results(file_format='my_fancy_format', expected_extension='.p', save_true_pred_labels=False)

    def save_experiment_results(self, file_format, expected_extension, save_true_pred_labels):
        project_dir_name = 'projectDir'
        exp_dir_name = 'experimentSubDir'
        current_time = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S')
        result_pkg = DummyFullResultPackage({'metric1': 33434, 'acc1': 223.43, 'loss': 4455.6},
                                            {'epoch': 20, 'lr': 0.334})
        training_history = DummyTrainingHistory().wrap_pre_prepared_history({})
        result_file_name_true = f'results_hyperParams_hist_{exp_dir_name}_{current_time}{expected_extension}'

        project_path = os.path.join(THIS_DIR, project_dir_name)
        exp_path = os.path.join(project_path, f'{exp_dir_name}_{current_time}')
        results_path = os.path.join(exp_path, 'results')
        result_file_path_true = os.path.join(results_path, result_file_name_true)

        saver = LocalResultsSaver(local_model_result_folder_path=THIS_DIR, file_format=file_format)
        experiment_results_paths = saver.save_experiment_results(result_pkg, training_history,
                                                                 project_dir_name, exp_dir_name,
                                                                 current_time,
                                                                 save_true_pred_labels=save_true_pred_labels)
        self.assertEqual(len(experiment_results_paths), 1)
        result_file_name, result_file_path = experiment_results_paths[0]

        self.assertTrue(os.path.exists(project_path))
        self.assertTrue(os.path.exists(exp_path))
        self.assertTrue(os.path.exists(results_path))
        self.assertTrue(os.path.exists(result_file_path))

        self.assertEqual(result_file_name, result_file_name_true)
        self.assertEqual(result_file_path, result_file_path_true)

        with open(result_file_path, 'rb') as f:
            if expected_extension == '.p':
                read_result_dict = pickle.load(f)
            elif expected_extension == '.json':
                read_result_dict = json.load(f)
            else:
                read_result_dict = pickle.load(f)

        self.assertEqual(read_result_dict['experiment_name'], exp_dir_name)
        self.assertEqual(read_result_dict['experiment_results_local_path'], results_path)
        self.assertEqual(read_result_dict['results'], result_pkg.result_dict)
        self.assertEqual(read_result_dict['hyperparameters'], result_pkg.hyper_params)
        self.assertEqual(read_result_dict['training_history'], training_history.get_train_history())

        if save_true_pred_labels:
            self.assertEqual(read_result_dict['y_true'], result_pkg.y_true)
            self.assertEqual(read_result_dict['y_predicted'], result_pkg.y_predicted)

        if os.path.exists(project_path):
            shutil.rmtree(project_path)

    def test_experiment_timestamp_not_provided(self):
        project_dir_name = 'projectDir'
        exp_dir_name = 'experimentSubDir'
        current_time = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S')
        file_format = 'pickle'
        expected_extension = '.p'
        result_pkg = DummyFullResultPackage({'metric1': 33434, 'acc1': 223.43, 'loss': 4455.6},
                                            {'epoch': 20, 'lr': 0.334})
        training_history = DummyTrainingHistory().wrap_pre_prepared_history({})
        result_file_name_true = f'results_hyperParams_hist_{exp_dir_name}_{current_time}{expected_extension}'

        project_path = os.path.join(THIS_DIR, project_dir_name)
        exp_path = os.path.join(project_path, f'{exp_dir_name}_{current_time}')
        results_path = os.path.join(exp_path, 'results')
        result_file_path_true = os.path.join(results_path, result_file_name_true)

        saver = LocalResultsSaver(local_model_result_folder_path=THIS_DIR, file_format=file_format)
        experiment_results_paths = saver.save_experiment_results(
            result_pkg, training_history,
            project_dir_name, exp_dir_name
        )

        self.assertEqual(result_file_path_true, experiment_results_paths[0][1])

        if os.path.exists(project_path):
            shutil.rmtree(project_path)

    def test_additional_results_dump_paths(self):
        project_dir_name = 'projectDir'
        exp_dir_name = 'experimentSubDir'
        current_time = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S')
        file_format = 'pickle'
        expected_extension = '.p'

        training_history = DummyTrainingHistory().wrap_pre_prepared_history({})
        result_file_name_true = f'results_hyperParams_hist_{exp_dir_name}_{current_time}{expected_extension}'

        project_path = os.path.join(THIS_DIR, project_dir_name)
        exp_path = os.path.join(project_path, f'{exp_dir_name}_{current_time}')
        results_path = os.path.join(exp_path, 'results')
        result_file_path_true = os.path.join(results_path, result_file_name_true)

        additional_results_paths = [
            ['BLAAAAA.txt', os.path.join(results_path, 'BLAAAAA.txt')],
            ['uuuuu.p', os.path.join(results_path, 'uuuuu.p')],
            ['aaaaaa.json', os.path.join(results_path, 'aaaaaa.json')]
        ]

        result_pkg = DummyFullResultPackage(
            {'metric1': 33434, 'acc1': 223.43, 'loss': 4455.6},
            {'epoch': 20, 'lr': 0.334},
            additional_results=additional_results_paths
        )

        saver = LocalResultsSaver(local_model_result_folder_path=THIS_DIR, file_format=file_format)
        experiment_results_paths = saver.save_experiment_results(
            result_pkg, training_history,
            project_dir_name, exp_dir_name,
            current_time
        )

        self.assertEqual(
            [[result_file_name_true, result_file_path_true]] + additional_results_paths,
            experiment_results_paths
        )

        if os.path.exists(project_path):
            shutil.rmtree(project_path)


class TestLocalResultsSaverSeparateFiles(unittest.TestCase):
    def test_save_experiment_results_pickle(self):
        self.save_experiment_results(file_format='pickle', expected_extension='.p', save_true_pred_labels=True)
        self.save_experiment_results(file_format='pickle', expected_extension='.p', save_true_pred_labels=False)

    def test_save_experiment_results_json(self):
        self.save_experiment_results(file_format='json', expected_extension='.json', save_true_pred_labels=True)
        self.save_experiment_results(file_format='json', expected_extension='.json', save_true_pred_labels=False)

    def test_save_experiment_results_unsupported_format(self):
        self.save_experiment_results(file_format='my_fancy_format', expected_extension='.p', save_true_pred_labels=True)
        self.save_experiment_results(file_format='my_fancy_format', expected_extension='.p', save_true_pred_labels=False)

    def save_experiment_results(self, file_format, expected_extension, save_true_pred_labels):
        project_dir_name = 'projectDir'
        exp_dir_name = 'experimentSubDir'
        current_time = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S')
        result_pkg = DummyFullResultPackage({'metric1': 33434, 'acc1': 223.43, 'loss': 4455.6},
                                            {'epoch': 20, 'lr': 0.334})
        training_history = DummyTrainingHistory().wrap_pre_prepared_history({})
        result_file_name_true = f'results_{exp_dir_name}_{current_time}{expected_extension}'
        hyper_param_file_name_true = f'hyperparams_{exp_dir_name}_{current_time}{expected_extension}'
        train_hist_file_name_true = f'train_history_{exp_dir_name}_{current_time}{expected_extension}'
        labels_file_name_true = f'true_pred_labels_{exp_dir_name}_{current_time}{expected_extension}'

        project_path = os.path.join(THIS_DIR, project_dir_name)
        exp_path = os.path.join(project_path, f'{exp_dir_name}_{current_time}')
        results_path = os.path.join(exp_path, 'results')
        result_file_path_true = os.path.join(results_path, result_file_name_true)
        hyper_param_file_path_true = os.path.join(results_path, hyper_param_file_name_true)
        train_hist_file_path_true = os.path.join(results_path, train_hist_file_name_true)
        labels_file_path_true = os.path.join(results_path, labels_file_name_true)

        saver = LocalResultsSaver(local_model_result_folder_path=THIS_DIR, file_format=file_format)
        experiment_results_paths = saver.save_experiment_results_separate_files(result_pkg, training_history,
                                                                                project_dir_name, exp_dir_name,
                                                                                current_time,
                                                                                save_true_pred_labels=save_true_pred_labels)
        self.assertEqual(len(experiment_results_paths), 4 if save_true_pred_labels else 3)
        result_file_name, result_file_path = experiment_results_paths[0]
        hyper_param_file_name, hyper_param_file_path = experiment_results_paths[1]
        train_hist_file_name, train_hist_file_path = experiment_results_paths[2]

        self.assertEqual(result_file_name, result_file_name_true)
        self.assertEqual(hyper_param_file_name, hyper_param_file_name_true)
        self.assertEqual(train_hist_file_name, train_hist_file_name_true)

        self.assertTrue(os.path.exists(project_path))
        self.assertTrue(os.path.exists(exp_path))
        self.assertTrue(os.path.exists(results_path))
        self.assertTrue(os.path.exists(result_file_path))
        self.assertTrue(os.path.exists(hyper_param_file_path))
        self.assertTrue(os.path.exists(train_hist_file_path))

        self.assertEqual(result_file_path_true, result_file_path)
        self.assertEqual(hyper_param_file_path_true, hyper_param_file_path)
        self.assertEqual(train_hist_file_path_true, train_hist_file_path)

        if save_true_pred_labels:
            labels_file_name, labels_file_path = experiment_results_paths[3]
            self.assertEqual(labels_file_name, labels_file_name_true)
            self.assertTrue(os.path.exists(labels_file_path))
            self.assertEqual(labels_file_path_true, labels_file_path)

        def read_result_file(f_path):
            with open(f_path, 'rb') as f:
                if expected_extension == '.p':
                    read_dict = pickle.load(f)
                elif expected_extension == '.json':
                    read_dict = json.load(f)
                else:
                    read_dict = pickle.load(f)
            return read_dict

        read_result_dict = read_result_file(result_file_path)
        read_hyper_param_dict = read_result_file(hyper_param_file_path)
        read_train_hist_dict = read_result_file(train_hist_file_path)

        self.assertEqual(read_result_dict['experiment_name'], exp_dir_name)
        self.assertEqual(read_result_dict['experiment_results_local_path'], results_path)
        self.assertEqual(read_result_dict['results'], result_pkg.result_dict)

        self.assertEqual(read_hyper_param_dict['experiment_name'], exp_dir_name)
        self.assertEqual(read_hyper_param_dict['experiment_results_local_path'], results_path)
        self.assertEqual(read_hyper_param_dict['hyperparameters'], result_pkg.hyper_params)

        self.assertEqual(read_train_hist_dict['experiment_name'], exp_dir_name)
        self.assertEqual(read_train_hist_dict['experiment_results_local_path'], results_path)
        self.assertEqual(read_train_hist_dict['training_history'], training_history.get_train_history())

        if save_true_pred_labels:
            read_labels_dict = read_result_file(labels_file_path)
            self.assertEqual(read_labels_dict['y_true'], result_pkg.y_true)
            self.assertEqual(read_labels_dict['y_predicted'], result_pkg.y_predicted)

        if os.path.exists(project_path):
            shutil.rmtree(project_path)

    def test_experiment_timestamp_not_provided(self):
        project_dir_name = 'projectDir'
        exp_dir_name = 'experimentSubDir'
        current_time = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S')
        file_format = 'pickle'
        expected_extension = '.p'
        result_pkg = DummyFullResultPackage({'metric1': 33434, 'acc1': 223.43, 'loss': 4455.6},
                                            {'epoch': 20, 'lr': 0.334})
        training_history = DummyTrainingHistory().wrap_pre_prepared_history({})
        result_file_name_true = f'results_{exp_dir_name}_{current_time}{expected_extension}'

        project_path = os.path.join(THIS_DIR, project_dir_name)
        exp_path = os.path.join(project_path, f'{exp_dir_name}_{current_time}')
        results_path = os.path.join(exp_path, 'results')
        result_file_path_true = os.path.join(results_path, result_file_name_true)

        saver = LocalResultsSaver(local_model_result_folder_path=THIS_DIR, file_format=file_format)
        experiment_results_paths = saver.save_experiment_results_separate_files(
            result_pkg, training_history,
            project_dir_name, exp_dir_name
        )

        self.assertEqual(result_file_path_true, experiment_results_paths[0][1])

        if os.path.exists(project_path):
            shutil.rmtree(project_path)

    def test_additional_results_dump_paths(self):
        project_dir_name = 'projectDir'
        exp_dir_name = 'experimentSubDir'
        file_format = 'pickle'
        expected_extension = '.p'
        current_time = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S')

        training_history = DummyTrainingHistory().wrap_pre_prepared_history({})
        result_file_name_true = f'results_{exp_dir_name}_{current_time}{expected_extension}'
        hyper_param_file_name_true = f'hyperparams_{exp_dir_name}_{current_time}{expected_extension}'
        train_hist_file_name_true = f'train_history_{exp_dir_name}_{current_time}{expected_extension}'

        project_path = os.path.join(THIS_DIR, project_dir_name)
        exp_path = os.path.join(project_path, f'{exp_dir_name}_{current_time}')
        results_path = os.path.join(exp_path, 'results')
        result_file_path_true = os.path.join(results_path, result_file_name_true)
        hyper_param_file_path_true = os.path.join(results_path, hyper_param_file_name_true)
        train_hist_file_path_true = os.path.join(results_path, train_hist_file_name_true)

        additional_results_paths = [
            ['BLAAAAA.txt', os.path.join(results_path, 'BLAAAAA.txt')],
            ['uuuuu.p', os.path.join(results_path, 'uuuuu.p')],
            ['aaaaaa.json', os.path.join(results_path, 'aaaaaa.json')]
        ]

        result_pkg = DummyFullResultPackage(
            {'metric1': 33434, 'acc1': 223.43, 'loss': 4455.6},
            {'epoch': 20, 'lr': 0.334},
            additional_results=additional_results_paths
        )

        saver = LocalResultsSaver(local_model_result_folder_path=THIS_DIR, file_format=file_format)
        experiment_results_paths = saver.save_experiment_results_separate_files(
            result_pkg, training_history,
            project_dir_name, exp_dir_name,
            current_time
        )

        self.assertEqual(
            [
                [result_file_name_true, result_file_path_true],
                [hyper_param_file_name_true, hyper_param_file_path_true],
                [train_hist_file_name_true, train_hist_file_path_true]
            ] + additional_results_paths,
            experiment_results_paths
        )

        if os.path.exists(project_path):
            shutil.rmtree(project_path)
