import unittest
import os
import shutil

from AIToolbox.experiment.result_reporting.report_generator import TrainingHistoryWriter
from AIToolbox.experiment.training_history import TrainingHistory

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class TestTrainingHistoryWriter(unittest.TestCase):
    def test_file_report(self):
        train_history = TrainingHistory().wrap_pre_prepared_history({'loss': [123.4, 1223.4, 13323.4, 13323.4, 99999],
                                                                     'accumulated_loss': [], 'val_loss': [],
                                                                     'NEW_METRIC': [13323.4, 133323.4]})

        result_writer = TrainingHistoryWriter(experiment_results_local_path=THIS_DIR)

        results_file_path_in_cloud_results_dir, results_file_local_path = \
            result_writer.generate_report(train_history, epoch=0,
                                          file_name=f'results.txt', results_folder_name='results')

        self.assertEqual(results_file_path_in_cloud_results_dir, 'results/results.txt')
        self.assertEqual(results_file_local_path, os.path.join(THIS_DIR, 'results/results.txt'))

        with open(os.path.join(THIS_DIR, 'results/results.txt'), 'r') as f:
            f_content = [l.strip() for l in f.readlines()]

        self.assertEqual(f_content,
                         ['============================', 'Epoch: 0', '============================',
                          'loss:\t99999', 'NEW_METRIC:\t133323.4', '', ''])

        train_history.insert_single_result_into_history('NEW_METRIC', 222222.3)
        train_history.insert_single_result_into_history('Completely_new_metric', 442.3)

        results_file_path_in_cloud_results_dir, results_file_local_path = \
            result_writer.generate_report(train_history, epoch=1,
                                          file_name=f'results.txt', results_folder_name='results')

        with open(os.path.join(THIS_DIR, 'results/results.txt'), 'r') as f:
            f_content = [l.strip() for l in f.readlines()]

        self.assertEqual(f_content,
                         ['============================', 'Epoch: 0', '============================',
                          'loss:\t99999', 'NEW_METRIC:\t133323.4', '', '',
                          '============================', 'Epoch: 1', '============================',
                          'loss:\t99999', 'NEW_METRIC:\t222222.3', 'Completely_new_metric:\t442.3', '', ''])

        project_path = os.path.join(THIS_DIR, 'results')
        if os.path.exists(project_path):
            shutil.rmtree(project_path)

    def test_multi_dir_write(self):
        train_history = TrainingHistory().wrap_pre_prepared_history({'loss': [123.4, 1223.4, 13323.4, 13323.4, 99999],
                                                                     'accumulated_loss': [], 'val_loss': [],
                                                                     'NEW_METRIC': [13323.4, 133323.4]})

        result_writer = TrainingHistoryWriter(experiment_results_local_path=THIS_DIR)

        results_file_path_in_cloud_results_dir, results_file_local_path = \
            result_writer.generate_report(train_history, epoch=0,
                                          file_name=f'results.txt', results_folder_name='results')

        self.assertEqual(results_file_path_in_cloud_results_dir, 'results/results.txt')
        self.assertEqual(results_file_local_path, os.path.join(THIS_DIR, 'results/results.txt'))

        results_file_path_in_cloud_results_dir_2, results_file_local_path_2 = \
            result_writer.generate_report(train_history, epoch=0,
                                          file_name=f'results.txt', results_folder_name='results_NEW')

        self.assertEqual(results_file_path_in_cloud_results_dir_2, 'results_NEW/results.txt')
        self.assertEqual(results_file_local_path_2, os.path.join(THIS_DIR, 'results_NEW/results.txt'))

        with open(os.path.join(THIS_DIR, 'results/results.txt'), 'r') as f:
            f_content = [l.strip() for l in f.readlines()]

        self.assertEqual(f_content,
                         ['============================', 'Epoch: 0', '============================',
                          'loss:\t99999', 'NEW_METRIC:\t133323.4', '', ''])

        with open(os.path.join(THIS_DIR, 'results_NEW/results.txt'), 'r') as f:
            f_content = [l.strip() for l in f.readlines()]

        self.assertEqual(f_content,
                         ['============================', 'Epoch: 0', '============================',
                          'loss:\t99999', 'NEW_METRIC:\t133323.4', '', ''])

        project_path = os.path.join(THIS_DIR, 'results')
        if os.path.exists(project_path):
            shutil.rmtree(project_path)

        project_path = os.path.join(THIS_DIR, 'results_NEW')
        if os.path.exists(project_path):
            shutil.rmtree(project_path)
