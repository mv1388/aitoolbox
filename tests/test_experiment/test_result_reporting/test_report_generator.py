import unittest
import os
import csv
import shutil

from aitoolbox.experiment.result_reporting.report_generator import TrainingHistoryWriter
from aitoolbox.experiment.training_history import TrainingHistory

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class TestTrainingHistoryWriter(unittest.TestCase):
    def test_file_report(self):
        train_history = TrainingHistory().wrap_pre_prepared_history({'loss': [123.4, 1223.4, 13323.4, 13323.4, 99999],
                                                                     'accumulated_loss': [], 'val_loss': [],
                                                                     'NEW_METRIC': [13323.4, 133323.4]})

        result_writer = TrainingHistoryWriter(experiment_results_local_path=THIS_DIR)

        results_file_path_in_cloud_results_dir, results_file_local_path = \
            result_writer.generate_report(train_history, epoch=0,
                                          file_name=f'results.txt', results_folder_name='results_txt')

        self.assertEqual(results_file_path_in_cloud_results_dir, 'results_txt/results.txt')
        self.assertEqual(results_file_local_path, os.path.join(THIS_DIR, 'results_txt/results.txt'))

        with open(os.path.join(THIS_DIR, 'results_txt/results.txt'), 'r') as f:
            f_content = [l.strip() for l in f.readlines()]

        self.assertEqual(f_content,
                         ['============================', 'Epoch: 0', '============================',
                          'loss:\t99999', 'NEW_METRIC:\t133323.4', '', ''])

        train_history.insert_single_result_into_history('NEW_METRIC', 222222.3)
        train_history.insert_single_result_into_history('Completely_new_metric', 442.3)

        results_file_path_in_cloud_results_dir, results_file_local_path = \
            result_writer.generate_report(train_history, epoch=1,
                                          file_name=f'results.txt', results_folder_name='results_txt')

        with open(os.path.join(THIS_DIR, 'results_txt/results.txt'), 'r') as f:
            f_content = [l.strip() for l in f.readlines()]

        self.assertEqual(f_content,
                         ['============================', 'Epoch: 0', '============================',
                          'loss:\t99999', 'NEW_METRIC:\t133323.4', '', '',
                          '============================', 'Epoch: 1', '============================',
                          'loss:\t99999', 'NEW_METRIC:\t222222.3', 'Completely_new_metric:\t442.3', '', ''])

        project_path = os.path.join(THIS_DIR, 'results_txt')
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

    def test_no_results_folder_name_write(self):
        result_writer = TrainingHistoryWriter(experiment_results_local_path=THIS_DIR)

        train_history = TrainingHistory().wrap_pre_prepared_history({'loss': [123.4, 1223.4, 13323.4, 13323.4, 99999],
                                                                     'accumulated_loss': [], 'val_loss': [],
                                                                     'NEW_METRIC': [13323.4, 133323.4]})

        results_file_path_in_cloud_results_dir, results_file_local_path = \
            result_writer.generate_report(train_history, epoch=0, file_name=f'results.txt')

        self.assertEqual(results_file_path_in_cloud_results_dir, 'results.txt')
        self.assertEqual(results_file_local_path, os.path.join(THIS_DIR, 'results.txt'))

        with open(os.path.join(THIS_DIR, 'results.txt'), 'r') as f:
            f_content = [l.strip() for l in f.readlines()]

        self.assertEqual(f_content,
                         ['============================', 'Epoch: 0', '============================',
                          'loss:\t99999', 'NEW_METRIC:\t133323.4', '', ''])

        project_path = os.path.join(THIS_DIR, 'results.txt')
        if os.path.exists(project_path):
            os.remove(project_path)

    def test_file_format_spec_write(self):
        result_writer = TrainingHistoryWriter(experiment_results_local_path=THIS_DIR)
        train_history = TrainingHistory(has_validation=False)

        # Writing initial results
        train_history.insert_single_result_into_history('loss', 123.4)
        train_history.insert_single_result_into_history('NEW_METRIC', 13323.4)
        self.execute_single_write(result_writer, train_history,
                                  expected_results=[['Epoch', 'loss', 'NEW_METRIC'],
                                                    ['0', '123.4', '13323.4']],
                                  epoch=0, file_name='results.tsv', results_folder_name='results_txt')

        # First additional results write
        train_history.insert_single_result_into_history('loss', 199.4)
        train_history.insert_single_result_into_history('NEW_METRIC', 222222.3)
        self.execute_single_write(result_writer, train_history,
                                  expected_results=[['Epoch', 'loss', 'NEW_METRIC'],
                                                    ['0', '123.4', '13323.4'],
                                                    ['1', '199.4', '222222.3']],
                                  epoch=1, file_name='results.tsv', results_folder_name='results_txt')

        # Second additional results write
        train_history.insert_single_result_into_history('loss', 10000.2)
        train_history.insert_single_result_into_history('NEW_METRIC', 144.3)
        train_history.insert_single_result_into_history('COMPLETELY_NEW_METRIC', 555.3)

        self.execute_single_write(result_writer, train_history,
                                  expected_results=[['Epoch', 'loss', 'NEW_METRIC'],
                                                    ['0', '123.4', '13323.4'],
                                                    ['1', '199.4', '222222.3'],
                                                    ['NEW_METRICS_DETECTED'],
                                                    ['Epoch', 'loss', 'NEW_METRIC', 'COMPLETELY_NEW_METRIC'],
                                                    ['2', '10000.2', '144.3', '555.3']],
                                  epoch=2, file_name='results.tsv', results_folder_name='results_txt')

        project_path = os.path.join(THIS_DIR, 'results_txt')
        if os.path.exists(project_path):
            shutil.rmtree(project_path)

    def execute_single_write(self, result_writer, train_history, expected_results, epoch,
                             file_name, results_folder_name):
        results_file_path_in_cloud_results_dir, results_file_local_path = \
            result_writer.generate_report(train_history, epoch=epoch,
                                          file_name=file_name, results_folder_name=results_folder_name,
                                          file_format='tsv')

        self.assertEqual(results_file_path_in_cloud_results_dir, f'{results_folder_name}/{file_name}')
        self.assertEqual(results_file_local_path, os.path.join(THIS_DIR, f'{results_folder_name}/{file_name}'))

        with open(os.path.join(THIS_DIR, f'{results_folder_name}/{file_name}'), 'r') as f:
            tsv_reader = csv.reader(f, delimiter='\t')
            output_lines = [l for l in tsv_reader]

        self.assertEqual(output_lines, expected_results)
