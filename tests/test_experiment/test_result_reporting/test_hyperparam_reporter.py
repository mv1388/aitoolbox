import unittest
import os
import shutil

from aitoolbox.experiment.result_reporting.hyperparam_reporter import HyperParamSourceReporter


THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class TestHyperParameterReporter(unittest.TestCase):
    def test_init(self):
        param_saver = HyperParamSourceReporter('my_project', 'fancy_experiment', '2019_01_01_00_11', THIS_DIR)

        experiment_path = os.path.join(THIS_DIR, 'my_project', 'fancy_experiment_2019_01_01_00_11')

        self.assertEqual(param_saver.experiment_dir_path, experiment_path)
        # self.assertFalse(os.path.exists(os.path.join(experiment_path, 'results')))
        # self.assertEqual(len(os.listdir(experiment_path)), 0)
        self.assertEqual(param_saver.file_name, 'hyperparams_list.txt')

        self.assertEqual(param_saver.local_hyperparams_file_path, os.path.join(experiment_path, 'hyperparams_list.txt'))

        project_path = os.path.join(THIS_DIR, 'my_project')
        if os.path.exists(project_path):
            shutil.rmtree(project_path)

    def test_save_args_to_text_file(self):
        args = {'a': 103, 'pram2': 444, 'LR': 0.0002, 'path': 'dasdas/33332edas'}

        param_saver = HyperParamSourceReporter('my_project', 'fancy_experiment', '2019_01_01_00_11', THIS_DIR)
        local_args_file_path = param_saver.save_hyperparams_to_text_file(args)
        self.check_saved_file_contents(local_args_file_path, args)

        project_path = os.path.join(THIS_DIR, 'my_project')
        if os.path.exists(project_path):
            shutil.rmtree(project_path)

    def check_saved_file_contents(self, local_args_file_path, args):
        with open(local_args_file_path, 'r') as f:
            f_lines = f.readlines()

        self.assertEqual(len(f_lines), len(args))
        self.assertEqual(sorted([el.split(':\t')[0] for el in f_lines]),
                         sorted(args.keys()))

        for line in f_lines:
            k, v = line.strip().split(':\t')
            self.assertEqual(str(args[k]), v)
