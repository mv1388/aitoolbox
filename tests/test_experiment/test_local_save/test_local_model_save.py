import unittest
import random
import shutil

from tests.utils import *

from aitoolbox.experiment.local_save.local_model_save import *
from aitoolbox.torchtrain.train_loop import TrainLoop

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class DummyLocalSubOptimalModelRemover(LocalSubOptimalModelRemover):
    def __init__(self, metric_name, num_best_kept=2):
        LocalSubOptimalModelRemover.__init__(self, metric_name, num_best_kept)
        self.paths_to_remove = []

    def rm_suboptimal_model(self, rm_model_paths):
        self.paths_to_remove += rm_model_paths


class TestLocalSubOptimalModelRemover(unittest.TestCase):
    def test_init(self):
        remover_loss = DummyLocalSubOptimalModelRemover('loss')
        self.assertTrue(remover_loss.decrease_metric)

        remover_acc = DummyLocalSubOptimalModelRemover('acc')
        self.assertFalse(remover_acc.decrease_metric)

        tl_train_metric_defaults = TrainLoop(NetUnifiedBatchFeed(), [], None, None, None, None).train_history.keys()
        tl_val_metric_defaults = TrainLoop(NetUnifiedBatchFeed(), [], [], None, None, None).train_history.keys()

        self.is_in_default_metric_list(tl_train_metric_defaults)
        self.is_in_default_metric_list(tl_val_metric_defaults)

        remover_non_def_1 = DummyLocalSubOptimalModelRemover('roogue')
        self.assertFalse(remover_non_def_1.is_default_metric)
        remover_non_def_2 = DummyLocalSubOptimalModelRemover('bla')
        self.assertFalse(remover_non_def_2.is_default_metric)

        self.assertEqual(remover_acc.model_save_history, [])

    def is_in_default_metric_list(self, tl_metric_defaults):
        for metric_name in tl_metric_defaults:
            remover_default = DummyLocalSubOptimalModelRemover(metric_name)
            self.assertTrue(remover_default.is_default_metric)
            self.assertTrue(metric_name in remover_default.default_metrics_list)

    def test_decide_if_remove_suboptimal_model_default_metric(self):
        remover_loss = DummyLocalSubOptimalModelRemover('loss', num_best_kept=2)

        history_1 = {'loss': [100.]}
        paths_1 = ['history_path_1.1', 'history_path_1.2']
        remover_loss.decide_if_remove_suboptimal_model(history_1, paths_1)
        self.assertEqual(remover_loss.model_save_history, [(paths_1, history_1['loss'][0])])

        history_2 = {'loss': [100., 50.]}
        paths_2 = ['history_path_2.1', 'history_path_2.2']
        remover_loss.decide_if_remove_suboptimal_model(history_2, paths_2)
        self.assertEqual(remover_loss.model_save_history[0], (paths_1, history_1['loss'][0]))
        self.assertEqual(remover_loss.model_save_history[1], (paths_2, history_2['loss'][1]))

        history_3 = {'loss': [100., 50., 25.]}
        paths_3 = ['history_path_3.1', 'history_path_3.2']
        remover_loss.decide_if_remove_suboptimal_model(history_3, paths_3)
        self.assertEqual(remover_loss.model_save_history[0], (paths_3, history_3['loss'][2]))
        self.assertEqual(remover_loss.model_save_history[1], (paths_2, history_3['loss'][1]))
        self.assertEqual(remover_loss.model_save_history,
                         [(['history_path_3.1', 'history_path_3.2'], 25.0),
                          (['history_path_2.1', 'history_path_2.2'], 50.0)])

        history_4 = {'loss': [100., 50., 25., 30.]}
        paths_4 = ['history_path_4.1', 'history_path_4.2']
        remover_loss.decide_if_remove_suboptimal_model(history_4, paths_4)
        self.assertEqual(remover_loss.model_save_history,
                         [(['history_path_3.1', 'history_path_3.2'], 25.0),
                          (['history_path_4.1', 'history_path_4.2'], 30.0)])

        self.assertEqual(remover_loss.paths_to_remove, paths_1 + paths_2)

    def test_decide_if_remove_suboptimal_model_custom_loss(self):
        remover_loss = DummyLocalSubOptimalModelRemover('my_loss', num_best_kept=2)

        history_1 = {'my_loss': [], 'loss': [1.]}
        paths_1 = ['history_path_1.1', 'history_path_1.2']
        remover_loss.decide_if_remove_suboptimal_model(history_1, paths_1)
        self.assertEqual(remover_loss.model_save_history, [])
        self.assertEqual(remover_loss.non_default_metric_buffer, paths_1)

        history_2 = {'my_loss': [100.], 'loss': [1., 2.]}
        paths_2 = ['history_path_2.1', 'history_path_2.2']
        remover_loss.decide_if_remove_suboptimal_model(history_2, paths_2)
        self.assertEqual(remover_loss.model_save_history, [(['history_path_1.1', 'history_path_1.2'], 100.0)])
        self.assertEqual(remover_loss.non_default_metric_buffer, paths_2)

        history_3 = {'my_loss': [100., 75.], 'loss': [1., 2., 3.]}
        paths_3 = ['history_path_3.1', 'history_path_3.2']
        remover_loss.decide_if_remove_suboptimal_model(history_3, paths_3)
        self.assertEqual(remover_loss.model_save_history,
                         [(['history_path_1.1', 'history_path_1.2'], 100.0),
                          (['history_path_2.1', 'history_path_2.2'], 75.0)])
        self.assertEqual(remover_loss.non_default_metric_buffer, paths_3)

        history_4 = {'my_loss': [100., 75., 50.], 'loss': [1., 2., 3., 4.]}
        paths_4 = ['history_path_4.1', 'history_path_4.2']
        remover_loss.decide_if_remove_suboptimal_model(history_4, paths_4)
        self.assertEqual(remover_loss.model_save_history,
                         [(['history_path_3.1', 'history_path_3.2'], 50.0),
                          (['history_path_2.1', 'history_path_2.2'], 75.0)])
        self.assertEqual(remover_loss.non_default_metric_buffer, paths_4)

        self.assertEqual(remover_loss.paths_to_remove, paths_1)

    def test_decide_if_remove_suboptimal_model_custom_non_loss(self):
        remover_loss = DummyLocalSubOptimalModelRemover('my_acc', num_best_kept=2)

        history_1 = {'my_acc': [], 'loss': [1.]}
        paths_1 = ['history_path_1.1', 'history_path_1.2']
        remover_loss.decide_if_remove_suboptimal_model(history_1, paths_1)
        self.assertEqual(remover_loss.model_save_history, [])
        self.assertEqual(remover_loss.non_default_metric_buffer, paths_1)

        history_2 = {'my_acc': [100.], 'loss': [1., 2.]}
        paths_2 = ['history_path_2.1', 'history_path_2.2']
        remover_loss.decide_if_remove_suboptimal_model(history_2, paths_2)
        self.assertEqual(remover_loss.model_save_history, [(['history_path_1.1', 'history_path_1.2'], 100.0)])
        self.assertEqual(remover_loss.non_default_metric_buffer, paths_2)

        history_3 = {'my_acc': [100., 75.], 'loss': [1., 2., 3.]}
        paths_3 = ['history_path_3.1', 'history_path_3.2']
        remover_loss.decide_if_remove_suboptimal_model(history_3, paths_3)
        self.assertEqual(remover_loss.model_save_history,
                         [(['history_path_1.1', 'history_path_1.2'], 100.0),
                          (['history_path_2.1', 'history_path_2.2'], 75.0)])
        self.assertEqual(remover_loss.non_default_metric_buffer, paths_3)

        history_4 = {'my_acc': [100., 75., 150.], 'loss': [1., 2., 3., 4.]}
        paths_4 = ['history_path_4.1', 'history_path_4.2']
        remover_loss.decide_if_remove_suboptimal_model(history_4, paths_4)
        self.assertEqual(remover_loss.model_save_history,
                         [(['history_path_3.1', 'history_path_3.2'], 150.0),
                          (['history_path_1.1', 'history_path_1.2'], 100.0)])
        self.assertEqual(remover_loss.non_default_metric_buffer, paths_4)

        self.assertEqual(remover_loss.paths_to_remove, paths_2)

    def test_num_best_kept_acc(self):
        for num_kept in range(2, 100):
            history = {'my_acc': [], 'loss': [1.]}
            remover_loss = DummyLocalSubOptimalModelRemover('my_acc', num_best_kept=num_kept)

            for i in range(100):
                paths = [f'history_path_{i}.1', f'history_path_{i}.2']
                remover_loss.decide_if_remove_suboptimal_model(history, paths)

                self.assertEqual(len(remover_loss.model_save_history), min(i, num_kept))

                history['my_acc'].append(random.uniform(0., 100.))

    def test_num_best_kept_loss(self):
        for num_kept in range(2, 100):
            history = {'my_acc': [11010.2], 'loss': []}
            remover_loss = DummyLocalSubOptimalModelRemover('loss', num_best_kept=num_kept)

            for i in range(1, 100):
                paths = [f'history_path_{i}.1', f'history_path_{i}.2']
                history['loss'].append(random.uniform(0., 100.))
                remover_loss.decide_if_remove_suboptimal_model(history, paths)

                self.assertEqual(len(remover_loss.model_save_history), min(i, num_kept))


class TestBaseLocalModelSaver(unittest.TestCase):
    def test_folder_structure_prep(self):
        self.prepare_folder_structure(checkpoint_model=True)
        self.prepare_folder_structure(checkpoint_model=False)

    def prepare_folder_structure(self, checkpoint_model):
        project_dir_name = 'projectDir'
        exp_dir_name = 'experimentSubDir'
        current_time = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S')

        base_final = BaseLocalModelSaver(local_model_result_folder_path=THIS_DIR, checkpoint_model=checkpoint_model)
        path = base_final.create_experiment_local_models_folder(project_dir_name, exp_dir_name, current_time)

        project_path = os.path.join(THIS_DIR, project_dir_name)
        exp_path = os.path.join(project_path, f'{exp_dir_name}_{current_time}')
        model_path = os.path.join(exp_path, 'checkpoint_model' if checkpoint_model else 'model')

        self.assertTrue(os.path.exists(project_path))
        self.assertTrue(os.path.exists(exp_path))
        self.assertTrue(os.path.exists(model_path))
        self.assertTrue(os.path.exists(path))
        self.assertEqual(model_path, path)

        if os.path.exists(project_path):
            shutil.rmtree(project_path)


class TestPyTorchLocalModelSaver(unittest.TestCase):
    def test_init(self):
        saver = PyTorchLocalModelSaver(local_model_result_folder_path=THIS_DIR, checkpoint_model=True)
        self.assertIsInstance(saver, AbstractLocalModelSaver)
        self.assertIsInstance(saver, BaseLocalModelSaver)

    def test_save_model(self):
        for epoch in range(100):
            self.save_model_local(checkpoint_model=True, epoch=epoch)
            self.save_model_local(checkpoint_model=False, epoch=epoch)

    def save_model_local(self, checkpoint_model, epoch):
        model = Net()
        project_dir_name = 'projectPyTorchLocalModelSaver'
        exp_dir_name = 'experimentSubDirPT'
        current_time = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S')
        model_name_true = f'model_{exp_dir_name}_{current_time}_E{epoch}.pth'

        project_path = os.path.join(THIS_DIR, project_dir_name)
        exp_path = os.path.join(project_path, f'{exp_dir_name}_{current_time}')
        model_path = os.path.join(exp_path, 'checkpoint_model' if checkpoint_model else 'model')

        model_file_path = os.path.join(model_path, model_name_true)

        model_checkpoint = {'model_state_dict': model.state_dict(), 'optimizer_state_dict': None,
                            'epoch': 10, 'hyperparams': {}}
        saver = PyTorchLocalModelSaver(local_model_result_folder_path=THIS_DIR, checkpoint_model=checkpoint_model)
        paths = saver.save_model(model_checkpoint, project_dir_name, exp_dir_name, current_time, epoch)
        model_name, model_local_path = paths

        self.assertTrue(os.path.exists(project_path))
        self.assertTrue(os.path.exists(exp_path))
        self.assertTrue(os.path.exists(model_path))

        self.assertTrue(os.path.exists(model_file_path))

        self.assertEqual(model_name_true, model_name)
        self.assertEqual(model_file_path, model_local_path)

        if os.path.exists(project_path):
            shutil.rmtree(project_path)

    def test_fail_check_model_dict_contents(self):
        project_dir_name = 'projectPyTorchLocalModelSaver'
        exp_dir_name = 'experimentSubDirPT'
        current_time = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S')

        model = Net()
        saver = PyTorchLocalModelSaver(local_model_result_folder_path=THIS_DIR)

        model_checkpoint = {'model_state_dict': model.state_dict(), 'optimizer_state_dict': None,
                            'epoch': 10, 'hyperparams': {}}

        for key_excluded in model_checkpoint.keys():
            model_check = {k: v for k, v in model_checkpoint.items() if k != key_excluded}

            with self.assertRaises(ValueError):
                saver.check_model_dict_contents(model_check)

        for key_excluded in model_checkpoint.keys():
            model_check = {k: v for k, v in model_checkpoint.items() if k != key_excluded}
            model_check['additional'] = 'aaaa'

            with self.assertRaises(ValueError):
                saver.check_model_dict_contents(model_check)

        for key_excluded in model_checkpoint.keys():
            model_check = {k: v for k, v in model_checkpoint.items() if k != key_excluded}

            with self.assertRaises(ValueError):
                saver.save_model(model_check, project_dir_name, exp_dir_name, current_time, 1)

        for key_excluded in model_checkpoint.keys():
            model_check = {k: v for k, v in model_checkpoint.items() if k != key_excluded}
            model_check['additional'] = 'aaaa'

            with self.assertRaises(ValueError):
                saver.save_model(model_check, project_dir_name, exp_dir_name, current_time, 1)


class TestKerasLocalModelSaver(unittest.TestCase):
    def test_init(self):
        saver = KerasLocalModelSaver(local_model_result_folder_path=THIS_DIR, checkpoint_model=True)
        self.assertIsInstance(saver, AbstractLocalModelSaver)
        self.assertIsInstance(saver, BaseLocalModelSaver)

    def test_save_model(self):
        for epoch in range(10):
            self.save_model_local(checkpoint_model=True, epoch=epoch)
            self.save_model_local(checkpoint_model=False, epoch=epoch)

    def save_model_local(self, checkpoint_model, epoch):
        model = keras_dummy_model()
        project_dir_name = 'projectKerasLocalModelSaver'
        exp_dir_name = 'experimentSubDirPT'
        current_time = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S')
        model_name_true = f'model_{exp_dir_name}_{current_time}_E{epoch}.h5'

        project_path = os.path.join(THIS_DIR, project_dir_name)
        exp_path = os.path.join(project_path, f'{exp_dir_name}_{current_time}')
        model_path = os.path.join(exp_path, 'checkpoint_model' if checkpoint_model else 'model')

        model_file_path = os.path.join(model_path, model_name_true)

        saver = KerasLocalModelSaver(local_model_result_folder_path=THIS_DIR, checkpoint_model=checkpoint_model)
        paths = saver.save_model(model, project_dir_name, exp_dir_name, current_time, epoch)
        model_name, model_local_path = paths

        self.assertTrue(os.path.exists(project_path))
        self.assertTrue(os.path.exists(exp_path))
        self.assertTrue(os.path.exists(model_path))

        self.assertTrue(os.path.exists(model_file_path))

        self.assertEqual(model_name_true, model_name)
        self.assertEqual(model_file_path, model_local_path)

        if os.path.exists(project_path):
            shutil.rmtree(project_path)
