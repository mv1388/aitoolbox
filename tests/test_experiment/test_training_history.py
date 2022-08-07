import unittest

from aitoolbox.experiment.training_history import TrainingHistory


class TestWrapPrePreparedTrainingHistory(unittest.TestCase):
    def test_abstract_callback_has_hook_methods(self):
        history = {'val_loss': [2.2513437271118164, 2.1482439041137695, 2.0187528133392334, 1.7953970432281494,
                                1.5492324829101562, 1.715561032295227, 1.631982684135437, 1.3721977472305298,
                                1.039527416229248, 0.9796673059463501],
                   'val_acc': [0.25999999046325684, 0.36000001430511475, 0.5, 0.5400000214576721, 0.5400000214576721,
                               0.5799999833106995, 0.46000000834465027, 0.699999988079071, 0.7599999904632568,
                               0.7200000286102295],
                   'loss': [2.3088033199310303, 2.2141530513763428, 2.113713264465332, 1.912109375, 1.666761875152588,
                            1.460097312927246, 1.6031768321990967, 1.534214973449707, 1.1710081100463867,
                            0.8969314098358154],
                   'acc': [0.07999999821186066, 0.33000001311302185, 0.3100000023841858, 0.5299999713897705,
                           0.5799999833106995, 0.6200000047683716, 0.4300000071525574, 0.5099999904632568,
                           0.6700000166893005, 0.7599999904632568]}
        train_hist = TrainingHistory().wrap_pre_prepared_history(history)

        self.assertEqual(train_hist.get_train_history(),
                         history)
        self.assertEqual(train_hist.train_history,
                         history)
        
    def test_trigger_exception_history_records(self):
        history = {'val_loss': [2.2513437271118164, 2.1482439041137695, 2.0187528133392334, 1.7953970432281494,
                                1.5492324829101562, 1.715561032295227, 1.631982684135437, 1.3721977472305298,
                                1.039527416229248, 0.9796673059463501],
                   'val_acc': [0.25999999046325684, 0.36000001430511475, 0.5, 0.5400000214576721, 0.5400000214576721,
                               0.5799999833106995, 0.46000000834465027],
                   'loss': [2.3088033199310303, 2.2141530513763428, 2.113713264465332, 1.912109375, 1.666761875152588,
                            1.460097312927246, 1.6031768321990967, 1.534214973449707, 1.1710081100463867,
                            0.8969314098358154],
                   'acc': [0.07999999821186066, 0.33000001311302185, 0.3100000023841858, 0.5299999713897705,
                           0.5799999833106995, 0.6200000047683716, 0.4300000071525574, 0.5099999904632568,
                           0.6700000166893005, 0.7599999904632568]}
        train_hist = TrainingHistory(strict_content_check=True).wrap_pre_prepared_history(history)

        with self.assertRaises(ValueError):
            train_hist.qa_check_history_records()


class TestTrainingHistory(unittest.TestCase):
    def test_init(self):
        th = TrainingHistory(has_validation=True)
        self.assertEqual(th.train_history, {'loss': [], 'accumulated_loss': [], 'val_loss': []})
        self.assertEqual(th.train_history, th.empty_train_history)

        th = TrainingHistory(has_validation=False)
        self.assertEqual(th.train_history, {'loss': [], 'accumulated_loss': []})
        self.assertEqual(th.train_history, th.empty_train_history)

        th = TrainingHistory()
        self.assertEqual(th.train_history, {'loss': [], 'accumulated_loss': [], 'val_loss': []})
        self.assertFalse(th.strict_content_check)
        self.assertEqual(th.train_history, th.empty_train_history)

    def test_insert_single_result_into_history(self):
        th = TrainingHistory()

        th.insert_single_result_into_history('loss', 123.4)
        self.assertEqual(th.train_history, {'loss': [123.4], 'accumulated_loss': [], 'val_loss': []})
        self.assertEqual(th.get_train_history_dict(), {'loss': [123.4], 'accumulated_loss': [], 'val_loss': []})

        th.insert_single_result_into_history('loss', 0.443)
        self.assertEqual(th.train_history, {'loss': [123.4, 0.443], 'accumulated_loss': [], 'val_loss': []})
        self.assertEqual(th.get_train_history_dict(), {'loss': [123.4, 0.443], 'accumulated_loss': [], 'val_loss': []})

        th.insert_single_result_into_history('NEW_METRIC', 0.443)
        self.assertEqual(th.train_history,
                         {'loss': [123.4, 0.443], 'accumulated_loss': [], 'val_loss': [], 'NEW_METRIC': [0.443]})
        self.assertEqual(th.get_train_history_dict(),
                         {'loss': [123.4, 0.443], 'accumulated_loss': [], 'val_loss': [], 'NEW_METRIC': [0.443]})

        th.insert_single_result_into_history('NEW_METRIC', 101.2)
        self.assertEqual(th.train_history,
                         {'loss': [123.4, 0.443], 'accumulated_loss': [], 'val_loss': [], 'NEW_METRIC': [0.443, 101.2]})
        self.assertEqual(th.get_train_history_dict(),
                         {'loss': [123.4, 0.443], 'accumulated_loss': [], 'val_loss': [], 'NEW_METRIC': [0.443, 101.2]})

    def test_test_insert_single_result_into_history_epoch_spec(self):
        th = TrainingHistory()

        th.insert_single_result_into_history('loss', 123.4)
        th.insert_single_result_into_history('loss', 1223.4)
        th.insert_single_result_into_history('loss', 1224443.4)
        self.assertEqual(th.train_history, {'loss': [123.4, 1223.4, 1224443.4], 'accumulated_loss': [], 'val_loss': []})

        th.insert_single_result_into_history('accumulated_loss', 1224443.4)
        th.insert_single_result_into_history('accumulated_loss', 1224443.4)
        self.assertEqual(th.train_history, {'loss': [123.4, 1223.4, 1224443.4],
                                            'accumulated_loss': [1224443.4, 1224443.4], 'val_loss': []})

    def test_get_train_history(self):
        th = TrainingHistory()

        th.insert_single_result_into_history('loss', 123.4)
        th.insert_single_result_into_history('loss', 0.443)
        th.insert_single_result_into_history('NEW_METRIC', 0.443)
        th.insert_single_result_into_history('NEW_METRIC', 101.2)
        self.assertEqual(th.train_history,
                         {'loss': [123.4, 0.443], 'accumulated_loss': [], 'val_loss': [], 'NEW_METRIC': [0.443, 101.2]})

        self.assertEqual(th.get_train_history(),
                         {'loss': [123.4, 0.443], 'accumulated_loss': [], 'val_loss': [], 'NEW_METRIC': [0.443, 101.2]})

    def test_get_train_history_epoch_spec(self):
        th = TrainingHistory()

        th.insert_single_result_into_history('loss', 123.4)
        th.insert_single_result_into_history('loss', 0.443)
        th.insert_single_result_into_history('NEW_METRIC', 0.443)
        th.insert_single_result_into_history('NEW_METRIC', 101.2)
        self.assertEqual(th.train_history,
                         {'loss': [123.4, 0.443], 'accumulated_loss': [], 'val_loss': [], 'NEW_METRIC': [0.443, 101.2]})

        self.assertEqual(th.get_train_history(),
                         {'loss': [123.4, 0.443], 'accumulated_loss': [], 'val_loss': [], 'NEW_METRIC': [0.443, 101.2]})

    def test_get_train_history_dict(self):
        th = TrainingHistory()
        th.insert_single_result_into_history('loss', 123.4)
        th.insert_single_result_into_history('loss', 0.443)
        th.insert_single_result_into_history('NEW_METRIC', 0.443)
        th.insert_single_result_into_history('NEW_METRIC', 101.2)
        self.assertEqual(th.train_history, th.get_train_history_dict())

    def test_str(self):
        th = self._build_dummy_history()
        self.assertEqual(str(th), str(th.train_history))

    def test_len(self):
        th = self._build_dummy_history()
        self.assertEqual(len(th), 4)
        self.assertEqual(len(th), len(th.train_history))

    def test_getitem(self):
        th = self._build_dummy_history()
        self.assertEqual(th['NEW_METRIC'], [13323.4, 133323.4])
        self.assertEqual(th['loss'], [123.4, 1223.4, 13323.4, 13323.4])
        self.assertEqual(th['val_loss'], [])

        with self.assertRaises(KeyError):
            a = th['missing_metric']

    def test_setitem(self):
        th = self._build_dummy_history()

        th['loss'] = 99999
        self.assertEqual(th.train_history,
                         {'loss': [123.4, 1223.4, 13323.4, 13323.4, 99999],
                          'accumulated_loss': [], 'val_loss': [], 'NEW_METRIC': [13323.4, 133323.4]})

        th['NEW_METRIC'] = 11111
        self.assertEqual(th.train_history,
                         {'loss': [123.4, 1223.4, 13323.4, 13323.4, 99999],
                          'accumulated_loss': [], 'val_loss': [], 'NEW_METRIC': [13323.4, 133323.4, 11111]})

        th['accumulated_loss'] = 22222
        self.assertEqual(th.train_history,
                         {'loss': [123.4, 1223.4, 13323.4, 13323.4, 99999],
                          'accumulated_loss': [22222], 'val_loss': [], 'NEW_METRIC': [13323.4, 133323.4, 11111]})

        th['CompletelyNewMetric'] = 55544
        self.assertEqual(th.train_history,
                         {'loss': [123.4, 1223.4, 13323.4, 13323.4, 99999],
                          'accumulated_loss': [22222], 'val_loss': [], 'NEW_METRIC': [13323.4, 133323.4, 11111],
                          'CompletelyNewMetric': [55544]})

    def test_contains(self):
        th = self._build_dummy_history()
        for k in th.train_history:
            self.assertTrue(k in th)
        self.assertFalse('missing_metric' in th)
        self.assertFalse('for sure missing_metric' in th)

    def test_iter(self):
        th = self._build_dummy_history()
        for i, (k_true, k) in enumerate(zip(th.train_history.keys(), th)):
            self.assertEqual(k_true, k)
        self.assertEqual(i+1, len(th.train_history))

        k_true = [k for k in th.train_history.keys()]
        k_th = [k for k in th]
        self.assertEqual(k_true, k_th)

    def test_keys(self):
        th = self._build_dummy_history()
        self.assertEqual(th.keys(), th.train_history.keys())

        k_true = [k for k in th.train_history.keys()]
        k_th = [k for k in th.keys()]
        self.assertEqual(k_true, k_th)

    def test_items(self):
        th = self._build_dummy_history()
        self.assertEqual(th.items(), th.train_history.items())

        k_true = [(k, v) for k, v in th.train_history.items()]
        k_th = [(k, v) for k, v in th.items()]
        self.assertEqual(k_true, k_th)

    def test__add_methods(self):
        th = self._build_dummy_history()

        th_added_1 = th + {'ADDITIONAL_metric': 122.3, 'addi': 344}
        self.assertEqual(
            th_added_1.train_history,
            {'loss': [123.4, 1223.4, 13323.4, 13323.4], 'accumulated_loss': [], 'val_loss': [],
             'NEW_METRIC': [13323.4, 133323.4], 'ADDITIONAL_metric': [122.3], 'addi': [344]}
        )

        th_added_2 = {'ADDITIONAL_metric': 13322.3, 'addi': 1001010} + th
        self.assertEqual(
            th_added_2.train_history,
            {'loss': [123.4, 1223.4, 13323.4, 13323.4], 'accumulated_loss': [], 'val_loss': [],
             'NEW_METRIC': [13323.4, 133323.4], 'ADDITIONAL_metric': [13322.3], 'addi': [1001010]}
        )

        th_added_pre_post = th_added_2 + {'ADDITIONAL_metric': 11.3, 'addi': 1}
        self.assertEqual(
            th_added_pre_post.train_history,
            {'loss': [123.4, 1223.4, 13323.4, 13323.4], 'accumulated_loss': [], 'val_loss': [],
             'NEW_METRIC': [13323.4, 133323.4], 'ADDITIONAL_metric': [13322.3, 11.3], 'addi': [1001010, 1]}
        )

        th += {'ADDITIONAL_metric': 122.3, 'addi': 344}
        self.assertEqual(
            th.train_history,
            {'loss': [123.4, 1223.4, 13323.4, 13323.4], 'accumulated_loss': [], 'val_loss': [],
             'NEW_METRIC': [13323.4, 133323.4], 'ADDITIONAL_metric': [122.3], 'addi': [344]}
        )

        with self.assertRaises(TypeError):
            th + [123.4, 1223.4, 13323.4, 13323.4]

        with self.assertRaises(TypeError):
            th + 123.4

        with self.assertRaises(TypeError):
            [123.4, 1223.4, 13323.4, 13323.4] + th

        with self.assertRaises(TypeError):
            123.4 + th

        with self.assertRaises(TypeError):
            th += [123.4, 1223.4, 13323.4, 13323.4]

        with self.assertRaises(TypeError):
            th += 123.4

    @staticmethod
    def _build_dummy_history():
        th = TrainingHistory()
        th.insert_single_result_into_history('loss', 123.4)
        th.insert_single_result_into_history('loss', 1223.4)
        th.insert_single_result_into_history('loss', 13323.4)
        th.insert_single_result_into_history('NEW_METRIC', 13323.4)
        th.insert_single_result_into_history('NEW_METRIC', 133323.4)
        th.insert_single_result_into_history('loss', 13323.4)
        return th
