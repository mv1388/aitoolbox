import unittest
import wandb

from aitoolbox.torchtrain.callbacks.wandb import AlertConfig, WandBTracking


class TestAlertConfig(unittest.TestCase):
    def test_attributes_using_defaults(self):
        alert = AlertConfig('my_metric', 10.3)

        self.assertEqual(
            list(alert.__annotations__.keys()),
            ['metric_name', 'threshold_value', 'objective', 'wandb_alert_level']
        )
        self.assertEqual(alert.metric_name, 'my_metric')
        self.assertEqual(alert.threshold_value, 10.3)
        self.assertEqual(alert.objective, 'maximize')
        self.assertIsNone(alert.wandb_alert_level)

    def test_attributes_specify_all_parameters(self):
        alert = AlertConfig('my_metric_new', 100.35, 'minimize', wandb.AlertLevel.ERROR)

        self.assertEqual(
            list(alert.__annotations__.keys()),
            ['metric_name', 'threshold_value', 'objective', 'wandb_alert_level']
        )
        self.assertEqual(alert.metric_name, 'my_metric_new')
        self.assertEqual(alert.threshold_value, 100.35)
        self.assertEqual(alert.objective, 'minimize')
        self.assertEqual(alert.wandb_alert_level, wandb.AlertLevel.ERROR)


# For all tests using WandBTracking set: anonymous='must', mode='offline'

class TestWandBTracking(unittest.TestCase):
    # def test_basic(self):
    #     wandb_cb = WandBTracking(
    #         project_name='bla', experiment_name='lala', local_model_result_folder_path='aaa', hyperparams={},
    #         is_project=False,
    #         anonymous='must', mode='offline'  # needed for testing without the user account
    #     )
    #     wandb_cb.on_train_loop_registration()

    def test_alert_checking(self):
        with self.assertRaises(TypeError):
            wand_cb = WandBTracking(alerts='metric')
            wand_cb.check_alerts()
        with self.assertRaises(TypeError):
            wand_cb = WandBTracking(alerts={'metric': 2})
            wand_cb.check_alerts()
        with self.assertRaises(TypeError):
            wand_cb = WandBTracking(alerts=['metric', 'metric_2', 'metric_3'])
            wand_cb.check_alerts()

        with self.assertRaises(TypeError):
            wand_cb = WandBTracking(alerts=[AlertConfig('metric', 10.), 'metric'])
            wand_cb.check_alerts()

        with self.assertRaises(ValueError):
            wand_cb = WandBTracking(
                alerts=[AlertConfig('metric', 10., objective='minimize'),
                        AlertConfig('metric', 10., objective='blaaa')]
            )
            wand_cb.check_alerts()
