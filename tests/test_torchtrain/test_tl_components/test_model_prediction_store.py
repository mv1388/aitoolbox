import unittest

from aitoolbox.torchtrain.tl_components.model_prediction_store import ModelPredictionStore


class TestModelPredictionStore(unittest.TestCase):
    def test_init(self):
        prediction_store = ModelPredictionStore()
        self.assertEqual(prediction_store.prediction_store, {'epoch': 0})

    def test_insert_train_predictions(self):
        prediction_store = ModelPredictionStore()

        prediction_store.insert_train_predictions(([1]*10, [1]*10, {}), 0)
        self.assertEqual(prediction_store.prediction_store, {'epoch': 0, 'train_pred': ([1]*10, [1]*10, {})})

        with self.assertRaises(ValueError):
            prediction_store.insert_train_predictions(([1] * 10, [1] * 10, {}), 0)

    def test_insert_val_predictions(self):
        prediction_store = ModelPredictionStore()

        prediction_store.insert_val_predictions(([1]*10, [1]*10, {}), 0)
        self.assertEqual(prediction_store.prediction_store, {'epoch': 0, 'val_pred': ([1]*10, [1]*10, {})})

        with self.assertRaises(ValueError):
            prediction_store.insert_val_predictions(([1] * 10, [12] * 10, {}), 0)

    def test_insert_test_predictions(self):
        prediction_store = ModelPredictionStore()

        prediction_store.insert_test_predictions(([1]*10, [1]*10, {}), 0)
        self.assertEqual(prediction_store.prediction_store, {'epoch': 0, 'test_pred': ([1]*10, [1]*10, {})})

        with self.assertRaises(ValueError):
            prediction_store.insert_test_predictions(([1] * 10, [12] * 10, {}), 0)

    def test_get_val_predictions(self):
        prediction_store = ModelPredictionStore()
        prediction_store.insert_val_predictions(([1] * 10, [1] * 10, {}), 0)

        self.assertEqual(prediction_store.get_val_predictions(0), ([1] * 10, [1] * 10, {}))

        with self.assertRaises(ValueError):
            prediction_store.get_val_predictions(3)

    def test_auto_purge(self):
        prediction_store = ModelPredictionStore(auto_purge=True)
        prediction_store.insert_val_predictions(([1] * 10, [1] * 10, {}), 0)
        prediction_store.insert_train_predictions(([1] * 10, [1] * 10, {}), 0)

        self.assertEqual(prediction_store.prediction_store,
                         {'epoch': 0, 'train_pred': ([1]*10, [1]*10, {}), 'val_pred': ([1]*10, [1]*10, {})})

        prediction_store.insert_val_predictions(([100] * 10, [1] * 10, {}), 1)
        self.assertEqual(prediction_store.prediction_store, {'epoch': 1, 'val_pred': ([100]*10, [1]*10, {})})

        prediction_store.insert_val_predictions(([100] * 10, [1111] * 10, {}), 2)
        self.assertEqual(prediction_store.prediction_store, {'epoch': 2, 'val_pred': ([100] * 10, [1111] * 10, {})})
