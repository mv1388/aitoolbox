import unittest
import numpy as np

from aitoolbox.torchtrain.tl_components.pred_collate_fns import *


class TestBatchPredCollateFns(unittest.TestCase):
    def test_append_predictions(self):
        preds = []

        preds_ep_1 = np.random.rand(100, 1)
        preds = append_predictions(preds_ep_1, preds)
        self.assertEqual([preds_ep_1], preds)

        preds_ep_2 = np.random.rand(45, 1)
        preds = append_predictions(preds_ep_2, preds)
        self.assertEqual([preds_ep_1, preds_ep_2], preds)

    def test_append_concat_predictions(self):
        preds = []
        preds_ep_1 = np.random.rand(100, 1)
        preds = append_predictions(preds_ep_1, preds)
        self.assertEqual([preds_ep_1], preds)
        preds_ep_2 = np.random.rand(45, 1)
        preds = append_predictions(preds_ep_2, preds)
        self.assertEqual([preds_ep_1, preds_ep_2], preds)

        preds_list = []
        preds_list_ep_1 = np.random.rand(100).tolist()
        preds_list = append_concat_predictions(preds_list_ep_1, preds_list)
        self.assertEqual(preds_list_ep_1, preds_list)

        preds_list_ep_2 = np.random.rand(45, 1).tolist()
        preds_list = append_concat_predictions(preds_list_ep_2, preds_list)
        self.assertEqual(preds_list_ep_1 + preds_list_ep_2, preds_list)


# class TestAllPredTransformFns(unittest.TestCase):
#     def test_torch_cat_transf(self):
#         pass
#
#     def test_not_list_torch_cat_transf(self):
#         pass
