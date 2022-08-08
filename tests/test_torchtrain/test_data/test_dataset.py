import unittest
from random import randint

from torch.utils.data.dataloader import DataLoader
from aitoolbox.torchtrain.data.dataset import BasicDataset, ListDataset


class TestBasicDataset(unittest.TestCase):
    def test_len(self):
        ds = BasicDataset(list(range(100)))
        self.assertEqual(len(ds), 100)

    def test_get_item(self):
        ds = BasicDataset(list(range(100)))
        for i in range(100):
            self.assertEqual(ds[i], i)


class TestListDataset(unittest.TestCase):
    def test_list_dataset_return_format(self):
        self.list_dataset_return_format_check(batch_size=5)
        self.list_dataset_return_format_check(batch_size=10)
        self.list_dataset_return_format_check(batch_size=20)
        self.list_dataset_return_format_check(batch_size=50)
        self.list_dataset_return_format_check(batch_size=100)

    def test_list_dataset_return_format_large_dataset(self):
        self.list_dataset_return_format_check(batch_size=5, dataset_size=10000)
        self.list_dataset_return_format_check(batch_size=10, dataset_size=10000)
        self.list_dataset_return_format_check(batch_size=20, dataset_size=10000)
        self.list_dataset_return_format_check(batch_size=50, dataset_size=10000)
        self.list_dataset_return_format_check(batch_size=100, dataset_size=10000)
        self.list_dataset_return_format_check(batch_size=200, dataset_size=10000)
        self.list_dataset_return_format_check(batch_size=500, dataset_size=10000)
        self.list_dataset_return_format_check(batch_size=1000, dataset_size=10000)

    def list_dataset_return_format_check(self, batch_size, dataset_size=100):
        feats_data_1 = [[i + j for j in range(10, 10 + randint(0, 20))] for i in range(dataset_size)]
        feats_data_2 = [[i + j for j in range(100, 100 + randint(0, 100))] for i in range(dataset_size)]
        feats_data_3 = [[i + j for j in range(1000, 1000 + randint(0, 200))] for i in range(dataset_size)]

        list_dataset = ListDataset(feats_data_1, feats_data_2, feats_data_3)
        dataloader = DataLoader(list_dataset, batch_size=batch_size,
                                collate_fn=lambda batch_data: list(zip(*batch_data)))

        for i, data in enumerate(dataloader):
            batch_feat_1, batch_feat_2, batch_feat_3 = data
            self.assertEqual(feats_data_1[i * batch_size:(i + 1) * batch_size], list(batch_feat_1))
            self.assertEqual(feats_data_2[i * batch_size:(i + 1) * batch_size], list(batch_feat_2))
            self.assertEqual(feats_data_3[i * batch_size:(i + 1) * batch_size], list(batch_feat_3))
