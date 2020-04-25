from torch.utils.data import Dataset


class BasicDataset(Dataset):
    def __init__(self, data):
        """Basic PyTorch dataset where each row (first dimension) represents the example

        Args:
            data (list or torch.Tensor): dataset
        """
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class ListDataset(Dataset):
    def __init__(self, *data_lists):
        """Dataset wrapping lists

        Each sample will be retrieved by indexing tensors along the first dimension. This is the list dataset version
        of PyTorch built-in TensorDataset.

        Args:
            *data_lists (list): data lists that have the same size of the first dimension. Input is represented as
                a list of data lists.

        Examples:
            .. code-block:: python

                list_dataset_1 = [...]
                list_dataset_2 = [...]
                list_dataset_3 = [...]
                ListDataset(list_dataset_1, list_dataset_2, list_dataset_3)
        """
        assert all(len(data_lists[0]) == len(data_l) for data_l in data_lists)
        self.data_lists = data_lists

    def __getitem__(self, index):
        return tuple(data_l[index] for data_l in self.data_lists)

    def __len__(self):
        return len(self.data_lists[0])
