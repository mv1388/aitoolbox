from torch.utils.data import Dataset


class BasicDataset(Dataset):
    def __init__(self, data):
        """

        Args:
            data:
        """
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """

        Args:
            idx:

        Returns:

        """
        return self.data[idx]
