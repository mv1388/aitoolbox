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
