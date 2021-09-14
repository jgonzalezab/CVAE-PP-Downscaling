import torch
from torch.utils.data import Dataset, DataLoader

# Paths
DATA_PATH = './data/'
MODELS_PATH = './models/'

# DataLoader
class MyDataset(Dataset):

    def __init__(self, x, y, transform = None):
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).float()
        self.transform = transform

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        x = self.x[idx, :, :, :]
        y = self.y[idx, :]

        if self.transform:
            x = self.transform(x)

        return x, y
