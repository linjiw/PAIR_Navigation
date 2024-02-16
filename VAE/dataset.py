from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import torch
class NumpyDataset(Dataset):
    def __init__(self, directory):
        self.directory = directory
        self.filenames = [f for f in os.listdir(directory) if f.endswith('.npy')]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        file_path = os.path.join(self.directory, self.filenames[idx])
        data = np.load(file_path)
        return torch.from_numpy(data).float().view(-1)
