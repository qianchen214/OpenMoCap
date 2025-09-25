import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np

class MocapDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        M = []
        J = []
        M1 = []
        self.file_name = []

        
        for filename in os.listdir(data_dir):
            if filename.endswith('.npz'):
                npz_path = os.path.join(data_dir, filename)
                data = np.load(npz_path)
                M1.append(data['M1'])
                M.append(data['M'])
                J.append(data['J_t'])



        self.M = np.concatenate(M, axis=0)
        self.J = np.concatenate(J, axis=0)
        self.M1 = np.concatenate(M1, axis=0)


    def __len__(self):
        return self.M.shape[0]

    def __getitem__(self, idx):
        return {'M1': self.M1[idx], 'M': self.M[idx], 'J': self.J[idx]}
