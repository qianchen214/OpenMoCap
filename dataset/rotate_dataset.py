import torch
from torch.utils.data import Dataset
from PIL import Image
import os

#from pyquaternion import Quaternion
import numpy as np
prev_list = [0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]



class RotateDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        M = []
        J = []
        M1 = []
        JR_rot_mat = []

        
        self.JOINTNUM = 24


        for filename in os.listdir(data_dir):
            if filename.endswith('.npz'):
                npz_path = os.path.join(data_dir, filename)
                data = np.load(npz_path)
                M1.append(data['M1'])
                M.append(data['M'])
                J.append(data['J_t'])
                JR_rot_mat.append(data['J_R'])

        self.M = np.concatenate(M, axis=0)
        self.J = np.concatenate(J, axis=0)
        self.M1 = np.concatenate(M1, axis=0)
        self.JR_rot_mat = np.concatenate(JR_rot_mat, axis=0)



    def __len__(self):
        return self.M.shape[0]

    def __getitem__(self, idx):
        return {'M1': self.M1[idx], 'M': self.M[idx], 'J': self.J[idx], 'JR_rot_mat': self.JR_rot_mat[idx]}



class RotateDataset_perfile(Dataset):
    def __init__(self, data_file):
        self.data_file = data_file
        M = []
        J = []
        M1 = []
        JR_rot_mat = []

        self.JOINTNUM = 24



        if self.data_file.endswith('.npz'):
            data = np.load(data_file)
            M1.append(data['M1'])
            M.append(data['M'])
            J.append(data['J_t'])
            JR_rot_mat.append(data['J_R'])


        self.M = np.concatenate(M, axis=0)
        self.J = np.concatenate(J, axis=0)
        self.M1 = np.concatenate(M1, axis=0)
        self.JR_rot_mat = np.concatenate(JR_rot_mat, axis=0)



    def __len__(self):
        return self.M.shape[0]

    def __getitem__(self, idx):
        return {'M1': self.M1[idx], 'M': self.M[idx], 'J': self.J[idx], 'JR_rot_mat': self.JR_rot_mat[idx]}
