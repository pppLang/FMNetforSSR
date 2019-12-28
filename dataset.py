import os
import random
import h5py
import numpy as np
import torch
import torch.utils.data as udata


class HyperDataset(udata.Dataset):
    def __init__(self, mode='train'):
        self.mode = mode

        if self.mode == 'train':
            self.h5f = h5py.File('/data0/langzhiqiang/data/train.h5', 'r')
        elif self.mode == 'test':
            self.h5f = h5py.File('/data0/langzhiqiang/data/test_final.h5', 'r')
        elif self.mode == 'train_rw':
            self.h5f = h5py.File('/data0/langzhiqiang/data/train_realworld.h5', 'r')
        elif self.mode == 'test_rw':
            self.h5f = h5py.File('/data0/langzhiqiang/data/test_final_realworld.h5', 'r')

        self.keys = list(self.h5f.keys())
        if 'train' in self.mode:
            random.shuffle(self.keys)
        else:
            self.keys.sort()
            
    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        key = str(self.keys[index])
        data = np.array(self.h5f[key])
        data = torch.Tensor(data)
        return data[0:31,:,:], data[31:34,:,:], data[34:65,:,:]

    def get_data_by_key(self, key):
        assert self.mode == 'train'
        data = np.array(self.h5f[key])
        data = torch.Tensor(data)
        return data[0:31,:,:], data[31:34,:,:], data[34:65,:,:]

    def close(self):
        self.h5f.close()

    def shuffle(self):
        if 'train' in self.mode:
            random.shuffle(self.keys)