import os

from data import common
from data import srdata

import numpy as np
import scipy.misc as misc

import torch
import torch.utils.data as data

class MSI_Liver(srdata.SRData):
    def __init__(self, args, train=True):
        #super(MSI_Liver, self).__init__(args, train)
        super().__init__(args, train)
        self.repeat = args.test_every // (args.n_train // args.batch_size)

    def _scan(self):
        list_hr = []
        list_lr = [[] for _ in self.scale]
        if self.train:
            idx_begin = 0
            idx_end = self.args.n_train
        else:
            idx_begin = self.args.n_train
            idx_end = self.args.offset_val + self.args.n_val

        filenames = os.listdir(self.dir_hr)
        hr_folds = os.listdir(self.dir_lr)
        for f in filenames:
            list_hr.append(os.path.join(self.dir_hr, f)) # + self.ext))
            for i, fold in enumerate(hr_folds):
                list_lr[i].append(os.path.join(self.dir_lr, fold, f))

        return list_hr, list_lr

    def _set_filesystem(self, dir_data):
        self.apath = dir_data
        if self.train:
            self.dir_hr = os.path.join(self.apath, 'train', 'HR')
            self.dir_lr = os.path.join(self.apath, 'train', 'LR')
            self.ext = '.npy'
        else:
            self.dir_hr = os.path.join(self.apath, 'test', 'HR')
            self.dir_lr = os.path.join(self.apath, 'test', 'LR')
            self.ext = '.npy'

    def _name_hrbin(self):
        raise NotImplementedError

    def _name_lrbin(self, scale):
        raise NotImplementedError

    def __len__(self):
        if self.train:
            return len(self.images_hr) * self.repeat
        else:
            return len(self.images_hr)

    def _get_index(self, idx):
        if self.train:
            return idx % len(self.images_hr)
        else:
            return idx

