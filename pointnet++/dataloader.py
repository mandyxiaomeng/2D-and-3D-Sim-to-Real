import torch
import torch.utils.data as data
import os
import sys
import h5py
import numpy as np
import glob
import random
from data_utils import *


def load_dir(data_dir, name='train_files.txt'):
    with open(os.path.join(data_dir,name),'r') as f:
        lines = f.readlines()
    return [os.path.join(data_dir, line.rstrip().split('/')[-1]) for line in lines]


def get_info(shapes_dir, isView=False):
    names_dict = {}
    if isView:
        for shape_dir in shapes_dir:
            name = '_'.join(os.path.split(shape_dir)[1].split('.')[0].split('_')[:-1])
            if name in names_dict:
                names_dict[name].append(shape_dir)
            else:
                names_dict[name] = [shape_dir]
    else:
        for shape_dir in shapes_dir:
            name = os.path.split(shape_dir)[1].split('.')[0]
            names_dict[name] = shape_dir

    return names_dict


class Camnet_data(data.Dataset):
    def __init__(self, pc_root, status='train', pc_input_num=1024, aug=True):
        super(Camnet_data, self).__init__()

        self.status = status
        self.pc_list = []
        self.lbl_list = []
        self.pc_input_num = pc_input_num
        self.aug = aug

        categorys = glob.glob(os.path.join(pc_root, '*'))
        categorys = [c.split(os.path.sep)[-1] for c in categorys]
        # sorted(categorys)
        categorys = sorted(categorys)

        if status == 'train':
            npy_list = glob.glob(os.path.join(pc_root, '*', 'train', '*.npy'))
        else:
            npy_list = glob.glob(os.path.join(pc_root, '*', 'test', '*.npy'))
        # names_dict = get_info(npy_list, isView=False)

        for _dir in npy_list:
            _dir = _dir.replace(os.sep, '/')
            self.pc_list.append(_dir)
            self.lbl_list.append(categorys.index(_dir.split('/')[-3]))
        print(f'{status} data num: {len(self.pc_list)}')

    def __getitem__(self, idx):
        lbl = self.lbl_list[idx]
        pc = np.load(self.pc_list[idx])[:self.pc_input_num].astype(np.float32)
        pc = normal_pc(pc)
        if self.aug:
            pc = rotation_point_cloud(pc)
            pc = jitter_point_cloud(pc)
        # print(pc.shape)
        pc = np.expand_dims(pc.transpose(), axis=2)
        return torch.from_numpy(pc).type(torch.FloatTensor), lbl

    def __len__(self):
        return len(self.pc_list)


class Cadnet_data(data.Dataset):
    def __init__(self, pc_root, status='train', pc_input_num=1024, aug=True, data_type='*.npy'):
        super(Cadnet_data, self).__init__()

        self.status = status
        self.pc_list = []
        self.lbl_list = []
        self.pc_input_num = pc_input_num
        self.aug = aug
        self.data_type = data_type

        categorys = glob.glob(os.path.join(pc_root, '*'))
        categorys = [c.split(os.path.sep)[-1] for c in categorys]
        # sorted(categorys)
        categorys = sorted(categorys)

        if status == 'train':
            pts_list = glob.glob(os.path.join(pc_root, '*', 'train', self.data_type))
        elif status == 'test':
            pts_list = glob.glob(os.path.join(pc_root, '*', 'test', self.data_type))
        else:
            pts_list = glob.glob(os.path.join(pc_root, '*', 'validation', self.data_type))
        # names_dict = get_info(pts_list, isView=False)

        for _dir in pts_list:
            _dir = _dir.replace(os.sep, '/')
            self.pc_list.append(_dir)
            self.lbl_list.append(categorys.index(_dir.split('/')[-3]))

        print(f'{status} data num: {len(self.pc_list)}')

    def __getitem__(self, idx):
        lbl = self.lbl_list[idx]
        if self.data_type == '*.pts':
            pc = np.array([[float(value) for value in xyz.split(' ')]
                           for xyz in open(self.pc_list[idx], 'r') if len(xyz.split(' ')) == 3])[:self.pc_input_num, :]
        elif self.data_type == '*.npy':
            pc = np.load(self.pc_list[idx])[:self.pc_input_num].astype(np.float32)
        pc = normal_pc(pc)
        if self.aug:
            pc = rotation_point_cloud(pc)
            pc = jitter_point_cloud(pc)
        pad_pc = np.zeros(shape=(self.pc_input_num-pc.shape[0], 3), dtype=float)
        pc = np.concatenate((pc, pad_pc), axis=0)
        pc = np.expand_dims(pc.transpose(), axis=2)
        return torch.from_numpy(pc).type(torch.FloatTensor), lbl

    def __len__(self):
        return len(self.pc_list)



if __name__ == "__main__":
    data = Cadnet_data(pc_root='/home/data/cadnet', status='validate')
    print (len(data))
    point, label = data[0]
    print (point.shape, label)
    





