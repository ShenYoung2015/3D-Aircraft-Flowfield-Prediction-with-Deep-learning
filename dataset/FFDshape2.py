from cv2 import getStructuringElement
import numpy as np
import os
import torch
from torch.utils.data import DataLoader, Dataset

class FFDshape_ptp(Dataset):
    def __init__(self, dataroot='D:\database\WR_fl',
                 transforms=None, split='test', npoints_range=[1024, 50000], 
                 augment=False, dp=False, normalize=False, eval_npoints=None):
        assert(split == 'train' or split == 'test')
        self.npoints_range = npoints_range
        self.transforms = transforms
        self.training = True if split=='train' else False
        self.eval_npoints = eval_npoints
        train_list_path = os.path.join(dataroot, 'trainlist.csv')
        train_files_list = self.read_list_file(train_list_path)
        
        test_list_path = os.path.join(dataroot, 'testlist.csv')
        test_files_list = self.read_list_file(test_list_path)

        self.train_files_list = train_files_list # train_files_list
        self.test_files_list = test_files_list

        self.caches = {}
        print(
            f'Training {len(self.train_files_list)} shapes. Testing {len(self.test_files_list)} shapes '
        )

    def read_list_file(self, file_path):
        base = os.path.dirname(file_path)
        files_list = []
        with open(file_path, 'r') as f:
            for line in f.readlines():
                name = line.strip().split(',')[0]
                cur = os.path.join(base, 'face_pcd', '{}.txt'.format(name))
                files_list.append(cur)
        return files_list


    def __getitem__(self, index):
        if index in self.caches:
            xyz_points, gts = self.caches[index]
        else:
            file = self.pcd[index]
            pc = np.loadtxt(file, delimiter=',').astype(np.float32)
            xyz_points = pc[:, :6]
            gts = pc[:, 7]# 7 cp; 8 9 10 shear-xyz
            self.caches[index] = [xyz_points, gts]

        xyz_points = torch.from_numpy(xyz_points).float()
        gts = torch.from_numpy(gts).float()
        if self.transforms is not None and self.training is True:
            xyz_points, gts = self.transforms(xyz_points, gts)
        elif self.eval_npoints is not None:
            samples = torch.randperm(xyz_points.shape[0])[:self.eval_npoints]
            xyz_points = xyz_points[samples,:].permute(1,0)
            gts = gts[samples]
        else:
            xyz_points = xyz_points.permute(1,0)

        return xyz_points, gts

    def __len__(self):
        if self.training == True:
            set_len = len(self.train_files_list)
        else:
            set_len = len(self.test_files_list)
        return set_len

    def train(self):
        self.training = True
        self.pcd = self.train_files_list
        if self.transforms is not None:
            self.transforms.set_mode('train')

    def eval(self):
        self.training = False
        self.pcd = self.test_files_list
        if self.transforms is not None:
            self.transforms.set_mode('eval')

