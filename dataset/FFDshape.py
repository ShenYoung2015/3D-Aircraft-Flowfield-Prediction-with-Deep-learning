from cv2 import getStructuringElement
import numpy as np
import os
import torch
from torch.utils.data import DataLoader, Dataset

class FFDshape_ptp(Dataset):
    def __init__(self, dataroot, transforms=None, split='test', npoints=1024, augment=False, dp=False, normalize=False):
        assert(split == 'train' or split == 'test')
        self.npoints = npoints

        self.transforms = transforms
        self.training = True
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
                cur = os.path.join(base, 'pc', '{}.txt'.format(name))
                files_list.append(cur)
        return files_list


    def __getitem__(self, index):
        if index in self.caches:
            return self.caches[index]
        file = self.pcd[index]
        pc = np.loadtxt(file, delimiter=',').astype(np.float32)
        xyz_points = pc[:, :6]
        gts = pc[:, -1]
        
        # resample
        # choice = np.random.choice(len(xyz_points), self.npoints, replace=True)
        # xyz_points = xyz_points[choice, :]
        # gts = gts[choice]

        xyz_points = torch.from_numpy(xyz_points).float()
        gts = torch.from_numpy(gts).float()
        if self.transforms is not None:
            xyz_points, gts = self.transforms(xyz_points, gts)

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

class FFDshape_ptp_tl(Dataset):
    def __init__(self, root, transforms=None, eval_num=0,split='test', npoints=1024, augment=False, dp=False, normalize=False):
        assert(split == 'train' or split == 'test')
        self.npoints = npoints
        self.transforms = transforms
        self.train_files_list = []
        self.test_files_list = []

        self.training = True
        train_name_list = []
        # 遍历目标文件夹中的所有文件和子文件夹
        for tmp, dirs, files in os.walk(os.path.join(root,'pc')):
            for file in files:
                # 如果文件以'.txt'结尾，则将其文件名（不带后缀）添加到列表中
                if file.endswith('.txt'):
                    file_name = os.path.splitext(file)[0]  # 获取文件名（不带后缀）
                    train_name_list.append(file_name)


        eval_root = root
        eval_root = eval_root.split('/')
        eval_root_tmp = eval_root[-1].split('_')
        eval_root_tmp[:2] = ['testing','N200']
        eval_root[-1] = '_'.join(eval_root_tmp)
        eval_root = '/'.join(eval_root)

        test_name_list = os.listdir(os.path.join(eval_root,'pc'))
        test_name_list = []
        # 遍历目标文件夹中的所有文件和子文件夹
        for tmp, dirs, files in os.walk(os.path.join(eval_root,'pc')):
            for file in files:
                # 如果文件以'.txt'结尾，则将其文件名（不带后缀）添加到列表中
                if file.endswith('.txt'):
                    file_name = os.path.splitext(file)[0]  # 获取文件名（不带后缀）
                    test_name_list.append(file_name)

        train_files_list = self.read_list_file(train_name_list, root)
        self.train_files_list = train_files_list
        test_files_list = self.read_list_file(test_name_list, eval_root)
        if eval_num > 0:
            test_files_list = test_files_list[:eval_num]
        self.test_files_list = test_files_list

        self.caches = {}
        print(
            f'Training {len(self.train_files_list)} shapes. Testing {len(self.test_files_list)} shapes '
        )

    def read_list_file(self, name_list, root):
        # base = os.path.dirname(file_path)
        files_list = []
        for shape_name in name_list:
            cur = os.path.join(root, 'pc', '{}.txt'.format(shape_name))
            files_list.append(cur)
        return files_list


    def __getitem__(self, index):
        if index in self.caches:
            return self.caches[index]
        file = self.pcd[index]
        pc = np.loadtxt(file, delimiter=',').astype(np.float32)
        xyz_points = pc[:, :6]
        gts = pc[:, 6]
        
        # resample
        # choice = np.random.choice(len(xyz_points), self.npoints, replace=True)
        # xyz_points = xyz_points[choice, :]
        # gts = gts[choice]

        xyz_points = torch.from_numpy(xyz_points).float()
        gts = torch.from_numpy(gts).float()
        if self.transforms is not None:
            xyz_points, gts = self.transforms(xyz_points, gts)
        else:
            xyz_points = xyz_points.T

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