from cv2 import getStructuringElement
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import DataLoader, Dataset

class FFDshape_ptp_ss(Dataset):
    def __init__(self, dataroot='D:\database\shapesffd3',
                 transforms=None, split='test', npoints_target=2048, 
                 augment=False, dp=False, normalize=False, eval_npoints=4096,
                 show_file=False, expand_num=1, tsample=0, gettest=False):
        assert(split == 'train' or split == 'test')

        self.expand_num = expand_num
        self.npoints_volume = npoints_target
        self.transforms = transforms
        self.training = True if split=='train' else False
        self.eval_npoints = eval_npoints
        self.show_file = show_file
        self.gettest = gettest
        train_list_path = os.path.join(dataroot, 'trainlist.csv')
        train_list_surf, train_list_volume = self.read_list_file(train_list_path)

        if tsample>0:
            train_list_surf = train_list_surf[:tsample]
            train_list_volume = train_list_volume[:tsample]
        test_list_path = os.path.join(dataroot, 'testlist.csv')
        if self.gettest:
            test_list_surf, test_list_volume = train_list_surf, train_list_volume
        else:
            test_list_surf, test_list_volume = self.read_list_file(test_list_path)
        
        # 

        if self.expand_num is not None:
            self.train_list_surf = self.expand_num*train_list_surf
            self.train_list_volume = self.expand_num*train_list_volume
        # !
        # self.test_list_surf = self.train_list_surf
        # self.test_list_volume = self.train_list_volume
        ## 
        self.test_list_surf = test_list_surf
        self.test_list_volume = test_list_volume

        

        self.caches = {}
        print(
            f'Training {len(self.train_list_surf)} shapes. Testing {len(self.test_list_surf)} shapes '
        )
        self.crt_file = []

    def read_list_file(self, file_path):
        base = os.path.dirname(file_path)
        files_list_surf = []
        files_list_volume = []
        with open(file_path, 'r') as f:
            for line in f.readlines():
                name = line.strip().split(',')[0]
                cur1 = os.path.join(base, 'pc', '{}.txt'.format(name))
                cur2 = os.path.join(base, 'vol_pc2', '{}.txt'.format(name))
                if os.path.exists(cur1) and os.path.exists(cur2):
                    files_list_surf.append(cur1)
                    files_list_volume.append(cur2)
        # print(files_list_surf)
        return files_list_surf, files_list_volume


    def __getitem__(self, index):
        surf_file = self.files_surf[index]
        volume_file = self.files_volume[index]
        if surf_file in self.caches:
            xyzn_surf, xyzw_volume, gts_surf, gts_vol  = self.caches[surf_file]
        else:
            if self.show_file:
                print(surf_file)
                self.crt_file = surf_file
            else:
                self.crt_file = []
            data_surf = pd.read_csv(surf_file, header=None).to_numpy().astype(np.float32)
            xyzn_surf = data_surf[:, :6] # 7 cp
            gts_surf = data_surf[:, 6]
            
            data_volume = pd.read_csv(volume_file, header=None).to_numpy().astype(np.float32)
            xyzw_volume = data_volume[:, [0,1,2,4,5]] # p
            gts_vol = data_volume[:, 3]# 3: cp, 4: vol, 5:dist



            # if self.training == True and surf_file not in self.caches:
            #     self.caches[surf_file] = [xyzn_surf, xyzw_volume, gts_surf, gts_vol]
        xyzn_surf = torch.from_numpy(xyzn_surf).float()
        xyzw_volume = torch.from_numpy(xyzw_volume).float()
        gts_surf = torch.from_numpy(gts_surf).float()
        gts_vol = torch.from_numpy(gts_vol).float()
        # normalize
        scale = xyzn_surf[:,:3].max(0)[0] - xyzn_surf[:,:3].min(0)[0]
        scale = scale.max()
        xyzw_volume[:,:3] = xyzw_volume[:,:3] / scale
        xyzn_surf[:,:3] = xyzn_surf[:,:3] / scale
        # shift
        shift = xyzn_surf[:, :3].mean(0).unsqueeze(0)
        xyzn_surf[:,:3] = xyzn_surf[:,:3] - shift
        xyzw_volume[:,:3] = xyzw_volume[:,:3] - shift

        # !
        radius = 1.5
        idx = torch.sqrt(torch.sum(xyzw_volume[:,:3]**2, dim=1))<radius
        xyzw_volume = xyzw_volume[idx, :]
        gts_vol = gts_vol[idx]
        ##
        if self.transforms is not None:
            if self.training is True:
                self.transforms.set_mode('train')
                # random scale
                ub, lb = 1.2, 0.8
                rd_scale = torch.rand(1)*(ub-lb) + lb
                xyzw_volume[:,:3] = rd_scale*xyzw_volume[:,:3]
                xyzn_surf[:,:3] = rd_scale*xyzn_surf[:,:3]
                xyzn_surf_source, xyzn_surf, xyz_volume, gts_surf, gts_vol = self.transforms(xyzn_surf, xyzw_volume, gts_surf, gts_vol)
                
            elif self.training is False:
                self.transforms.set_mode('eval')# eval
                xyzn_surf_source, xyzn_surf, xyz_volume, gts_surf, gts_vol = self.transforms(xyzn_surf, xyzw_volume, gts_surf, gts_vol)
        else:
            xyzn_surf = xyzn_surf.permute(1,0)
            xyz_volume = xyzw_volume[:, :6].permute(1,0)

        xyzn_surf[3:,:] = 0
        # out [C, N]
        return xyzn_surf_source, xyzn_surf, xyz_volume, gts_surf, gts_vol

    def __len__(self):
        if self.training == True:
            set_len = len(self.train_list_surf)
        else:
            set_len = len(self.test_list_surf)
        return set_len

    def train(self):
        self.training = True
        self.files_surf = self.train_list_surf
        self.files_volume = self.train_list_volume

        if self.transforms is not None:
            self.transforms.set_mode('train')

    def eval(self):
        self.training = False
        self.files_surf = self.test_list_surf
        self.files_volume = self.test_list_volume
        if self.transforms is not None:
            self.transforms.set_mode('eval')
