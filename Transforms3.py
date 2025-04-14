import torch
import torch.nn as nn
from torchvision.transforms import Compose
import math
import random
from utils import voxel_down_sample
from FeatureExtractorPart.utils import index_points, farthest_point_sample


class PCDPretreatment(nn.Module):
    """
    点云预处理与部分数据增强
    """

    def __init__(self, surf_npoint_source=1024, target_npoint=1024, padding=True, down_sample='random', mode='train', normal=True,
                 data_augmentation=None, random_drop=0, resampling=False, mix_ratio=0.5):
        super().__init__()
        self.surf_npoint_source = surf_npoint_source
        self.target_npoint = target_npoint
        self.num = surf_npoint_source
        self.padding = padding
        self.normal = normal
        self.random_drop = random_drop
        self.resampling = resampling
        self.mode = mode
        self.sampling = down_sample
        self.set_sampling(down_sample)
        self.mix_ratio = mix_ratio
        self.data_aug = data_augmentation if data_augmentation is not None else nn.Identity()
        self.volume_npoint = int(mix_ratio*self.target_npoint)
        self.surf_npoint = self.target_npoint - self.volume_npoint

    def forward(self, xyzn_surf, xyzw_volume, gts_surf, gts_vol):
        """
        :param pcd: <torch.Tensor> (N, 3+) 点云矩阵
        :return: <torch.Tensor> (3+, N)
        """
        # mix_ratio = self.mix_ratio
        xyzn_surf_source = xyzn_surf.clone()
        

        if self.mode == 'train':
            remain_points = ~torch.isinf(gts_surf) & ~(gts_surf>5)
            xyzn_surf = xyzn_surf[remain_points]
            gts_surf = gts_surf[remain_points]
            
            remain_points = ~torch.isinf(gts_vol) & ~(gts_vol>5)
            xyzw_volume = xyzw_volume[remain_points]
            gts_vol = gts_vol[remain_points]
            reverse_idx = torch.randperm(xyzw_volume.shape[0])[:(xyzw_volume.shape[0]//2)]
            xyzw_volume[reverse_idx, 1] = -xyzw_volume[reverse_idx, 1]
            # 随机丢弃一定比率的点
            if self.random_drop > 0:
                drop_ratio = random.uniform(0, self.random_drop)
                remain_points = torch.rand(size=(xyzn_surf_source.shape[0],), device=xyzn_surf_source.device) >= drop_ratio
                xyzn_surf_source = xyzn_surf_source[remain_points]

                remain_points = torch.rand(size=(xyzn_surf.shape[0],), device=xyzn_surf.device) >= drop_ratio
                xyzn_surf = xyzn_surf[remain_points]
                gts_surf = gts_surf[remain_points]

                remain_points = torch.rand(size=(xyzw_volume.shape[0],), device=xyzw_volume.device) >= drop_ratio
                xyzw_volume = xyzw_volume[remain_points]
                gts_vol = gts_vol[remain_points]

            xyz_volume = xyzw_volume[:,:3]
            # 调节点云数量
            # weight = abs(xyzw_volume[:,3])/(xyzw_volume[:,4]**2 + 1e-0)# (xyzw_volume[:,4]**2 + 5e-1)
            weight = abs(xyzw_volume[:,3])**0.3
            # !
            generator = None
            # generator = torch.Generator()
            # generator.manual_seed(1)
            ##
            xyz_volume, gts_vol = self.pcd_padding(xyz_volume, self.volume_npoint, weight=weight, gts=gts_vol, generator=generator)
            # generator.manual_seed(1)
            xyzn_surf, gts_surf = self.pcd_padding(xyzn_surf, self.surf_npoint, weight=None, gts=gts_surf, generator=generator)
            # generator.manual_seed(1)
            xyzn_surf_source, _ = self.pcd_padding(xyzn_surf_source, self.surf_npoint_source, gts=None, generator=generator)
            

            # 点云数量无关的数据增强
            for aug in self.data_aug.transforms:
                if aug.__str__()=='RandomRT()':
                    xyzn_surf_source, xyzn_surf, xyz_volume = aug(xyzn_surf_source, xyzn_surf, xyz_volume)
                else:
                    if self.normal:
                        xyzn_surf_source[:6, :] = aug(xyzn_surf_source[:6, :])
                        xyzn_surf[:6, :] = aug(xyzn_surf[:6, :])
                    else:
                        xyzn_surf_source[:3, :] = aug(xyzn_surf_source[:3, :])
                        xyzn_surf[:6, :] = aug(xyzn_surf[:6, :])
                    xyz_volume[:6, :] = aug(xyz_volume[:6, :])
        elif self.mode == 'eval':
            remain_points = ~torch.isinf(gts_surf) & ~(gts_surf>20)
            xyzn_surf = xyzn_surf[remain_points]
            gts_surf = gts_surf[remain_points]
            
            remain_points = ~torch.isinf(gts_vol) & ~(gts_vol>20)
            xyzw_volume = xyzw_volume[remain_points]
            gts_vol = gts_vol[remain_points]

            generator = torch.Generator()
            # weight = abs(xyzw_volume[:,3])/(xyzw_volume[:,4]**2 + 1e-0)# (xyzw_volume[:,4]**2 + 5e-1)
            weight = abs(xyzw_volume[:,3])**0.3
            xyz_volume = xyzw_volume[:,:3]
            generator = None
            # generator.manual_seed(1)
            xyz_volume, gts_vol = self.pcd_padding(xyz_volume, self.volume_npoint, weight=weight, gts=gts_vol, generator=generator)# , generator=generator
            # generator.manual_seed(1)
            xyzn_surf, gts_surf = self.pcd_padding(xyzn_surf, self.surf_npoint, weight=None, gts=gts_surf, generator=generator)# , generator=generator
            # generator.manual_seed(1)
            xyzn_surf_source, _ = self.pcd_padding(xyzn_surf_source, self.surf_npoint_source, gts=None, generator=generator)# , generator=generator

            if xyz_volume.shape[0]==3:
                xyz_volume = torch.concat((xyz_volume,torch.zeros(xyz_volume.shape).to(xyz_volume.device)), dim=0)
        return xyzn_surf_source, xyzn_surf, xyz_volume, gts_surf, gts_vol
    def pcd_padding(self, pcd, num, weight=None, gts=None, generator=None):
        # 调节点云数量
        if pcd.shape[0] < num:
            padding = torch.zeros(size=(num - pcd.shape[0], pcd.shape[1]), device=pcd.device)
            padding[:, 2] = -10
            pcd = torch.cat((pcd, padding), dim=0)
        elif pcd.shape[0] > num:
            xyz = pcd
            if weight is None:
                if generator is None:
                    downsample_ids = torch.randperm(xyz.shape[0])[:num]
                else:
                    downsample_ids = torch.randperm(xyz.shape[0], generator=generator)[:num]
                pcd = xyz[downsample_ids, :]
                
            else:
                if generator is None:
                    downsample_ids = torch.multinomial(weight, num)
                else:
                    generator.manual_seed(1)
                    downsample_ids = torch.multinomial(weight, num, generator=generator)
                pcd = xyz[downsample_ids, :]

            if gts is not None:
                gts = gts[downsample_ids]

        pcd = pcd.T # [C, N]
        return pcd, gts

    def min_dis(self, pcd):
        """
        基于距离的下采样，保留距离最近的点
        """
        dis = torch.norm(pcd[:, :3], p=2, dim=1)  # 计算点云直线距离
        _, sorted_ids = torch.sort(dis)
        sorted_ids = sorted_ids[:self.num]  # 只保留距离最近的num个点
        pcd = pcd[sorted_ids]
        return pcd, sorted_ids

    def random(self, pcd):
        """
        随机点云下采样
        """
        downsample_ids = torch.randperm(pcd.shape[0])[:self.num]
        pcd = pcd[downsample_ids]
        return pcd, downsample_ids

    def voxel_down_sample(self, pcd):
        return voxel_down_sample(pcd, voxel_size=0.01, num=self.num, padding=self.padding)

    def fps(self, pcd):
        sample_ids = farthest_point_sample(pcd[:, :3].unsqueeze(0), self.num)
        pcd = index_points(pcd.unsqueeze(0), sample_ids)[0]
        return pcd, sample_ids.squeeze(0)

    def set_padding(self, option: bool):
        self.padding = option

    def set_sampling(self, sampling):
        if sampling == 'dis':
            self.down_sample = self.min_dis
        elif sampling == 'voxel':
            self.down_sample = self.voxel_down_sample
        elif sampling == 'random':
            self.down_sample = self.random
        elif sampling == 'fps':
            self.down_sample = self.fps
        elif sampling == 'identical':
            self.down_sample = lambda x: x
        else:
            raise ValueError

    def set_mode(self, mode):
        self.mode = mode
        # if self.resampling:
        #     if mode == 'train':
        #         self.down_sample = self.random
        #     else:
        #         self.set_sampling(self.sampling)


class RandomRT(nn.Module):
    def __init__(self, r_mean=0, r_std=0.5, t_mean=0, t_std=0.1, p=1) -> None:
        super().__init__()
        self.r_mean = r_mean
        self.r_std = r_std
        self.t_mean = t_mean
        self.t_std = t_std
        self.p = p

    def forward(self, xyzn_surf_source, xyzn_surf, xyz_volume):
        """
            pcd: Tensor [3+, n]
        """
        surf_npoint_source, surf_npoint, volume_npoint = xyzn_surf_source.shape[1], xyzn_surf.shape[1], xyz_volume.shape[1]
        xyz_volume_expand = torch.concat((xyz_volume, torch.zeros((3, volume_npoint))))
        pcd = torch.concat((xyzn_surf_source, xyzn_surf, xyz_volume_expand), axis=1)
        # if random.random() > self.p:
        #     return xyzn_surf, xyz_volume

        # 生成X-axis随机角度
        x = (torch.rand(size=(1,)) - 0.5) * 2 * self.r_std
        R_x = torch.tensor([[1, 0, 0],
                            [0, math.cos(x), -math.sin(x)],
                            [0, math.sin(x), math.cos(x)]])
        R_aug = R_x
        R_aug.to(pcd.device)

        if self.t_std > 0:
            T_aug = (torch.rand(size=(3, 1)) - 0.5) * 2 * self.t_std
        else:
            T_aug = torch.zeros(size=(3, 1), device=pcd.device)
        pcd[:3, :] = R_aug @ pcd[:3, :] + T_aug
        if pcd.shape[0] >= 6:
            pcd[3:6, :] = R_aug @ pcd[3:6, :]
            
        xyzn_surf_source, xyzn_surf, xyz_volume = pcd[:, :surf_npoint_source], pcd[:, surf_npoint_source:-volume_npoint], pcd[:, -volume_npoint:]
        return xyzn_surf_source, xyzn_surf, xyz_volume


class RandomPosJitter(nn.Module):
    """点云位置随机抖动"""
    def __init__(self, mean=0, std=0.01, p=1):
        super().__init__()
        self.mean = mean
        self.std = std
        self.p = p

    def forward(self, input):
        """
        :param input:
            pcd: Tensor [3+, n]
        :return:
        """
        if random.random() > self.p:
            return input
        pcd = input
        pos_jitter = (torch.rand(size=(3, pcd.shape[1])) - 0.5) * 2 * self.std
        pcd[:3, :] += pos_jitter
        return pcd


def get_data_augment(data_aug):
    aug_list, data_augment, random_sample, random_drop = [], None, False, 0
    if 'RT' in data_aug and data_aug['RT'] is not None:
        aug_list.append(RandomRT(*data_aug['RT']))
    if 'jitter' in data_aug and data_aug['jitter'] is not None:
        aug_list.append(RandomPosJitter(*data_aug['jitter']))
    if 'random_sample' in data_aug and data_aug['random_sample'] is not None:
        random_sample = data_aug['random_sample']
    if 'random_drop' in data_aug and data_aug['random_drop'] is not None:
        random_drop = data_aug['random_drop']

    if len(aug_list) == 1:
        data_augment = aug_list[0]
    elif len(aug_list) > 1:
        data_augment = Compose(aug_list)
    return data_augment, random_sample, random_drop
