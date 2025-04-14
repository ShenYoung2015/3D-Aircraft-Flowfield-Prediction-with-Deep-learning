# group att
import torch
import torch.nn as nn
import torch.nn.functional as F
import math as math
import time
import numpy as np
from torch.autograd import Variable

def farthest_point_sample(H, npoint, xyz_surf, wid=64):
    """
    最远点采样
    随机选择一个初始点作为采样点，循环的将与当前采样点距离最远的点当作下一个采样点，直至满足采样点的数量需求
    """
    xyz = xyz_surf[:,:3,:].permute(0,2,1).unsqueeze(1)
    xyz = xyz.expand(-1,H,-1,-1)
    # xyz = xyz.permute(0,2,1,3)
    device = xyz.device
    B, _, N, C = xyz.shape
    npoint = min(npoint, N)
    centroids = torch.zeros(B, H, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, H, N).to(device) * 1e10  # 每个点与最近采样点的最小距离
    # farthest = torch.argmin(xyz.mean(-1), dim=-1).to(device)
    farthest = torch.randint(0, N, (B, H), dtype=torch.long).to(device)
    if wid is not None:
        group_indices = torch.zeros((B, H, npoint, wid), dtype=torch.long).to(device)
    else:
        group_indices = None

    for i in range(npoint):
        centroids[:, :, i] = farthest
        indices = farthest.unsqueeze(-1).unsqueeze(-1)
        indices = indices.expand(-1,-1,-1,C)
        centroid = torch.gather(xyz, dim=2, index=indices).contiguous().view(B, H, 1, -1) 
        dist = torch.nn.functional.pairwise_distance(xyz, centroid)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
        if wid is not None:
            group_indices[:,:,i,:] = torch.topk(dist, wid, dim=-1, largest=False)[1]
    return centroids.permute(0,2,1), group_indices.permute(0,2,1,3), xyz

def avg_max_gather(group_phase, Kph_fps, num_group=16):
    B, H, N, C = group_phase.shape
    Kph_fps = Kph_fps.unsqueeze(-1).expand(-1,-1,-1,-1,C) # [B, num_group, H, G_size, C]
    group_phase = group_phase.unsqueeze(1).expand(-1,num_group,-1,-1,-1)
    resized_phase = torch.gather(group_phase, dim=3, index=Kph_fps).permute(0,1,3,2,4)# [B, num_group, G_size, H, C]
    resized_phase2 = (resized_phase - resized_phase[:,:,:1,:,:])
    # avg_weight = torch.mean(resized_phase, dim=2, keepdim=True)
    max_weight = torch.max(resized_phase, dim=2, keepdim=True)[0].squeeze(2)
    mean_weight = torch.mean(resized_phase, dim=2, keepdim=True).squeeze(2)
    max_weight2 = torch.max(resized_phase2, dim=2, keepdim=True)[0].squeeze(2)
    # avg_weight = avg_weight.squeeze(2)
    max_feat = torch.concat((max_weight, mean_weight, max_weight2), dim=-1)
    
    return max_feat

# class LN_transpose(nn.Module):
#     def __init__(self, module):
#         super(LN_transpose, self).__init__()
#         self.module = module

#     def forward(self, x):
#         if isinstance(self.module, nn.LayerNorm):
#             x = x.transpose(-1, 1)
        
#         x = self.module(x)
        
#         if isinstance(self.module, nn.LayerNorm):
#             x = x.transpose(-1, 1)
#         return x

class mlp_reg_head(nn.Module):
    def __init__(self, in_channels, mlps, dropout=None, NORM=None, sampler=False):
        super(mlp_reg_head, self).__init__()
        self.Conv = nn.Conv1d
        self.ACT = nn.LeakyReLU
        self.bias = True if NORM is None else False
        # self.bias = True
        self.mlp = nn.ModuleList([])
        mlps.append(1)
        # self.mlp = []
        for i, channel in enumerate(mlps):
            if dropout is not None and i > 0:
                self.mlp.append(dropout)
            self.mlp.append(self.Conv(in_channels=in_channels if i==0 else mlps[i-1],
                                      out_channels=channel, kernel_size=1, bias=self.bias))
            if NORM is not None and i < len(mlps)-1:
                self.mlp.append(NORM(channel))
            if i == len(mlps)-1 and not sampler:
                break
            if self.ACT is not None:
                if self.ACT.__name__ == 'ReLU':
                    self.mlp.append(self.ACT(inplace=False))
                elif self.ACT.__name__ == 'LeakyReLU':
                    self.mlp.append(self.ACT(0.01))
                else:
                    self.mlp.append(self.ACT())

    def forward(self, x):
        # x = self.mlp(x)
        for module in self.mlp:
            if isinstance(module, nn.LayerNorm):
                x = x.transpose(-1, 1)
            x = module(x)
            if isinstance(module, nn.LayerNorm):
                x = x.transpose(-1, 1)
        return x

class MLPs(nn.Module):
    def __init__(self, in_channels, mlps, dropout=None, NORM=None, act=nn.ReLU, drop_last_act=False, dim=1):
        super(MLPs, self).__init__()
        if dim==1:
            self.Conv = nn.Conv1d
        elif dim==2:
            self.Conv = nn.Conv2d
        self.ACT = act
        self.bias = True if NORM is None else False
        # self.bias = True
        self.mlp = nn.ModuleList([])
        # self.mlp = []
        for i, channel in enumerate(mlps):
            if dropout is not None and i > 0:
                self.mlp.append(dropout)
            # 每层为conv-bn-relu
            self.mlp.append(self.Conv(in_channels=in_channels if i==0 else mlps[i-1],
                            out_channels=channel, kernel_size=1, bias=self.bias))
            if NORM is not None:
                self.mlp.append(NORM(channel))
            if self.ACT is not None:
                if i<len(mlps)-1 or not drop_last_act:
                    if act.__name__ == 'ReLU':
                        self.mlp.append(self.ACT(inplace=False))
                    elif act.__name__ == 'LeakyReLU':
                        self.mlp.append(self.ACT(0.01))
                    else:
                        self.mlp.append(self.ACT())
        # self.mlp = nn.Sequential(*self.mlp)
        
    def forward(self, x):
        # x = self.mlp(x)
        for module in self.mlp:
            if isinstance(module, nn.LayerNorm):
                x = x.transpose(-1, 1)
            x = module(x)
            if isinstance(module, nn.LayerNorm):
                x = x.transpose(-1, 1)
        return x

class att_phase(nn.Module):
    def __init__(self, map_channels=64, number_group=32, group_size=2, dropout=None, h=1):
        super(att_phase, self).__init__()
        self.number_group1 = number_group
        self.group_size1 = group_size
        self.dropout = dropout
        MLP_norm = nn.BatchNorm1d # nn.LayerNorm, nn.BatchNorm1d
        assert map_channels % h == 0
        self.d_k = map_channels//h
        self.h = h
        
        mlp_query = [map_channels, map_channels]
        mlp_key = [map_channels, map_channels]
        mlp_value = [map_channels, map_channels]
        mlp_after_conv = [map_channels//self.h, map_channels//self.h]
        # Q
        self.query = MLPs(map_channels, mlp_query, NORM=MLP_norm,act=nn.LeakyReLU, dropout=self.dropout)# , drop_last_act=True
        # K
        self.key1 = MLPs(map_channels, mlp_key, NORM=MLP_norm, act=nn.LeakyReLU, dropout=self.dropout)# , drop_last_act=True
        # V
        self.value1 = MLPs(map_channels, mlp_value, NORM=MLP_norm,  act=nn.LeakyReLU, dropout=self.dropout)# , drop_last_act=True

        self.after_conv = MLPs(2*map_channels//self.h, mlp_after_conv, NORM=nn.BatchNorm2d, act=nn.LeakyReLU, dropout=self.dropout, dim=2, drop_last_act=True)# nn.BatchNorm2d , drop_last_act=True
        # self.query1 = MLPs(2*map_channels//self.h, mlp_after_conv, NORM=None, act=nn.GELU, dropout=self.dropout, dim=2)# nn.BatchNorm2d drop_last_act=True
        self.show_attention = False
        self.att_cache = []
        ##
        # self.feat_normals_mlp = MLPs(3, mlp_value, NORM=MLP_norm,  act=nn.LeakyReLU, dropout=self.dropout)
    def forward(self, x1_surf, x1_q):
        batch_size = x1_surf.shape[0]
        # (B, N, C)
        # Q
        Q = (self.query(x1_q.permute(0, 2, 1).contiguous())).permute(0, 2, 1)
        # K V
        K = self.key1(x1_surf.permute(0, 2, 1).contiguous())
        V = self.value1(K)
        
        K, V = K.permute(0, 2, 1), V.permute(0, 2, 1)  # [BCN]
        K = K.contiguous().view(batch_size, -1, self.h, self.d_k).transpose(1, 2)# B, h, N, dk
        V = V.contiguous().view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        Q = Q.contiguous().view(batch_size, -1, self.h, self.d_k).transpose(1, 2)

        scores_s = Q@K.transpose(-2,-1) / math.sqrt(Q.size(-1))
        p_attn = F.softmax(scores_s, dim = -1)
        p_attn = p_attn/(1e-9 + p_attn.sum(dim=3, keepdim=True))
        if self.show_attention:
            self.att_cache = [p_attn]

        l1_q = p_attn@V
        ##
        l1_q = torch.concat((l1_q-Q, l1_q), dim=-1)
        ##
        l1_q = self.after_conv((l1_q).transpose(1,3)).transpose(1,3)
        l1_q = l1_q.transpose(1, 2).contiguous() \
             .view(batch_size, -1, self.h * self.d_k)
        l1_q = l1_q.permute(0,2,1)
        return l1_q

class set_abstract(nn.Module):
    def __init__(self, fps_npts, map_channels_list, wids, ga_channels, target_group, dropout=None, NORM=None, act=nn.LeakyReLU):
        super(set_abstract, self).__init__()
        self.fps_npts = fps_npts
        self.wids = wids
        self.map_channels_list = map_channels_list
        self.target_group = target_group
        sa_conv = nn.ModuleList([])
        for i, fps_npt in enumerate(fps_npts):
            if fps_npt==1:
                sa_conv.append(MLPs(map_channels_list[i]*2, [map_channels_list[i], ga_channels*target_group], NORM=NORM, act=act, dropout=dropout))
            else:
                sa_conv.append(MLPs(map_channels_list[i], [map_channels_list[i], map_channels_list[i+1]], NORM=NORM, act=act, dropout=dropout))
        self.sa_conv = sa_conv
    def forward(self, x1_surf, xyzn_surf):
        xyz_surf = xyzn_surf[:,:3,:]
        feat_surf = x1_surf# BCN
        B,C,N = x1_surf.shape
        for i, fps_npt in enumerate(self.fps_npts):
            if fps_npt>1:
                wid = round(self.wids[i]*N/8192) if i==0 else self.wids[i]
                surf_fps = farthest_point_sample(H=1, npoint=fps_npt, xyz_surf=xyz_surf, wid=wid)[1]
                surf_centroid = torch.gather(feat_surf.unsqueeze(1).expand(-1,fps_npt,-1,-1), dim=3, index=surf_fps.expand(-1,-1,self.map_channels_list[i],-1))
                feat_surf = surf_centroid.max(dim=-1)[0].permute(0,2,1)
                feat_surf = self.sa_conv[i](feat_surf)
                xyz_surf = torch.gather(xyz_surf, dim=2, index=surf_fps[:,:,:,0].permute(0,2,1).expand(-1, 3,-1))
            else:
                feat_surf = torch.concat((feat_surf.max(dim=2, keepdim=True)[0], feat_surf.mean(dim=2, keepdim=True)),
                                         dim=1)
                feat_surf = self.sa_conv[i](feat_surf)
                feat_surf = feat_surf.reshape(B,-1,self.target_group)
        return feat_surf

class global_feat_reshape(nn.Module):
    def __init__(self, ga_channels, group_trans=[1024,256], dropout=None, act=nn.LeakyReLU):
        super(global_feat_reshape, self).__init__()
        self.ga_channels = ga_channels
        self.dropout = dropout
        self.group_trans = group_trans
        self.mlp1 = MLPs(group_trans[0], [group_trans[0]//2, group_trans[1]], 
                          NORM=nn.BatchNorm1d, act=act, dropout=self.dropout)
        self.mlp2 = MLPs(ga_channels, [ga_channels], 
                          NORM=nn.BatchNorm1d, act=act, dropout=self.dropout)
    def forward(self, global_feat):
        B = global_feat.shape[0]
        global_feat = self.mlp1(global_feat.permute(0,2,1))
        global_feat = self.mlp2(global_feat.permute(0,2,1))
        return global_feat
    
def surf_att(global_feat, surf_k_source, x1_surf):
    scores = global_feat@surf_k_source / math.sqrt(global_feat.size(-1))
    p_attn = F.softmax(scores, dim = -1)
    p_attn = p_attn/(1e-9 + p_attn.sum(dim=-1, keepdim=True))
    x1_surf = (p_attn@x1_surf.permute(0,2,1)).permute(0,2,1)
    return x1_surf, p_attn

class PointAR(nn.Module):
    def __init__(self, number_groups=[64, 64], sampler=False):
        super(PointAR, self).__init__()
        self.sampler = sampler
        bias = False
        in_channels = 6
        map_channels = 64# 256
        self.map_channels = map_channels
        mlp_FM = [16, map_channels]
        mlp_FM2 = [map_channels]
        self.ga_channels = 16
        # self.dropout1 = nn.Dropout(p=0.2)
        self.dropout1 = None
        
        # Global map
        self.Feat_map_source = MLPs(6, mlp_FM, NORM=nn.BatchNorm1d, act=nn.LeakyReLU, dropout=self.dropout1)
        self.Feat_map_vol = MLPs(6, mlp_FM, NORM=nn.BatchNorm1d, act=nn.LeakyReLU, dropout=self.dropout1)

        ##
        self.Feat_map_source2 = MLPs(6+map_channels*2, mlp_FM2, NORM=nn.BatchNorm1d, act=nn.LeakyReLU, dropout=self.dropout1)# 6 + featmap + max + min
        self.Feat_map_q2 = MLPs(6+map_channels*2, mlp_FM2, NORM=nn.BatchNorm1d, act=nn.LeakyReLU, dropout=self.dropout1)
        
        # att_phase
        k_source_mlp =  nn.ModuleList([])
        global_extract =  nn.ModuleList([])
        vol_att =  nn.ModuleList([])
        for i, number_group in enumerate(number_groups):
            # 1
            k_source_mlp.append(MLPs(map_channels, [map_channels//2, self.ga_channels], NORM=nn.BatchNorm1d, act=nn.LeakyReLU, dropout=self.dropout1))
            # 2
            if i==0:
                global_extract.append(set_abstract(fps_npts=[512, 128, 1], wids=[32, 16, 512], 
                                                   map_channels_list=[map_channels, 64, 128, 256], 
                                                    dropout=self.dropout1, NORM=nn.BatchNorm1d, act=nn.LeakyReLU,
                                                    ga_channels=self.ga_channels, target_group=number_groups[0]))
            else:
                global_extract.append(global_feat_reshape(ga_channels=self.ga_channels, 
                                                          group_trans=[number_groups[i-1], number_group], 
                                                          dropout=None, act=nn.LeakyReLU))

            # 3
            att_phase_modu = att_phase(map_channels=map_channels, number_group=number_group, 
                          dropout=self.dropout1)
            if torch.cuda.is_available():
                att_phase_modu.to('cuda')
            vol_att.append(att_phase_modu)

        self.k_source_mlp = k_source_mlp
        self.vol_att = vol_att
        self.global_extract = global_extract
        # regression head
        self.fuse = MLPs(map_channels*(1+len(number_groups)), [map_channels*2, map_channels], NORM=nn.BatchNorm1d, act=nn.LeakyReLU, dropout=self.dropout1)
        self.head = mlp_reg_head(in_channels=map_channels, 
                            mlps=[map_channels, 16], 
                            dropout=self.dropout1, sampler=self.sampler, NORM=nn.BatchNorm1d)


        
    def forward(self, xyzn_surf_source, xyz_volume):
        # in (B, C, N)
        vol_record = []

        xyzn_surf_tmp = xyzn_surf_source.clone()
        xyz_volume_tmp = xyz_volume.clone()

        N_surf_source, N_q = xyzn_surf_source.shape[-1], xyz_volume.shape[-1]        
        x1_surf = self.Feat_map_source(xyzn_surf_tmp)
        x1_vol = self.Feat_map_vol(xyz_volume_tmp)
        # att sampling
        surf_k_source = self.k_source_mlp[0](x1_surf)
        global_feat = self.global_extract[0](x1_surf=x1_surf, xyzn_surf=xyzn_surf_tmp)
        x1_surf = surf_att(global_feat=global_feat.permute(0,2,1), surf_k_source=surf_k_source, x1_surf=x1_surf)
        x1_surf = x1_surf[0]

        vol_record.append(x1_vol)
        
        for i in range(len(self.vol_att)):
            if i>0:
                global_feat = self.global_extract[i](global_feat)
                surf_k_source = self.k_source_mlp[i](x1_surf)
                x1_surf = surf_att(global_feat=global_feat.permute(0,2,1), surf_k_source=surf_k_source, x1_surf=x1_surf)
                x1_surf = x1_surf[0]
            x1_vol_tmp = self.vol_att[i](x1_surf=x1_surf.permute(0, 2, 1), x1_q=x1_vol.permute(0, 2, 1)) + x1_vol
            vol_record.append(x1_vol_tmp-x1_vol)# x1_volume_tmp
            x1_vol = x1_vol_tmp

        l1_vol = torch.cat((vol_record), dim=1)
        l1_vol = self.fuse(l1_vol) + vol_record[0]
        x = self.head(l1_vol).permute(0, 2, 1)
        return x.permute(0, 2, 1)
    
if __name__ == '__main__':
    torch.random.manual_seed(0)
    xyzn_surf = torch.rand(2, 6, 10000)
    pcd_q = torch.rand(2, 6, 512)
    model = PointAR([256, 64])
    pred = model(xyzn_surf, pcd_q)
    print(pred.shape)