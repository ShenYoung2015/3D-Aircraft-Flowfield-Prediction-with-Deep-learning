# group att
import torch
import torch.nn as nn
import torch.nn.functional as F
import math as math
import time
import numpy as np

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
        mlps.append(3)
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
        elif dim==3:
            self.Conv = nn.Conv3d
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
                        self.mlp.append(self.ACT(0.1))
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
    def __init__(self, map_channels=64, number_group=32, group_size=2, dropout=None, h=4):
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

        l1_q = p_attn@V
        ## !
        l1_q = torch.concat((l1_q-Q, l1_q), dim=-1)
        ##
        l1_q = self.after_conv((l1_q).transpose(1,3)).transpose(1,3)
        l1_q = l1_q.transpose(1, 2).contiguous() \
             .view(batch_size, -1, self.h * self.d_k)
        l1_q = l1_q.permute(0,2,1)
        return l1_q, p_attn

    
def surf_att(global_feat, surf_k_source, x1_surf):
    l1_surf_tmp = global_feat.clone()
    scores = global_feat@surf_k_source / math.sqrt(global_feat.size(-1))
    p_attn = F.softmax(scores, dim = -1)
    p_attn = p_attn/(1e-9 + p_attn.sum(dim=-1, keepdim=True))
    x1_surf = p_attn@x1_surf
    return x1_surf+l1_surf_tmp, p_attn

def local_surf_gather(xyzn_surf_source, surf_fps, fps_npt):
    xyzn_surf = xyzn_surf_source.unsqueeze(2).expand(-1,-1,surf_fps.shape[-2],-1)
    xyzn_group = torch.gather(xyzn_surf.unsqueeze(2).expand(-1,-1,fps_npt,-1,-1), dim=-1,index=surf_fps.expand(-1,xyzn_surf.shape[1],-1,-1,-1))
    xyzn_centroid = xyzn_group[:,:,:,:,:1]
    xyzn_surround = xyzn_group[:,:,:,:,1:] - xyzn_centroid# B,XYZ,FPS,H,wid
    xyzn_centroid = xyzn_centroid.squeeze(-1)
    return xyzn_centroid, xyzn_surround

class PointAR(nn.Module):
    def __init__(self, number_groups=[1024, 256, 64], sampler=False):
        super(PointAR, self).__init__()
        self.number_groups = number_groups
        self.sampler = sampler
        bias = False
        in_channels = 6
        map_channels = 256# 256
        self.map_channels = map_channels
        mlp_FM = [16, map_channels//2, map_channels]
        out_surroud = 64
        # self.dropout1 = nn.Dropout(p=0.2)
        self.dropout1 = None
        
        # Global map
        self.Feat_map_vol = MLPs(6, mlp_FM, NORM=nn.BatchNorm1d, act=nn.LeakyReLU, dropout=self.dropout1)
        # self.source_init = MLPs(6, mlp_FM, NORM=nn.BatchNorm1d, act=nn.LeakyReLU, dropout=self.dropout1)
        # att_phase
        Feat_map_source = nn.ModuleList([])
        k_source_mlp = nn.ModuleList([])
        vol_att = nn.ModuleList([])
        surround_mlp = nn.ModuleList([])
        centroid_mlp = nn.ModuleList([])
        for i, number_group in enumerate(number_groups):
            # 0
            if i==0:
                Feat_map_source.append(MLPs(6, mlp_FM, NORM=nn.BatchNorm1d, act=nn.LeakyReLU, dropout=self.dropout1))
            else:
                Feat_map_source.append(MLPs(map_channels//4, [map_channels//4, map_channels//4], NORM=nn.BatchNorm2d, act=nn.LeakyReLU, dim=2, dropout=self.dropout1))# surf h

            # 1
            k_source_mlp.append(MLPs(map_channels//4, [map_channels//4], NORM=nn.BatchNorm2d, act=nn.LeakyReLU, dim=2, dropout=self.dropout1))

            # 3
            att_phase_modu = att_phase(map_channels=map_channels, number_group=number_group, 
                          dropout=self.dropout1)
            if torch.cuda.is_available():
                att_phase_modu.to('cuda')
            vol_att.append(att_phase_modu)
            # 4
            surround_mlp.append(MLPs(6, [out_surroud//2, out_surroud], dim=3, NORM=nn.BatchNorm3d, act=nn.LeakyReLU, dropout=self.dropout1))
            # 5
            centroid_mlp.append(MLPs(6 + out_surroud + map_channels//4, [map_channels//4, map_channels//4], dim=2, NORM=nn.BatchNorm2d, act=nn.LeakyReLU, dropout=self.dropout1))
        self.Feat_map_source = Feat_map_source
        self.k_source_mlp = k_source_mlp
        self.vol_att = vol_att
        self.surround_mlp = surround_mlp
        self.centroid_mlp = centroid_mlp
        # regression head
        self.fuse = MLPs(map_channels*(len(number_groups)), [map_channels*2, map_channels], NORM=nn.BatchNorm1d, act=nn.LeakyReLU, dropout=self.dropout1)
        self.head = mlp_reg_head(in_channels=map_channels, 
                            mlps=[map_channels, 128, 32], 
                            dropout=self.dropout1, sampler=self.sampler, NORM=nn.BatchNorm1d)
        # att cache
        self.att = False

        
    def forward(self, xyzn_surf_source, xyz_volume):
        # in (B, C, N)
        self.att_cache, self.xyzn_surf_source = [], []
        fps_npt = self.number_groups
        surf_H = 4
        if xyzn_surf_source.shape[-1]>8000 and xyzn_surf_source.shape[0]==1:
            idx = farthest_point_sample(H=1, npoint=2048, xyz_surf=xyzn_surf_source, wid=1)[1]
            idx = idx[0,:,0,0]
            xyzn_surf_source = xyzn_surf_source[:,:,idx]
        batch_size, N_surf_source, N_q = xyzn_surf_source.shape[0], xyzn_surf_source.shape[-1], xyz_volume.shape[-1]
        vol_record = []
        x1_surf = xyzn_surf_source.clone()
        if self._att:
            self.xyzn_surf_source = xyzn_surf_source.clone()
        # x1_surf = self.source_init(x1_surf)
        # x1_surf = x1_surf.reshape(batch_size,-1,surf_H,N_surf_source)
        x1_vol = xyz_volume.clone()

        # att-2 vol-q
        x1_vol = self.Feat_map_vol(x1_vol)
        # vol_record.append(x1_vol)
        for i in range(len(self.vol_att)):
            # att-1 surf sampling 
            surf_fps = farthest_point_sample(H=surf_H, npoint=fps_npt[i], 
                                             xyz_surf=xyzn_surf_source, 
                                             wid=round(16*(N_surf_source/2048)*(256/fps_npt[i])))[1]# [B,FPS,H,wid]
            surf_fps = surf_fps.unsqueeze(1)# [B,1(C),fps,H,wid]
            # print(surf_fps.shape[2], surf_fps.shape[-1])
            # att-1 surf (VK)
            x1_surf = self.Feat_map_source[i](x1_surf).reshape(batch_size,-1,surf_H,N_surf_source)
            surf_k_source = self.k_source_mlp[i](x1_surf)

            # !!!
            xyzn_centroid, xyzn_surround = local_surf_gather(xyzn_surf_source, surf_fps, fps_npt[i])
            l1_surround = self.surround_mlp[i](xyzn_surround).mean(-1)# conv3d & pooling
            l1_surf = torch.concat((xyzn_centroid, l1_surround, x1_surf.mean(-1,keepdim=True).permute(0,1,3,2).expand(-1,-1,xyzn_centroid.shape[-2],-1)), dim=1)
            l1_surf = self.centroid_mlp[i](l1_surf).permute(0,3,2,1).contiguous()# [B,H,fps,C]
            
            # att-1 surf att (Q)
            # index: [B,C,fps,H-surf, wid]
            # l1_surf.unsqueeze(2).expand(-1,-1,fps_npt,-1,-1)
            # surf_fps[i].expand(-1,l1_surf.shape[1],-1,-1,-1)
            # l1_surf[0,0,3,surf_fps[i][0,0,0,3,2]]==l1_surf(new)[0,0,0,3,2]

            # l1_surf = torch.gather(x1_surf.unsqueeze(2).expand(-1,-1,fps_npt[i],-1,-1), dim=-1,index=surf_fps.expand(-1,x1_surf.shape[1],-1,-1,-1))
            # l1_surf = l1_surf.mean(dim=-1).permute(0,3,2,1).contiguous()# [B,H,fps,C]
            # att-1 mul
            if not self._att:
                l1_surf = surf_att(global_feat=l1_surf, surf_k_source=surf_k_source.permute(0,2,1,3), x1_surf=x1_surf.permute(0,2,3,1))[0]
            else:
                l1_surf, p_att = surf_att(global_feat=l1_surf, surf_k_source=surf_k_source.permute(0,2,1,3), x1_surf=x1_surf.permute(0,2,3,1))
                self.att_cache.append(p_att)
                
            # att-2 mul, in [B,N,C]
            l1_surf = l1_surf.transpose(1,2).contiguous().reshape(batch_size,-1,self.map_channels)# l1_surf(new)[0,5,:].reshape(-1) == l1_surf[0,:,5,:].reshape(-1)
            if not self._att:
                x1_vol_tmp = self.vol_att[i](x1_surf=l1_surf, x1_q=x1_vol.permute(0, 2, 1))[0]
            else:
                x1_vol_tmp, p_att = self.vol_att[i](x1_surf=l1_surf, x1_q=x1_vol.permute(0, 2, 1))
                self.att_cache.append(p_att)
            vol_record.append(x1_vol_tmp)# x1_volume_tmp
            x1_vol = x1_vol_tmp

        l1_vol = torch.cat((vol_record), dim=1)
        l1_vol = self.fuse(l1_vol)
        x = self.head(l1_vol).permute(0, 2, 1)
        return x.permute(0, 2, 1)
    
if __name__ == '__main__':
    import torch.nn as nn
    from audtorch.metrics.functional import pearsonr
    torch.random.manual_seed(0)
    xyzn_surf = torch.rand(1, 6, 10000)
    pcd_q = torch.rand(1, 6, 512)
    model = PointAR([64,128,256])
    gt = torch.rand(pcd_q.shape[0], 3, pcd_q.shape[-1])
    pred = model(xyzn_surf, pcd_q)
    crt = nn.MSELoss()

    print(pred.shape)
    print(crt(pred, gt))
    print(crt(pred.permute(0,2,1), gt.permute(0,2,1)))
    