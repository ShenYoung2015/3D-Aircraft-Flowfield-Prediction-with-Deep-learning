# 目前的model需要法向朝外
import datetime
import time
from Parameters import *
mix_ratio = 1
surf_npoint_source = 8192
target_npoint = 2048
args = parser.parse_known_args()[0]
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_index
sys.path.insert(1, os.path.dirname(os.path.abspath(__name__)))
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
import torch
from modelPAR_volumeAL import PointAR
from dataset.FFDshape_volume import FFDshape_ptp_ss
from Loss import LabelSmoothingCE, reg_loss
from Transforms3 import PCDPretreatment, get_data_augment
from Trainer_volume_post import Trainer
from utils import IdentityScheduler
import numpy as np
from sklearn import metrics
import trimesh

stl_dir = r'F:\database\shapesffd3\shapes'
CFD_dir = r'F:\database\ss_mach8.04_datagen\shapesffd3_1\eulerrawdata'

pth_file = r'result_train\PAR_h8_model=basic_c_ds=SS2000_aug=basic_lr=0.001_wd=1e-08_bs=16_Adam_cosine\PAR_h8_SS2000_epoch300.pth'
cmd = f'python main_PAR_vol_al.py --name PAR_h8 -bs 16 --lr 3e-4 -ec 5 --dataset none -ne 300 --att_cfg 512 256 128 -en 1 -ts 1400 --optimizer Adam --auto_cast False -cp {pth_file}'
cmd_list = cmd.split()
param_list = []
param_translate = {'-bs':'--batch_size', '-ec':'--eval_cycle', '-ne':'--num_epochs',
                    '-en':'--expand_num', '-ts':'--train_sample_idx','-cp':'--checkpoint'}
# 遍历list
for i in range(len(cmd_list)):
    if cmd_list[i].startswith('-') or cmd_list[i].startswith('--'):
        if cmd_list[i].startswith('-') and not cmd_list[i].startswith('--'):
            key = param_translate[cmd_list[i]]
        else:
            key = cmd_list[i]
        j = 1
        param_list.append(key)
        while (i+j)<len(cmd_list) and not cmd_list[i+j].startswith('-'):
            param_list.append(cmd_list[i+j])
            j = j+1

print(param_list)
timestamp = os.path.getmtime(param_list[-1])
dt = datetime.datetime.fromtimestamp(timestamp)
print(f"pth生成时间：{dt.year}-{dt.month:02d}-{dt.day:02d} {dt.hour:02d}:{dt.minute:02d}:{dt.second:02d}")
args = parser.parse_args(param_list)

def eval_init():
    # 解析参数
    if args.use_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        gpus = list(range(torch.cuda.device_count()))
        torch.cuda.set_device('cuda:{}'.format(gpus[0]))
    else:
        args.device = torch.device('cpu')
    
    att_cfg = args.att_cfg
    print(att_cfg)
    model_cfg = MODEL_CONFIG[args.model_cfg]
    max_input = model_cfg['max_input']
    normal = model_cfg['normal']
    
    if args.optimizer.lower() == 'adamw':
        Optimizer = torch.optim.AdamW
    elif args.optimizer == 'Adam':
        Optimizer = torch.optim.Adam
    elif args.optimizer == 'SGD':
        Optimizer = torch.optim.SGD
    
    if args.scheduler.lower() == 'identity':
        Scheduler = IdentityScheduler
    else:
        args.scheduler = 'cosine'
        Scheduler = torch.optim.lr_scheduler.CosineAnnealingLR

    # 数据变换、加载数据集
    logger.info('Prepare Data')
    '''数据变换、加载数据集'''
    data_augment, random_sample, random_drop = get_data_augment(DATA_AUG_CONFIG[args.data_aug])
   
    transforms = PCDPretreatment(surf_npoint_source=surf_npoint_source, target_npoint=target_npoint, down_sample='random', normal=normal,
                                 data_augmentation=data_augment, random_drop=random_drop, resampling=random_sample, mix_ratio=mix_ratio)

    '''Prepare dataset'''
    if args.dataset_path is None or args.dataset_path == 'default':
        if args.dataset == 'WR_fl':
            # default_dataset_path_list = [r'F:\WR_fl']
            default_dataset_path_list = [r'../MHATT_simp4/dataroot']
        elif args.dataset == 'SS2000':
            default_dataset_path_list = [r'D:\database\shapesffd3']
        # else:
        #     raise ValueError
        
        # for path in default_dataset_path_list:
        #     if os.path.exists(path):
        #         args.dataset_path = path
        #         break
        # else:  # this is for-else block, indent is not missing
        #     raise FileNotFoundError(f'Dataset path not found.')
        logger.info(f'Load default dataset from {args.dataset_path}')
    dataset = None
    if args.dataset == 'WR_fl':
        dataset = FFDshape_ptp(dataroot=args.dataset_path, transforms=transforms,
                                npoints_target=target_npoint, eval_npoints=surf_npoint_source, 
                                expand_num=args.expand_num, tsample=args.train_sample_idx)
    elif args.dataset == 'SS2000':
        dataset = FFDshape_ptp_ss(dataroot=args.dataset_path, transforms=transforms,
                                npoints_target=target_npoint, eval_npoints=surf_npoint_source, 
                                expand_num=args.expand_num, tsample=args.train_sample_idx,
                                gettest=args.gettest)

    # 模型与损失函数
    logger.info('Prepare Models...')
    model = PointAR(att_cfg if att_cfg is not None else [512,256]).to(device=args.device)
    # sampler = PointAR([1], sampler=True).to(device=args.device)
    sampler = None
    if args.optimizer == 'SGD':
        optimizer = Optimizer(model.parameters(), lr=args.lr)
    else:
        optimizer = Optimizer(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = Scheduler(optimizer, T_max=args.num_epochs, eta_min=0)# eta_min=args.lr * 0.001
    if sampler is not None:
        optimizer_s = Optimizer(sampler.parameters(), lr=args.lr, weight_decay=args.wd)
        scheduler_s = Scheduler(optimizer_s, T_max=args.num_epochs, eta_min=0)
    else:
        optimizer_s = None
        scheduler_s = None
    criterion = reg_loss().to(args.device)
    
    


    # 训练器
    logger.info('Trainer launching...')
    trainer = Trainer(
        args=args,
        model=model,
        sampler=None,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        dataset=dataset,
        mode=args.mode,
        optimizer_s=None,
        scheduler_s=None
    )
    # trainer.post()
    # trainer.run()
    # trainer.test()
    return trainer

import logging
logging.basicConfig(level=logging.DEBUG)
import os
import tecplot
import numpy as np
import pandas as pd
import sys
from tecplot.constant import *
tecplot.session.connect(port=7600)

eval = eval_init()
model = eval.model
device = eval.args.device


aoa = 0
aoa_str = '{:.4f}'.format(aoa)


# 批量计算气动力
# 输出xyz气动力系数、计算耗时
from sklearn.metrics import r2_score
df = pd.DataFrame(columns=['c_forces_pred_x', 'c_forces_pred_y', 'c_forces_pred_z', 
                           'c_forces_gt_x', 'c_forces_gt_y', 'c_forces_gt_z', 
                           'duration','mse','pcc', 'R2'])
for shape_id in range(1601,2001):
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    
    shape_name = 'shape_{:04d}'.format(shape_id)
    cgns_file = os.path.join(CFD_dir, f'{shape_name}.cgns')
    stl_file = os.path.join(stl_dir, f'{shape_name}.stl')
    if not os.path.exists(cgns_file):
        cgns_file.replace('.cgns','.dat')
    print(f'read file: {cgns_file}')

    mesh_data = trimesh.load(stl_file)
    points = mesh_data.vertices
    faces = mesh_data.faces
    normals = mesh_data.face_normals
    xyzn_surf = points[faces, :].mean(1)

    # normalize
    scale = xyzn_surf[:,:3].max(0) - xyzn_surf[:,:3].min(0)
    xyzn_surf[:,:3] = xyzn_surf[:,:3] / scale.max()
    points[:,:3] = points[:,:3] / scale.max()

    # shift
    shift = xyzn_surf[:, :3].mean(0)
    xyzn_surf[:,:3] = xyzn_surf[:,:3] - shift
    points[:,:3] = points[:,:3] - shift
    # check normals
    normals_cross = (xyzn_surf - xyzn_surf.mean(0)) * normals
    normals_cross = normals_cross[:,2]
    normals_cross = sum(normals_cross>0)/len(normals_cross)
    # if normals_cross<-0.7:
    #     normals = -normals
    #     print('reverse')
    # elif normals_cross>0.7:
    #     normals = normals
    # else:
    #     raise ValueError
    normals = -normals
    tri_areas = mesh_data.area_faces
    # to tensor
    xyzn_surf = torch.from_numpy(np.concatenate((xyzn_surf, normals), axis=1)).unsqueeze(0).permute(0,2,1).float()

    # import plotly.graph_objects as go
    # fig = go.Figure(data=go.Scatter3d(
    #     x=xyzn_surf[0,0,:],
    #     y=xyzn_surf[0,1,:],
    #     z=xyzn_surf[0,2,:],
    #     mode='markers',
    #     marker=dict(
    #         size=1,
    #         color=xyzn_surf[0,4,:],                # 设置颜色为xyzn_surf
    #         colorscale='Viridis',   # 选择一种颜色映射
    #         colorbar=dict(thickness=20), # 添加颜色条
    #         opacity=0.8
    #     )
    # ))

    # fig.update_layout(margin=dict(r=10, b=10, l=10, t=10))
    # fig.show()

    # surface
    # surface容易出现漏洞状的可视化结果，尤其是在mirror侧，怀疑是mirror缺少拓扑的原因
    # 考虑用单侧进行线性插值（vol to surf），再对单侧surf镜像化，再使用镜像化的surf进行插值。
    # ZONE name: FlowZone1, SurfaceZone1
    torch.manual_seed(0)
    tecplot.new_layout()
    frame = tecplot.active_frame()
    dataset = tecplot.data.load_cgns(cgns_file)

    # mirror operate
    mirror_flow_zone = dataset.zone('FlowZone1').copy()
    mirror_flow_zone.name = 'FlowZone1_mirror'
    mirror_flow_zone.values('CoordinateY')[:] = -mirror_flow_zone.values('CoordinateY')[:]
    # interpolate
    tecplot.data.operate.interpolate_linear(dataset.zone('SurfaceZone1'), 
                                            source_zones=[dataset.zone('FlowZone1'), dataset.zone('FlowZone1_mirror')])
    # tecplot.data.operate.interpolate_inverse_distance(dataset.zone('SurfaceZone1'), 
    #                                         source_zones=[dataset.zone('FlowZone1'), dataset.zone('FlowZone1_mirror')])

    gt = dataset.zone('SurfaceZone1').values('CoefPressure').as_numpy_array()
    x = dataset.zone('SurfaceZone1').values('CoordinateX').as_numpy_array()
    y = dataset.zone('SurfaceZone1').values('CoordinateY').as_numpy_array()
    z = dataset.zone('SurfaceZone1').values('CoordinateZ').as_numpy_array()
    nodemap = np.array(dataset.zone('SurfaceZone1').nodemap[:])
    xyz_volume = torch.concat((torch.tensor(x.reshape(-1, 1)),
                            torch.tensor(y.reshape(-1, 1)),
                            torch.tensor(z.reshape(-1, 1))), 
                                dim=1)
    # # norm & shift
    xyz_volume = xyz_volume/ scale.max()
    xyz_volume = xyz_volume - shift
    xyz_volume = torch.concat((xyz_volume, torch.zeros(xyz_volume.shape[0],3)), dim=1).permute(1,0).unsqueeze(0).float()
    device = 'cuda'
    model.to(device)
    model.eval()
    # Pred

    start_time = time.time() # 记录开始时间
    with torch.no_grad():
        pred = model(xyzn_surf.to(device), xyz_volume.to(device)).squeeze(0)
    end_time = time.time() # 记录结束时间
    print('运行时间：', end_time - start_time, '秒')
    duration_time = end_time - start_time
    
    pred = np.array(pred[0,:].to('cpu'))
    diff = pred-gt
    remove_pts = np.zeros(pred.shape[0], dtype=bool)
    nodemap_plot = nodemap
    xyz_volume_plot = xyz_volume

    # CFD计算的是半构型，结果会是全构型的一半
    # 这里计算的是全构型
    pred_faces = pred[faces].mean(-1).reshape(-1,1)
    gt_faces = gt[faces].mean(-1).reshape(-1,1)
    s_ref = 1
    c_forces_pred = (tri_areas.reshape(-1,1)*(scale.max()**2)*pred_faces*normals).sum(0)/s_ref
    c_forces_gt = (tri_areas.reshape(-1,1)*(scale.max()**2)*gt_faces*normals).sum(0)/s_ref
    # print(c_forces_pred)
    # print(c_forces_gt)
    mse = np.mean(np.square(gt[~remove_pts]  - pred[~remove_pts]))
    corrcoef = np.corrcoef(gt[~remove_pts], pred[~remove_pts])[0][1]
    R2 = r2_score(gt[~remove_pts], pred[~remove_pts])
    data = {'c_forces_pred_x': c_forces_pred[0], 'c_forces_pred_y': c_forces_pred[1], 'c_forces_pred_z': c_forces_pred[2],
            'c_forces_gt_x': c_forces_gt[0], 'c_forces_gt_y': c_forces_gt[1], 'c_forces_gt_z': c_forces_gt[2], 
            'duration':duration_time, 'mse':mse, 'pcc':corrcoef, 'R2':R2}

    df = df.append(data, ignore_index=True)
df.to_csv('data_collection/forces_data.csv', index=False)
