from Parameters import *
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_index
sys.path.insert(1, os.path.dirname(os.path.abspath(__name__)))
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
import torch
from modelPAR_volumeAL import PointAR
from dataset.FFDshape_volume import FFDshape_ptp, FFDshape_ptp_ss
from Loss import LabelSmoothingCE, reg_loss, weighted_focal_mse_loss
from Transforms3 import PCDPretreatment, get_data_augment
from Trainer_volume_post import Trainer
from utils import IdentityScheduler

def main():
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
    surf_npoint_source = 4096
    target_npoint = 2048
    mix_ratio = 0.5
    transforms = PCDPretreatment(surf_npoint_source=surf_npoint_source, target_npoint=target_npoint, down_sample='random', normal=normal,
                                 data_augmentation=data_augment, random_drop=random_drop, resampling=random_sample, mix_ratio=mix_ratio)

    '''Prepare dataset'''
    if args.dataset_path is None or args.dataset_path == 'default':
        if args.dataset == 'WR_fl':
            # default_dataset_path_list = [r'F:\WR_fl']
            default_dataset_path_list = [r'../MHATT_simp4/dataroot']
        elif args.dataset == 'SS2000':
            default_dataset_path_list = [r'D:\database\shapesffd3']
        else:
            raise ValueError
        
        for path in default_dataset_path_list:
            if os.path.exists(path):
                args.dataset_path = path
                break
        else:  # this is for-else block, indent is not missing
            raise FileNotFoundError(f'Dataset path not found.')
        logger.info(f'Load default dataset from {args.dataset_path}')
    if args.dataset == 'WR_fl':
        dataset = FFDshape_ptp(dataroot=args.dataset_path, transforms=transforms,
                                npoints_target=target_npoint, eval_npoints=surf_npoint_source, 
                                expand_num=args.expand_num, tsample=args.train_sample_idx)
    elif args.dataset == 'SS2000':
        dataset = FFDshape_ptp_ss(dataroot=args.dataset_path, transforms=transforms,
                                npoints_target=target_npoint, eval_npoints=surf_npoint_source, 
                                expand_num=args.expand_num, tsample=args.train_sample_idx)
    else:
        raise ValueError

    # 模型与损失函数
    logger.info('Prepare Models...')
    model = PointAR(att_cfg if att_cfg is not None else [512,256]).to(device=args.device)
    
    print(att_cfg)
    # sampler = PointAR([1], sampler=True).to(device=args.device)
    sampler = None
    if args.optimizer == 'SGD':
        optimizer = Optimizer(model.parameters(), lr=args.lr)
    else:
        optimizer = Optimizer(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = Scheduler(optimizer, T_max=args.num_epochs, eta_min=args.lr * 0.001)
    
    
    if sampler is not None:
        optimizer_s = Optimizer(sampler.parameters(), lr=args.lr, weight_decay=args.wd)
        scheduler_s = Scheduler(optimizer_s, T_max=args.num_epochs, eta_min=args.lr * 0.001)
    else:
        optimizer_s = None
        scheduler_s = None
    weighted_focal_mse_loss.to(args.device)
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
    trainer.post()


if __name__ == "__main__":
    main()
    print('Done.')
