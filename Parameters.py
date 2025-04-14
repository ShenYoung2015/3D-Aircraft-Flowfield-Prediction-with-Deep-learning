import argparse
import sys
import os


def str_to_bool(s):
    if (s.lower() == 'true'):
        return True
    elif (s.lower() == 'false'):
        return False
    else:
        raise TypeError(f'str {s} can not convert to bool.')


parser = argparse.ArgumentParser(description='Feature Extractor for Alignment')

# 基本参数
parser.add_argument('--name',                   default='PAR_v04',            type=str,
                    help='Name of the model')
parser.add_argument('--mode',                   default='train',                type=str,
                    choices=['train', 'test'],
                    help='train or test')
parser.add_argument('--att_cfg', nargs='+' , type=int,
                help='attention configure')
# 数据参数
parser.add_argument('--dataset',                default='SS2000',           type=str,
                    choices=['ModelNet40','WR_fl','SS2000','none'],
                    help='Dataset name')
parser.add_argument('--dataset_path',           default=None,                   type=str,
                    help='Path to checkpoint file')
parser.add_argument('--expand_num', '-en',      default=1,                     type=int,
                    help='Expanding samples for training')
parser.add_argument('--train_sample_idx', '-ts',      default=400,                     type=int,
                    help='5 means sample shape 1~5')
parser.add_argument('--gettest', default=False,                     type=str_to_bool,
                    help='using training sample as test')

# 训练参数
parser.add_argument('--init' ,     default=False,                     type=str_to_bool,
                    help='clean epochs for checkpoint')
parser.add_argument('--batch_size', '-bs',      default=16,                     type=int,
                    help='Batch size for training')
parser.add_argument('--num_epochs', '-ne',      default=300,                    type=int,
                    help='Number of epochs for training')
parser.add_argument('--lr', '--learning-rate',  default=1e-3,                   type=float,
                    help='initial learning rate for optimizer')
parser.add_argument('--wd', '--weight_decay',   default=1e-8,                   type=float,
                    help='Weight decay for optimizer')
parser.add_argument('--momentum',               default=0.9,                    type=float,
                    help='Momentum value for optimizer')
parser.add_argument('--data_aug',               default='basic',                type=str,
                    help='configuration for data augment')
parser.add_argument('--optimizer',              default='Adam',                type=str,
                    help='optimizer')
parser.add_argument('--scheduler',              default='cosine',               type=str,
                    help='scheduler')
parser.add_argument('--eval_num',              default=400,               type=int,
                    help='number of sample in test set')


# 模型参数
parser.add_argument('--model_cfg',              default='basic_c',              type=str,
                    help='Configuration for building pcd backbone')

# 日志参数
parser.add_argument('--eval_cycle', '-ec',      default=1,                     type=int,
                    help='Evaluate every n epochs')
parser.add_argument('--log_cycle', '-lc',       default=320,                    type=int,
                    help='Log every n steps')
parser.add_argument('--save_cycle', '-sc',      default=10,                      type=int,
                    help='Save every n epochs')
parser.add_argument('--checkpoint', '-cp',      default='',                     type=str,
                    help='Checkpoint file name')

# 设备参数
parser.add_argument('--num_workers', '-nw',     default=2,                     type=int,
                    help='Number of workers used in dataloader')
parser.add_argument('--use_cuda',               default='True',                 type=str_to_bool,
                    help='Using cuda to run')
parser.add_argument('--auto_cast',              default='False',                type=str_to_bool,
                    help='Using torch.cuda.amp.autocast to accelerate computing')
parser.add_argument('--gpu_index',              default='0',                    type=str,
                    help='Index of gpu')

'''数据增强配置参数'''
aug_None = {}
aug_basic = {'RT': [0, 3.14159, 0, 0.2, 1], 'jitter': [0, 0.0001, 1], 'random_drop': 0, 'random_sample': False}
no_aug = {'RT': [0, 0, 0, 0.1, 1], 'jitter': [0, 0, 1], 'random_drop': 0.01, 'random_sample': True}# r_mean=0, r_std=0.5, t_mean=0, t_std=0.1, p=1
DATA_AUG_CONFIG = {'None': aug_None, 'basic': aug_basic, 'no':no_aug}

'''模型配置参数'''
basic_c = {
    'type': 'regression',
    'num_class': 40,
    'max_input': 2048,  # 输入点最大数量
    'npoint': [512, 128, 32, 8],
    'radius_list': [[0.1, 0.2], [0.2, 0.4, 0.4], [0.4, 0.8], [0.8, 1.6]],
    'nsample_list': [[16, 16], [16, 16, 16], [16, 16], [8, 8]],
    'coor_dim': 3,
    'width': 32,
    'expansion': 4,
    'normal': True,
    'head': [512, 256]
}

MODEL_CONFIG = {
    'basic_c': basic_c,
}
