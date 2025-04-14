import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast as autocast
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm
from utils import MetricLogger, fakecast
from utils import show_pcd
from audtorch.metrics.functional import pearsonr
import numpy as np

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.manual_seed(1)


class Trainer:
    """
    训练器，输入待训练的模型、参数，封装训练过程
    """
    def __init__(self, args, model, optimizer, scheduler, criterion, dataset, mode, optimizer_s, scheduler_s, sampler=None):
        self.args = args
        self.model = model
        self.sampler = sampler
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.dataset = dataset
        self.mode = mode
        self.dataloader = None
        self.epoch = 1
        self.step = 1
        self.optimizer_s = optimizer_s
        self.scheduler_s = scheduler_s

        self.epoch_metric_logger = MetricLogger()

        # 恢复检查点
        if self.args.checkpoint != '':
            checkpoint = torch.load(self.args.checkpoint, map_location=self.args.device)
            # Load model
            if 'model' in checkpoint:
                model_state_dict = checkpoint['model']
                if model_state_dict.keys() != self.model.state_dict().keys():
                    logger.info("Load model Failed, keys not match..")
                else:
                    self.model.load_state_dict(model_state_dict)
                    logger.info("Load model state")
                    if 'optimizer' in checkpoint:
                        self.optimizer.load_state_dict(checkpoint['optimizer'])
                        logger.info("Load optimizer state")
                    if 'scheduler' in checkpoint:
                        self.scheduler.load_state_dict(checkpoint['scheduler'])
                        logger.info("Load scheduler state")

            if 'epoch' in checkpoint:
                self.epoch = checkpoint['epoch'] + 1
                logger.info(f"Load epoch, current = {self.epoch}")

            if 'step' in checkpoint:
                self.step = checkpoint['step'] + 1
                logger.info(f"Load step, current = {self.step}")
            logger.info(f'Load checkpoint complete: \'{self.args.checkpoint}\'')
        else:
            logger.info(f'{mode} with a initial model')

        if self.args.auto_cast:
            self.cast = autocast
        else:
            self.cast = fakecast

        # 创建训练、测试结果保存目录
        self.log = f'{self.args.name}_model={self.args.model_cfg}_ds={self.args.dataset}_aug={self.args.data_aug}_' \
                   f'lr={self.args.lr}_wd={self.args.wd}_bs={self.args.batch_size}_' \
                   f'{self.args.optimizer}_{self.args.scheduler}'
        if self.mode == 'train':
            self.save_root = os.path.join('./result_train', self.log)
        elif self.mode == 'test':
            self.save_root = os.path.join('./result_test', self.log)
        else:
            raise ValueError
        os.makedirs(self.save_root, exist_ok=True)
        logger.info(f'save root = \'{self.save_root}\'')
        logger.info(f'run in {self.args.device}')

    def run(self):
        if self.mode == 'train':
            self.train()
        elif self.mode == 'test':
            self.test()

    def train(self):
        # tensorboard可视化训练过程，记录训练时的相关数据，使用指令:tensorboard --logdir=runs
        self.writer = SummaryWriter(os.path.join('./runs', self.log))

        self.dataloader = DataLoader(dataset=self.dataset,
                                     batch_size=self.args.batch_size,
                                     num_workers=self.args.num_workers,# self.args.num_workers
                                     shuffle=True,
                                     pin_memory=True,
                                     drop_last=False)

        start_epoch = self.epoch
        # init eval
        # self.test(init=True)
        for ep in range(start_epoch, self.args.num_epochs + 1):
            # 记录日志
            self.writer.add_scalar("learning_rate", self.optimizer.param_groups[0]['lr'], ep)

            # 单轮训练
            self.train_one_epoch()

            # 动态学习率
            self.scheduler.step()

            
            if self.sampler is not None:
                self.scheduler_s.step()

            # 定期保存
            if self.epoch % self.args.save_cycle == 0:
                self.save()

            # 定期验证
            if self.epoch % self.args.eval_cycle == 0:
                self.test()

            self.epoch += 1

        self.save(finish=True)

    def train_one_epoch(self):
        self.model.train()
        self.dataset.train()

        epoch_loss, epoch_acc = [], []
        count = self.args.log_cycle // self.args.batch_size

        loop = tqdm(self.dataloader, total=len(self.dataloader), leave=False)
        loop.set_description('train'+str(self.epoch))

        for data in loop:
            xyzn_surf_source, xyzn_surf, xyz_volume, gts_surf, gts_vol = data
            xyzn_surf_source = xyzn_surf_source.to(self.args.device, non_blocking=True)
            xyzn_surf, xyz_volume = xyzn_surf.to(self.args.device, non_blocking=True), xyz_volume.to(self.args.device, non_blocking=True)
            gts_surf, gts_vol = gts_surf.to(self.args.device, non_blocking=True), gts_vol.to(self.args.device, non_blocking=True)

            npt_surf = xyzn_surf.shape[-1]
            npt_vol = xyz_volume.shape[-1]
            if gts_surf.ndim==2:
                gts_surf = gts_surf.unsqueeze(1)
            if gts_vol.ndim==2:
                gts_vol = gts_vol.unsqueeze(1)
            pcd_target = torch.concat((xyzn_surf, xyz_volume), dim=-1)
            gts = torch.concat((gts_surf, gts_vol), dim=-1)
            # xyzn_surf, xyz_volume, label = xyzn_surf.to(self.args.device, non_blocking=True), xyz_volume.to(self.args.device, non_blocking=True), label.to(self.args.device, non_blocking=True)

            
            if self.epoch%10 == 0:
                xyzn_surf_out = np.array(xyzn_surf_source[0,:,:].to('cpu'))
                xyz_volume_out = np.array(pcd_target[0,:,:].to('cpu'))
                label_out = np.array(gts[0,0,:].to('cpu'))
                if not os.path.exists('temp'):
                    os.mkdir('temp')
                np.savetxt(f'temp/xyzn_surf_ep{self.epoch}.csv', xyzn_surf_out)
                np.savetxt(f'temp/xyz_volume_ep{self.epoch}.csv', xyz_volume_out)
                np.savetxt(f'temp/label_ep{self.epoch}.csv', label_out)
            if torch.isnan(gts).any() or torch.isinf(gts).any():
                print(gts)
            
            # 前向传播与反向传播
            xyzn_surf_source = xyzn_surf_source.clone().detach()
            pcd_target = pcd_target.clone().detach()
            gts = gts.clone().detach()
            with self.cast():
            # with torch.autocast(device_type='cuda', dtype=torch.float16):
                points_reg = self.model(xyzn_surf_source, pcd_target)
                # loss = self.criterion(points_reg, gts) 
                # !
                points_reg_tmp = points_reg
                gts_tmp = gts
                points_reg_tmp[:,:,-npt_vol:], gts_tmp[:,:,-npt_vol:] = points_reg_tmp[:,:,-npt_vol:]*2, gts_tmp[:,:,-npt_vol:]*2
                loss = self.criterion(points_reg_tmp, gts_tmp) 
            
            self.epoch_metric_logger.add_metric('loss', loss.item())
            self.epoch_metric_logger.add_metric('corr', pearsonr(points_reg, gts).mean().item())
            acc_th = 0.05# 精度阈值0.05的准确率
            acc_surf = torch.sum(abs(points_reg[:,:,:npt_surf]-gts[:,:,:npt_surf])<acc_th)/points_reg[:,:,:npt_surf].numel()
            acc_vol = torch.sum(abs(points_reg[:,:,-npt_vol:]-gts[:,:,-npt_vol:])<acc_th/2)/points_reg[:,:,-npt_vol:].numel()
            acc = torch.sum(abs(points_reg-gts)<acc_th)/points_reg.numel()
            self.epoch_metric_logger.add_metric('acc', acc.item())
            self.epoch_metric_logger.add_metric('acc_surf', acc_surf.item())
            self.epoch_metric_logger.add_metric('acc_vol', acc_vol.item())
            # loop.set_postfix(acc=acc.item())
            # loop.set_postfix(acc_surf=acc_surf.item())
            loop.set_postfix(loss=loss.item())

            # torch.autograd.set_detect_anomaly = True

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # 记录日志
            # loop.set_postfix(train_loss=loss.item(), acc=f'{acc * 100:.2f}%')
            epoch_loss.append(loss.item())
            epoch_acc.append(acc.item())
            count -= 1
            if count <= 0:
                count = self.args.log_cycle // self.args.batch_size
                self.writer.add_scalar("train/step_loss", sum(epoch_loss[-count:]) / count, self.step)
                # print('train/step_loss', sum(epoch_loss[-count:]) / count, 'in epoch', self.epoch)
                # self.writer.add_scalar("train/step_acc", sum(epoch_acc[-count:]) / count, self.step)
                self.step += 1
        loop.set_postfix(lr=self.optimizer.param_groups[0]['lr'])
        self.epoch_metric_logger.add_metric(f'lr', self.optimizer.param_groups[0]['lr'])
        self.writer.add_scalar("train/epoch_loss", sum(epoch_loss) / len(epoch_loss), self.epoch)
        # print('Train MSE ', sum(epoch_loss) / len(epoch_loss), 'in epoch', self.epoch)
        print(f'Train Epoch {self.epoch:>4d} {self.epoch_metric_logger.tostring()}')
        # self.writer.add_scalar("train/epoch_acc", sum(epoch_acc) / len(epoch_acc), self.epoch)
        logger.info(f'Train Epoch {self.epoch:>4d} ' + self.epoch_metric_logger.tostring())
        self.epoch_metric_logger.clear()

    def test(self, init=False):
        self.model.eval()
        self.dataset.eval()
        if self.mode == 'test':
            self.epoch -= 1
        
        self.dataset.transforms.set_padding(False)
        eval_dataloader = DataLoader(dataset=self.dataset,
                                     batch_size=16,
                                     num_workers=min(self.args.num_workers, 16),
                                     pin_memory=False,
                                     drop_last=False,
                                     shuffle=False)
        
        loop = tqdm(eval_dataloader, total=len(eval_dataloader), leave=False)
        loop.set_description('eval')
        # torch.manual_seed(0)# ! seed fixed

        for data in loop:
            xyzn_surf_source, xyzn_surf, xyz_volume, gts_surf, gts_vol = data
            xyzn_surf_source = xyzn_surf_source.to(self.args.device, non_blocking=True)
            xyzn_surf, xyz_volume = xyzn_surf.to(self.args.device, non_blocking=True), xyz_volume.to(self.args.device, non_blocking=True)
            gts_surf, gts_vol = gts_surf.to(self.args.device, non_blocking=True), gts_vol.to(self.args.device, non_blocking=True)
            npt_surf = xyzn_surf.shape[-1]
            npt_vol = xyz_volume.shape[-1]
            if gts_surf.ndim==2:
                gts_surf = gts_surf.unsqueeze(1)
            if gts_vol.ndim==2:
                gts_vol = gts_vol.unsqueeze(1)
            pcd_target = torch.concat((xyzn_surf, xyz_volume), dim=-1)
            gts = torch.concat((gts_surf, gts_vol), dim=-1)
            # xyzn_surf, xyz_volume, label = data
            # xyzn_surf, xyz_volume, label = xyzn_surf.to(self.args.device, non_blocking=True), xyz_volume.to(self.args.device, non_blocking=True), label.to(self.args.device, non_blocking=True)
            # label = label.unsqueeze(1)
            if torch.isnan(gts).any() or torch.isinf(gts).any():
                print(gts)
            # 前向传播
            with torch.no_grad():
                points_reg = self.model(xyzn_surf_source, pcd_target)
                loss = self.criterion(points_reg, gts)
            # print metrics
            self.epoch_metric_logger.add_metric('loss', loss.item())
            loop.set_postfix(eval_loss=loss.item())
            

            acc_th = 0.05# 精度阈值0.05的准确率
            acc_surf = torch.sum(abs(points_reg[:,:,:npt_surf]-gts[:,:,:npt_surf])<acc_th)/points_reg[:,:,:npt_surf].numel()
            acc_vol = torch.sum(abs(points_reg[:,:,-npt_vol:]-gts[:,:,-npt_vol:])<acc_th/2)/points_reg[:,:,-npt_vol:].numel()
            acc = torch.sum(abs(points_reg-gts)<acc_th)/points_reg.numel()
            self.epoch_metric_logger.add_metric('acc', acc.item())
            self.epoch_metric_logger.add_metric('acc_surf', acc_surf.item())
            self.epoch_metric_logger.add_metric('acc_vol', acc_vol.item())
            loop.set_postfix(acc=acc.item())            
            loop.set_postfix(acc_surf=acc_surf.item())

            confidence = 0.9# 置信区间
            dev_conf, _ = torch.sort(abs(gts-points_reg).reshape(-1))
            id = int(confidence*dev_conf.numel())
            dev_conf = dev_conf[id]
            self.epoch_metric_logger.add_metric(f'dev-{confidence}', dev_conf.item())
            self.epoch_metric_logger.add_metric('corr', pearsonr(points_reg, gts).mean().item())
            loop.set_postfix(dev=dev_conf.item())

        self.dataset.transforms.set_padding(True)
        if init:
            print('Eval MSE ', self.epoch_metric_logger.tostring(), 'initial')
        else:
            print('Eval MSE ', self.epoch_metric_logger.tostring(), 'in epoch', self.epoch)
        metric = self.epoch_metric_logger.get_average_value()

        if self.mode == 'train':
            self.writer.add_scalar("eval/loss", metric['loss'], self.epoch)
            # self.writer.add_scalar("eval/acc", metric['acc'], self.epoch)
        self.epoch_metric_logger.clear()

    def save(self, finish=False):
        model_state_dict = self.model.state_dict()
        if not finish:
            state = {
                'model': model_state_dict,
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'epoch': self.epoch,
                'step': self.step,
            }
            file_path = os.path.join(self.save_root, f'{self.args.name}_{self.args.dataset}_epoch{self.epoch}.pth')
        else:
            state = {
                'model': model_state_dict,
            }
            file_path = os.path.join(self.save_root, f'{self.args.name}_{self.args.dataset}.pth')
        torch.save(state, file_path)
        # sampler
        if self.sampler is not None:
            model_state_dict = self.sampler.state_dict()
            if not finish:
                state = {
                    'model': model_state_dict,
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                    'epoch': self.epoch,
                    'step': self.step,
                }
                file_path = os.path.join(self.save_root, f'{self.args.name}_{self.args.dataset}_s_epoch{self.epoch}.pth')
            else:
                state = {
                    'model': model_state_dict,
                }
                file_path = os.path.join(self.save_root, f'{self.args.name}_{self.args.dataset}_s.pth')

            torch.save(state, file_path)


