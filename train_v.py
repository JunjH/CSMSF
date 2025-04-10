import argparse
import shutil
import time
import os
import pdb
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import loaddata
from utils import util, utils
import copy
import numpy as np

import matplotlib.image
import matplotlib.pyplot as plt
plt.set_cmap("jet")

from models import modules, net, mobilenetv2


parser = argparse.ArgumentParser(description='training on vision')
parser.add_argument('--epochs', default=50, type=int, help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,help='manual epoch number (useful on restarts)')

parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, help='initial learning rate')
parser.add_argument('--lr_policy', type=str, default='plateau', help='{}learning rate policy: lambda|step|plateau')
parser.add_argument('--lr_decay_iters', type=int, default=7, help='multiply by a gamma every lr_decay_iters iterations')
parser.add_argument('--gamma', type=float, default=0.5, help='factor to decay learning rate every lr_decay_iters with')

parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', '--wd', default=0, type=float, help='weight decay (default: 1e-4)')
parser.add_argument('--name', default='model_v', type=str, help='name of experiment')

parser.add_argument('--data_path', type=str, default='./MS2_dataset/')
parser.add_argument('--train_file', type=str, default='train_list.csv')
parser.add_argument('--val_file', type=str, default='val_list.csv')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def main():
    global args
    args = parser.parse_args()

    original_model = mobilenetv2.mobilenet_v2(pretrained=True)
    encoder = modules.E_mvnet2_img(original_model)
    model = net.model(encoder, block_channel=[24, 32, 96, 160])
    print('Number of parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    model.to(device)
    batch_size = 8

    cudnn.benchmark = True
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)

    train_loader = loaddata.getTrainingData(batch_size, args.data_path, args.train_file)
    test_loader =  loaddata.getTestingData(batch_size, args.data_path, args.val_file)

    best_prec1 = 0

    model = model.cuda()
    scheduler = utils.define_scheduler(optimizer, args)

    for epoch in range(args.start_epoch, args.epochs):
        if args.lr_policy is not None and args.lr_policy != 'plateau':
            scheduler.step()
            lr = optimizer.param_groups[0]['lr']
            print('lr is set to {}'.format(lr))

        adjust_learning_rate(optimizer, epoch)
        train(train_loader, model, optimizer)
        total_score = test(test_loader, model)

        if args.lr_policy == 'plateau':
            scheduler.step(total_score)
            lr = optimizer.param_groups[0]['lr']
            print('LR plateaued, hence is set to {}'.format(lr))

        is_best = total_score > best_prec1
        best_prec1 = max(total_score, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best)


def train(train_loader, net, optimizer):
    net.train()

    for i, sample in enumerate(train_loader):
        img, lidar_img_sd, lidar_img_gt, thr, lidar_thr_sd, lidar_thr_gt = sample['img'], sample['lidar_img_sd'], sample['lidar_img_gt'], sample['thr'], sample['lidar_thr_sd'], sample['lidar_thr_gt']

        img = img.cuda()
        lidar_img_sd, lidar_img_gt = lidar_img_sd.cuda(), lidar_img_gt.cuda()
        lidar_thr_sd, lidar_thr_gt = lidar_thr_sd.cuda(), lidar_thr_gt.cuda()
        # pdb.set_trace()
        
        optimizer.zero_grad()
        pred, um, _,_,_,_   = net(img)
        pred = torch.nn.functional.upsample(pred, size=[lidar_img_gt.size(2),lidar_img_gt.size(3)], mode='bilinear', align_corners=True)
        um = torch.nn.functional.upsample(um, size=[lidar_img_gt.size(2),lidar_img_gt.size(3)], mode='bilinear', align_corners=True)
        
        mask = (lidar_img_gt > 0)
        lidar_img_gt = lidar_img_gt[mask]
        pred, um = pred[mask], um[mask]
        
        loss_d = (torch.exp(-um) * (pred/lidar_img_gt.median()-lidar_img_gt/lidar_img_gt.median())**2 + 2*um).mean()

        loss_d.backward()
        optimizer.step()

        if i % 5000 == 0:
            print(i, loss_d.item())
            print('mae',(pred-lidar_img_gt).abs().mean().item())
            print(i,um.mean().item())
         
    
    
def test(test_loader, net):
    net.eval()

    totalNumber = 0

    errorSum = {'MSE': 0, 'RMSE': 0, 'ABS_REL': 0, 'LG10': 0,
                'MAE': 0,  'DELTA1': 0, 'DELTA2': 0, 'DELTA3': 0}

    with torch.no_grad():
        for i, sample in enumerate(test_loader):
            img, lidar_img_sd, lidar_img_gt, thr, lidar_thr_sd, lidar_thr_gt = sample['img'], sample['lidar_img_sd'], sample['lidar_img_gt'], sample['thr'], sample['lidar_thr_sd'], sample['lidar_thr_gt']

            img = img.cuda()
            lidar_img_sd, lidar_img_gt = lidar_img_sd.cuda(), lidar_img_gt.cuda()
            lidar_thr_sd, lidar_thr_gt = lidar_thr_sd.cuda(), lidar_thr_gt.cuda()

            output, um,  _,_,_,_  = net(img)
            output = torch.nn.functional.upsample(output, size=[lidar_img_gt.size(2),lidar_img_gt.size(3)], mode='bilinear', align_corners=True)

            mask = (lidar_img_gt > 0)
            lidar_img_gt = lidar_img_gt[mask]
            output = output[mask]
            
            batchSize = img.size(0)
            totalNumber = totalNumber + batchSize
            errors = util.evaluateError(output, lidar_img_gt)
            errorSum = util.addErrors(errorSum, errors, batchSize)
            averageError = util.averageErrors(errorSum, totalNumber)

        print(averageError)

    return averageError['DELTA1']






def adjust_learning_rate(optimizer, epoch):
    lr = args.lr * (0.5 ** (epoch // 5))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr





class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    directory = "runs/%s/" % (args.name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)

    if is_best:
        shutil.copyfile(filename, 'runs/%s/' %
                        (args.name) + 'model_best.pth.tar')




if __name__ == '__main__':
    main()
