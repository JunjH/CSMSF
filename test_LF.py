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
from utils import util
import copy
import numpy as np
import random
import matplotlib.image
import matplotlib.pyplot as plt
plt.set_cmap("plasma")


from models import modules, net, mobilenetv2


parser = argparse.ArgumentParser(description='EF: inference')
parser.add_argument('--data_path', type=str, default='/media/hujunjie/nami/multi-sensor-fusion/MS2_dataset/')
parser.add_argument('--file', type=str, default='test_day_list.csv')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def main():
    global args
    args = parser.parse_args()

    original_model = mobilenetv2.mobilenet_v2(pretrained=True)
    encoder_img = modules.E_mvnet2_img(original_model)

    original_model2 = mobilenetv2.mobilenet_v2(pretrained=True)
    encoder_lidar = modules.E_mvnet2_lidar(original_model2)

    original_model3 = mobilenetv2.mobilenet_v2(pretrained=True)
    encoder_thr = modules.E_mvnet2_img(original_model3)

    model = net.model_LF(encoder_img, encoder_lidar, encoder_thr, block_channel=[24, 32, 96, 160])
    print('Number of parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    model.to(device)

    checkpoint = torch.load("./runs/model_LF.pth.tar")
    model.load_state_dict(checkpoint['state_dict'])

    batch_size = 1

    cudnn.benchmark = True

    test_loader =  loaddata.getTestingData(batch_size, args.data_path, args.file)
    
    test(test_loader, model)
    
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
            thr = thr.cuda()
            lidar_thr_sd, lidar_thr_gt = lidar_thr_sd.cuda(), lidar_thr_gt.cuda()

            output, um = net(img, lidar_thr_sd,thr)

            output = torch.nn.functional.upsample(output, size=[lidar_thr_gt.size(2),lidar_thr_gt.size(3)], mode='bilinear', align_corners=True)

            mask = (lidar_thr_gt > 0)
            lidar_thr_gt = lidar_thr_gt[mask]
            output = output[mask]
            
            batchSize = img.size(0)
            totalNumber = totalNumber + batchSize
            errors = util.evaluateError(output, lidar_thr_gt)
            errorSum = util.addErrors(errorSum, errors, batchSize)
            averageError = util.averageErrors(errorSum, totalNumber)
        print(averageError)

    return averageError['DELTA1']


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


if __name__ == '__main__':
    main()
