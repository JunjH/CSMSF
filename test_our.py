import argparse
import shutil
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import loaddata
import copy
import numpy as np
import random
from utils import util
from models import modules, net, mobilenetv2
import pdb

parser = argparse.ArgumentParser(description='CSMSF: inference')
parser.add_argument('--data_path', type=str, default='/media/hujunjie/nami/multi-sensor-fusion/MS2_dataset/')
parser.add_argument('--file', type=str, default='test_day_list.csv')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def main():
    global args
    args = parser.parse_args()

    original_model = mobilenetv2.mobilenet_v2(pretrained=True)
    encoder_v = modules.E_mvnet2_img(original_model)
    model_v = net.model(encoder_v, block_channel=[24, 32, 96, 160]).to(device)
    checkpoint = torch.load("./runs/model_v.pth.tar")
    model_v.load_state_dict(checkpoint['state_dict'])

    original_model2 = mobilenetv2.mobilenet_v2(pretrained=True)
    encoder_l = modules.E_mvnet2_lidar(original_model2)
    model_l = net.model(encoder_l, block_channel=[24, 32, 96, 160]).to(device)
    checkpoint2 = torch.load("./runs/model_l.pth.tar")
    model_l.load_state_dict(checkpoint2['state_dict'])

    encoder_t = copy.deepcopy(encoder_v)
    model_t = net.model(encoder_t, block_channel=[24, 32, 96, 160]).to(device)
    checkpoint3 = torch.load("./runs/model_t.pth.tar")
    model_t.load_state_dict(checkpoint3['state_dict'])

    ######################### load module for image lidar fusion ###########################################
    model_vl = net.fusion_2sensors(block_channel=[24, 32, 96, 160]).to(device)
    checkpoint = torch.load("./runs/model_vl.pth.tar")
    model_vl.load_state_dict(checkpoint['state_dict'])

    ######################### load module for image thr fusion ###########################################
    model_vt = net.fusion_2sensors(block_channel=[24, 32, 96, 160]).to(device)
    checkpoint = torch.load("./runs/model_vt.pth.tar")
    model_vt.load_state_dict(checkpoint['state_dict'])

    ######################### load module for thr lidar fusion ###########################################
    model_lt = net.fusion_2sensors(block_channel=[24, 32, 96, 160]).to(device)
    checkpoint = torch.load("./runs/model_lt.pth.tar")
    model_lt.load_state_dict(checkpoint['state_dict'])

    ######################### load module for img lidar thr fusion ###########################################
    model_vlt = net.fusion_3sensors(block_channel=[24, 32, 96, 160]).to(device)
    checkpoint = torch.load("./runs/model_vlt.pth.tar")
    model_vlt.load_state_dict(checkpoint['state_dict'])

    batch_size = 1
    cudnn.benchmark = True

    test_loader =  loaddata.getTestingData(batch_size, args.data_path, args.file)


    #############possible noise type: {}
    sensor_type = random.choice(['v','l','t','vl','vt','lt','vlt'])
    test(test_loader, model_v, model_l, model_t, model_vl, model_vt, model_lt, model_vlt, sensor_type)

def test(test_loader, model_v, model_l, model_t, model_vl, model_vt, model_lt, model_vlt, sensor_type):
    model_v.eval()
    model_l.eval()
    model_t.eval()
    model_vl.eval()
    model_vt.eval()
    model_lt.eval()
    model_vlt.eval()

    totalNumber = 0

    errorSum = {'MSE': 0, 'RMSE': 0, 'ABS_REL': 0, 'LG10': 0,
                'MAE': 0,  'DELTA1': 0, 'DELTA2': 0, 'DELTA3': 0}


    for i, sample in enumerate(test_loader):
        img, thr, lidar, gt_img, gt_thr = sample['img'], sample['thr'], sample['lidar_thr_sd'], sample['lidar_img_gt'], sample['lidar_thr_gt']
        img, thr, lidar, gt_img, gt_thr  = img.cuda(), thr.cuda(), lidar.cuda(), gt_img.cuda(), gt_thr.cuda()

        with torch.no_grad():
            _,_,x_img_1,x_img_2,x_img_3,x_img_4 = model_v(img)
            _,_,x_lidar_1, x_lidar_2, x_lidar_3, x_lidar_4 = model_l(lidar)
            _,_,x_thr_1, x_thr_2, x_thr_3, x_thr_4 = model_t(thr)

            valid_camera = True
            valid_lidar = True
            valid_thr = True

            if(sensor_type=='vlt'):     # all sensors are valid 
                output, _ = model_vlt(x_img_1,x_img_2,x_img_3,x_img_4,x_lidar_1, x_lidar_2, x_lidar_3, x_lidar_4,x_thr_1, x_thr_2, x_thr_3, x_thr_4)
            elif(sensor_type=='vl'):
                valid_thr = False
                output, _ = model_vl(x_img_1,x_img_2,x_img_3,x_img_4,x_lidar_1, x_lidar_2, x_lidar_3, x_lidar_4)
            elif(sensor_type=='vt'):
                valid_lidar = False
                output, _ = model_vt(x_img_1,x_img_2,x_img_3,x_img_4,x_thr_1, x_thr_2, x_thr_3, x_thr_4)
            elif(sensor_type=='lt'):
                valid_camera = False
                output, _ = model_lt(x_lidar_1, x_lidar_2, x_lidar_3, x_lidar_4,x_thr_1, x_thr_2, x_thr_3, x_thr_4)
            elif(sensor_type=='t'):
                valid_camera = False
                valid_lidar = False
                output, _, _, _,_,_ = model_t(thr)
            elif(sensor_type=='l'):
                valid_camera = False
                valid_thr = False
                output, _, _, _,_,_= model_l(lidar)
            elif(sensor_type=='v'):
                valid_lidar = False
                valid_thr = False
                output, _,_, _,_,_ = model_v(img)


        if(valid_camera and not valid_lidar and not valid_thr): ######## should use ground truth of image view when only visual camera is valid
            gt = gt_img
        else:
            gt = gt_thr
        
        output = F.upsample(output, size=[gt.size(2),gt.size(3)], mode='bilinear', align_corners=True)

        batchSize = img.size(0)
        totalNumber = totalNumber + batchSize

        mask = (gt > 0)
        gt = gt[mask]
        output = output[mask]
        errors = util.evaluateError(output, gt)
        errorSum = util.addErrors(errorSum, errors, batchSize)
        averageError = util.averageErrors(errorSum, totalNumber)
                    
    print(averageError)





if __name__ == '__main__':
    main()
