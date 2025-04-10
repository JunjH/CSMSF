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
from utils import util, generate_gaussian_noise, generate_adv_noise, generate_thermal_noise, generate_density_decrease
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

    mean_feas_v = torch.load('runs/mean_feas_img.pt')
    mean_feas_l = torch.load('runs/mean_feas_lidar.pt')
    mean_feas_t = torch.load('runs/mean_feas_thr.pt')

    delta = 0.7 ###### similarity threshold 

    #############possible noise type: {}
    # noise_type = random.choice(['non_uniform_thr','gaussian','adv','mechanical','density_decrease',None])
    noise_type = 'None'
    print(noise_type)
    test(test_loader, model_v, model_l, model_t, model_vl, model_vt, model_lt, model_vlt, mean_feas_v, mean_feas_l, mean_feas_t, delta, noise_type)


def test(test_loader, model_v, model_l, model_t, model_vl, model_vt, model_lt, model_vlt, mean_feas_v, mean_feas_l, mean_feas_t, delta, noise_type):
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

    count_failure_img = 0
    count_failure_lidar = 0
    count_failure_thr = 0
    count_failure_case = 0

    for i, sample in enumerate(test_loader):
        img, thr, lidar, gt_img, gt_thr = sample['img'], sample['thr'], sample['lidar_thr_sd'], sample['lidar_img_gt'], sample['lidar_thr_gt']
        img, thr, lidar, gt_img, gt_thr  = img.cuda(), thr.cuda(), lidar.cuda(), gt_img.cuda(), gt_thr.cuda()

        valid_camera = True
        valid_lidar = True
        valid_thr = True

        if(noise_type == 'non_uniform_thr'):
            thr = generate_thermal_noise.generate_noise(thr, sigma_strip=0.1)        
        elif(noise_type == 'gaussian'):
            img = generate_gaussian_noise.generate_noise_img(img.clone(), std=0.1)       
            lidar = generate_gaussian_noise.generate_noise_lidar(lidar.clone(),std=0.1)
            thr = generate_gaussian_noise.generate_noise_img(thr.clone(),std=0)
        elif(noise_type == 'adv'):
            img = generate_adv_noise.generate_adv_img(img, gt_img, model_v, epsilon=0,iteration=1)
            lidar = generate_adv_noise.generate_adv_img(lidar, gt_thr, model_l, epsilon=0.1*lidar.max(),iteration=1)
            thr = generate_adv_noise.generate_adv_img(thr, gt_thr, model_t, epsilon=0.1,iteration=1)
        elif(noise_type == 'mechanical'): #########machnical failure can be directly identified by quantifyiing sparsity of the input
            thr.zero_() 
            lidar.zero_() 
            valid_thr = False
            valid_lidar = False
        elif(noise_type == 'density_decrease'): #########LiDAR density decrease can be directly identified by quatifying sparsity of the input     
            lidar  = generate_density_decrease.reduce_density(lidar, decrease_rate=0.95)     
            valid_lidar = False
        else:
            pass

        with torch.no_grad():
            _,_,x_img_1,x_img_2,x_img_3,x_img_4 = model_v(img)
            _,_,x_lidar_1, x_lidar_2, x_lidar_3, x_lidar_4 = model_l(lidar)
            _,_,x_thr_1, x_thr_2, x_thr_3, x_thr_4 = model_t(thr)

        if(F.cosine_similarity(mean_feas_v,x_img_1).mean()<delta):
            count_failure_img += 1
            valid_camera = False
        if(F.cosine_similarity(mean_feas_l,x_lidar_1).mean()<delta):
            count_failure_lidar += 1
            valid_lidar = False
        if(F.cosine_similarity(mean_feas_t,x_thr_1).mean()<delta):           
            count_failure_thr += 1
            valid_thr = False

        if(valid_camera and valid_lidar and valid_thr):     # all sensors are valid 
            output, _ = model_vlt(x_img_1,x_img_2,x_img_3,x_img_4,x_lidar_1, x_lidar_2, x_lidar_3, x_lidar_4,x_thr_1, x_thr_2, x_thr_3, x_thr_4)
        elif(valid_camera and valid_lidar and not valid_thr):
            output, _ = model_vl(x_img_1,x_img_2,x_img_3,x_img_4,x_lidar_1, x_lidar_2, x_lidar_3, x_lidar_4)
        elif(valid_camera and not valid_lidar and valid_thr):
            output, _ = model_vt(x_img_1,x_img_2,x_img_3,x_img_4,x_thr_1, x_thr_2, x_thr_3, x_thr_4)
        elif(not valid_camera and valid_lidar and valid_thr):
            output, _ = model_lt(x_lidar_1, x_lidar_2, x_lidar_3, x_lidar_4,x_thr_1, x_thr_2, x_thr_3, x_thr_4)
        elif(not valid_camera and not valid_lidar and valid_thr):
            output, _, _, _,_,_ = model_t(thr)
        elif(not valid_camera and valid_lidar and not valid_thr):
            output, _, _, _,_,_= model_l(lidar_thr_sd)
        elif(valid_camera and not valid_lidar and not valid_thr):
            output, _,_, _,_,_ = model_v(img)

        elif(not valid_camera and not valid_lidar and not valid_thr):
            count_failure_case += 1
            flag = np.argmax([F.cosine_similarity(mean_feas_v,x_img_1).mean().data.cpu(),F.cosine_similarity(mean_feas_l,x_lidar_1).mean().data.cpu(),F.cosine_similarity(mean_feas_t,x_thr_1).mean().data.cpu()])
            if(flag==0):
                output, _,_, _,_,_ = model_v(img)
            elif(flag==1):
                output, _, _, _,_,_= model_l(lidar)
            elif(flag==2):
                output, _, _, _,_,_ = model_t(thr)

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
                
    print(count_failure_img ,count_failure_lidar,count_failure_thr, count_failure_case,totalNumber)
    print(averageError)





if __name__ == '__main__':
    main()
