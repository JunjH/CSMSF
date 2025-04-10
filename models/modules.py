from collections import OrderedDict
import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils import model_zoo
import copy
import numpy as np
import pdb

class _UpProjection(nn.Sequential):

    def __init__(self, num_input_features, num_output_features):
        super(_UpProjection, self).__init__()

        self.conv1 = nn.Conv2d(num_input_features, num_output_features, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(num_output_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(num_output_features, num_output_features, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_2 = nn.BatchNorm2d(num_output_features)
        self.conv2 = nn.Conv2d(num_input_features, num_output_features, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(num_output_features)

    def forward(self, x, size):
        x = F.upsample(x, size=size, mode='bilinear')
        x_conv1 = self.relu(self.bn1(self.conv1(x)))
        bran1 = self.bn1_2(self.conv1_2(x_conv1))
        bran2 = self.bn2(self.conv2(x))

        out = self.relu(bran1 + bran2)

        return out




class hswish(nn.Module):

    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out

class E_mvnet(nn.Module):

    def __init__(self, original_model):
        super(E_mvnet, self).__init__()

        self.mv = original_model.model[:-1]

    def forward(self, x):
        x_block1 =self.mv[0:4](x)
        x_block2 = self.mv[4:6](x_block1)
        x_block3 = self.mv[6:12](x_block2)
        x_block4 = self.mv[12:](x_block3)
        # pdb.set_trace()
        return x_block1, x_block2, x_block3, x_block4

class E_mvnet2_EF(nn.Module):

    def __init__(self, original_model):
        super(E_mvnet2_EF, self).__init__()

        self.mv2 = original_model.features[:17]

        self.conv1_d = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1_d = nn.BatchNorm2d(32)
        self.relu_d = nn.ReLU(inplace=True)

        self.conv1_t = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1_t = nn.BatchNorm2d(32)
        self.relu_t = nn.ReLU(inplace=True)

    def forward(self, rgb, lidar, thermal):
        x_rgb = self.mv2[0](rgb)
        x_lidar = self.relu_d(self.bn1_d(self.conv1_d(lidar)))
        x_thermal = self.relu_t(self.bn1_t(self.conv1_t(thermal)))

        x_block1 =self.mv2[1:4](x_rgb + x_lidar + x_thermal)
        x_block2 = self.mv2[4:7](x_block1)
        x_block3 = self.mv2[7:14](x_block2)
        x_block4 = self.mv2[14:](x_block3)
  
        return x_block1, x_block2, x_block3, x_block4


class E_mvnet2_img(nn.Module):

    def __init__(self, original_model):
        super(E_mvnet2_img, self).__init__()

        self.mv2 = original_model.features[:17]

    def forward(self, rgb):

        x_rgb = self.mv2[0](rgb)

        x_block1 =self.mv2[1:4](x_rgb)
        x_block2 = self.mv2[4:7](x_block1)
        x_block3 = self.mv2[7:14](x_block2)
        x_block4 = self.mv2[14:](x_block3)
        
        return x_block1, x_block2, x_block3, x_block4


class E_mvnet2_lidar(nn.Module):

    def __init__(self, original_model):
        super(E_mvnet2_lidar, self).__init__()


        self.conv1_d = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1_d = nn.BatchNorm2d(32)
        self.relu_d = nn.ReLU(inplace=True)

        self.mv2 = original_model.features[:17]

    def forward(self, d):

        x_d = self.relu_d(self.bn1_d(self.conv1_d(d)))

        x_block1 =self.mv2[1:4](x_d)
        x_block2 = self.mv2[4:7](x_block1)
        x_block3 = self.mv2[7:14](x_block2)
        x_block4 = self.mv2[14:](x_block3)
        
        return x_block1, x_block2, x_block3, x_block4



class E_mvnet2_thermal(nn.Module):

    def __init__(self, original_model):
        super(E_mvnet2_thermal, self).__init__()

        self.conv1_d = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1_d = nn.BatchNorm2d(32)
        self.relu_d = nn.ReLU(inplace=True)

        self.mv2 = original_model.features[:17]

    def forward(self, t):

        x_d = self.relu_d(self.bn1_d(self.conv1_d(t)))

        x_block1 =self.mv2[1:4](x_d)
        x_block2 = self.mv2[4:7](x_block1)
        x_block3 = self.mv2[7:14](x_block2)
        x_block4 = self.mv2[14:](x_block3)

        return x_block1, x_block2, x_block3, x_block4


class E_resnet(nn.Module):

    def __init__(self, original_model, num_features=2048):
        super(E_resnet, self).__init__()
        self.conv1 = original_model.conv1
        self.bn1 = original_model.bn1
        self.relu = original_model.relu
        self.maxpool = original_model.maxpool

        self.layer1 = original_model.layer1
        self.layer2 = original_model.layer2
        self.layer3 = original_model.layer3
        self.layer4 = original_model.layer4

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x_block1 = self.layer1(x)
        x_block2 = self.layer2(x_block1)
        x_block3 = self.layer3(x_block2)
        x_block4 = self.layer4(x_block3)

        return x_block1, x_block2, x_block3, x_block4

class E_resnet2(nn.Module):

    def __init__(self, original_model, num_features=2048):
        super(E_resnet2, self).__init__()
        self.conv1_rgb = original_model.conv1
        self.bn1_rgb = original_model.bn1
        self.relu_rgb = original_model.relu
        self.maxpool_rgb = original_model.maxpool

        self.conv1_d = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1_d = nn.BatchNorm2d(64)
        self.relu_d = nn.ReLU(inplace=True)
        self.maxpool_d = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = original_model.layer1
        self.layer2 = original_model.layer2
        self.layer3 = original_model.layer3
        self.layer4 = original_model.layer4

    def forward(self, rgb, d, mode=0):
        if(mode==0):
            x_rgb = self.maxpool_rgb(self.relu_rgb(self.bn1_rgb(self.conv1_rgb(rgb))))
            x_d = self.maxpool_d(self.relu_d(self.bn1_d(self.conv1_d(d))))
            x_block1 = self.layer1(x_rgb + x_d)
        elif(mode==1):
            x_rgb = self.maxpool_rgb(self.relu_rgb(self.bn1_rgb(self.conv1_rgb(rgb))))
            x_block1 = self.layer1(x_rgb)
        elif(mode==2):
            x_d = self.maxpool_d(self.relu_d(self.bn1_d(self.conv1_d(d))))
            x_block1 = self.layer1(x_d)

        x_block2 = self.layer2(x_block1)
        x_block3 = self.layer3(x_block2)
        x_block4 = self.layer4(x_block3)

        return x_block1, x_block2, x_block3, x_block4

class E_resnet3(nn.Module):

    def __init__(self, original_model, num_features=2048):
        super(E_resnet3, self).__init__()
        self.conv1_rgb = original_model.conv1
        self.bn1_rgb = original_model.bn1
        self.relu_rgb = original_model.relu
        self.maxpool_rgb = original_model.maxpool

        self.conv1_d = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1_d = nn.BatchNorm2d(64)
        self.relu_d = nn.ReLU(inplace=True)
        self.maxpool_d = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv1_sd = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1_sd = nn.BatchNorm2d(64)
        self.relu_sd = nn.ReLU(inplace=True)
        self.maxpool_sd = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = original_model.layer1
        self.layer2 = original_model.layer2
        self.layer3 = original_model.layer3
        self.layer4 = original_model.layer4

    def forward(self, rgb, sd, d):
        x_rgb = self.maxpool_rgb(self.relu_rgb(self.bn1_rgb(self.conv1_rgb(rgb))))
        x_sd = self.maxpool_sd(self.relu_sd(self.bn1_sd(self.conv1_sd(sd))))
        x_d = self.maxpool_d(self.relu_d(self.bn1_d(self.conv1_d(d))))

        x_block1 = self.layer1(x_rgb + x_d + x_sd)
        x_block2 = self.layer2(x_block1)
        x_block3 = self.layer3(x_block2)
        x_block4 = self.layer4(x_block3)

        return x_block1, x_block2, x_block3, x_block4



class D(nn.Module):
    def __init__(self, block_channel, num_features = 512):
        super(D, self).__init__()
        self.up1 = _UpProjection(num_input_features=block_channel[3], num_output_features=block_channel[2])
        self.up2 = _UpProjection(num_input_features=block_channel[2], num_output_features=block_channel[1])
        self.up3 = _UpProjection(num_input_features=block_channel[1], num_output_features=block_channel[0])
        self.up4 = _UpProjection(num_input_features=block_channel[0], num_output_features=block_channel[0] // 2)


    def forward(self, x_block1, x_block2, x_block3, x_block4):
        x_d1 = self.up1(x_block4, [x_block3.size(2), x_block3.size(3)])
        x_d2 = self.up2(x_d1 + x_block3, [x_block2.size(2), x_block2.size(3)])
        x_d3 = self.up3(x_d2 + x_block2, [x_block1.size(2), x_block1.size(3)])
        x_d4 = self.up4(x_d3 + x_block1, [x_block1.size(2)*2, x_block1.size(3)*2])

        return x_d4, x_d3, x_d2, x_d1


class fusion_layer(nn.Module):

    def __init__(self, block_channel, num_features = 512):
        super(fusion_layer, self).__init__()
        self.att4_img = self.attention(block_channel[3], block_channel[3] // 16)
        self.att3_img = self.attention(block_channel[2], block_channel[2] // 16)
        self.att2_img = self.attention(block_channel[1], block_channel[1] // 16)
        self.att1_img = self.attention(block_channel[0], block_channel[0] // 16)

        self.att4_lidar = self.attention(block_channel[3], block_channel[3] // 16)
        self.att3_lidar = self.attention(block_channel[2], block_channel[2] // 16)
        self.att2_lidar = self.attention(block_channel[1], block_channel[1] // 16)
        self.att1_lidar = self.attention(block_channel[0], block_channel[0] // 16)


        self.up1 = _UpProjection(num_input_features=block_channel[3]*2, num_output_features=block_channel[2])
        self.up2 = _UpProjection(num_input_features=block_channel[2]*3, num_output_features=block_channel[1])
        self.up3 = _UpProjection(num_input_features=block_channel[1]*3, num_output_features=block_channel[0])
        self.up4 = _UpProjection(num_input_features=block_channel[0]*3, num_output_features=block_channel[0] // 2)

    def attention(self, features1, features2):
        prior = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        conv1 = nn.Conv2d(features1, features2, kernel_size=1, bias=False)
        relu = nn.ReLU()
        conv2 = nn.Conv2d(features2, features1, kernel_size=1, bias=False)
        sigmoid = nn.Sigmoid()
        return nn.Sequential(prior, conv1, relu, conv2, sigmoid)

    def forward(self, x_img_1, x_img_2, x_img_3, x_img_4, x_lidar_1, x_lidar_2, x_lidar_3, x_lidar_4):
        x_att1_img = self.att1_img(x_img_1)
        x_att2_img = self.att2_img(x_img_2)
        x_att3_img = self.att3_img(x_img_3)
        x_att4_img = self.att4_img(x_img_4)

        x_att1_lidar = self.att1_lidar(x_lidar_1)
        x_att2_lidar = self.att2_lidar(x_lidar_2)
        x_att3_lidar = self.att3_lidar(x_lidar_3)
        x_att4_lidar = self.att4_lidar(x_lidar_4)

        x1 = torch.cat((x_lidar_4 * x_att4_lidar,x_img_4 * x_att4_img),1)
        x_d1 = self.up1(x1, [x_lidar_3.size(2), x_lidar_3.size(3)])

        x2 = torch.cat((x_d1, x_lidar_3 * x_att3_lidar,x_img_3 * x_att3_img),1)
        x_d2 = self.up2(x2,  [x_lidar_2.size(2), x_lidar_2.size(3)])

        x3 = torch.cat((x_d2, x_lidar_2 * x_att2_lidar, x_img_2 * x_att2_img),1)
        x_d3 = self.up3(x3,  [x_lidar_1.size(2), x_lidar_1.size(3)])

        x4 = torch.cat((x_d3, x_lidar_1 * x_att1_lidar,x_img_1 * x_att1_img),1)
        x_d4 = self.up4(x4,  [x_lidar_1.size(2)*2, x_lidar_1.size(3)*2])

        return x_d4


class fusion_layer2(nn.Module):

    def __init__(self, block_channel, num_features = 512):
        super(fusion_layer2, self).__init__()
        self.att4_img = self.attention(block_channel[3], block_channel[3] // 16)
        self.att3_img = self.attention(block_channel[2], block_channel[2] // 16)
        self.att2_img = self.attention(block_channel[1], block_channel[1] // 16)
        self.att1_img = self.attention(block_channel[0], block_channel[0] // 16)

        self.att4_lidar = self.attention(block_channel[3], block_channel[3] // 16)
        self.att3_lidar = self.attention(block_channel[2], block_channel[2] // 16)
        self.att2_lidar = self.attention(block_channel[1], block_channel[1] // 16)
        self.att1_lidar = self.attention(block_channel[0], block_channel[0] // 16)

        self.att4_thr = self.attention(block_channel[3], block_channel[3] // 16)
        self.att3_thr = self.attention(block_channel[2], block_channel[2] // 16)
        self.att2_thr = self.attention(block_channel[1], block_channel[1] // 16)
        self.att1_thr = self.attention(block_channel[0], block_channel[0] // 16)

        self.up1 = _UpProjection(num_input_features=block_channel[3]*3, num_output_features=block_channel[2])
        self.up2 = _UpProjection(num_input_features=block_channel[2]*4, num_output_features=block_channel[1])
        self.up3 = _UpProjection(num_input_features=block_channel[1]*4, num_output_features=block_channel[0])
        self.up4 = _UpProjection(num_input_features=block_channel[0]*4, num_output_features=block_channel[0] // 2)

    def attention(self, features1, features2):
        prior = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        conv1 = nn.Conv2d(features1, features2, kernel_size=1, bias=False)
        # bn = nn.BatchNorm2d(features)
        relu = nn.ReLU()
        conv2 = nn.Conv2d(features2, features1, kernel_size=1, bias=False)
        sigmoid = nn.Sigmoid()
        return nn.Sequential(prior, conv1, relu, conv2, sigmoid)

    def forward(self, x_img_1, x_img_2, x_img_3, x_img_4, x_lidar_1, x_lidar_2, x_lidar_3, x_lidar_4, x_thr_1, x_thr_2, x_thr_3, x_thr_4):
        x_att1_img = self.att1_img(x_img_1)
        x_att2_img = self.att2_img(x_img_2)
        x_att3_img = self.att3_img(x_img_3)
        x_att4_img = self.att4_img(x_img_4)

        x_att1_lidar = self.att1_lidar(x_lidar_1)
        x_att2_lidar = self.att2_lidar(x_lidar_2)
        x_att3_lidar = self.att3_lidar(x_lidar_3)
        x_att4_lidar = self.att4_lidar(x_lidar_4)

        x_att1_thr = self.att1_thr(x_thr_1)
        x_att2_thr = self.att2_thr(x_thr_2)
        x_att3_thr = self.att3_thr(x_thr_3)
        x_att4_thr = self.att4_thr(x_thr_4)

        x1 = torch.cat((x_lidar_4 * x_att4_lidar,x_img_4 * x_att4_img, x_thr_4 * x_att4_thr),1)
        x_d1 = self.up1(x1, [x_lidar_3.size(2), x_lidar_3.size(3)])

        x2 = torch.cat((x_d1, x_lidar_3 * x_att3_lidar,x_img_3 * x_att3_img,x_thr_3 * x_att3_thr),1)
        x_d2 = self.up2(x2,  [x_lidar_2.size(2), x_lidar_2.size(3)])

        x3 = torch.cat((x_d2, x_lidar_2 * x_att2_lidar, x_img_2 * x_att2_img, x_thr_2 * x_att2_thr),1)
        x_d3 = self.up3(x3,  [x_lidar_1.size(2), x_lidar_1.size(3)])

        x4 = torch.cat((x_d3, x_lidar_1 * x_att1_lidar,x_img_1 * x_att1_img,x_thr_1 * x_att1_thr),1)
        x_d4 = self.up4(x4,  [x_lidar_1.size(2)*2, x_lidar_1.size(3)*2])

        return x_d4

class fusion_layer3(nn.Module):

    def __init__(self, block_channel, num_features = 512):
        super(fusion_layer3, self).__init__()
        self.att4_img = self.attention(block_channel[3], block_channel[3] // 16)
        self.att3_img = self.attention(block_channel[2], block_channel[2] // 16)
        self.att2_img = self.attention(block_channel[1], block_channel[1] // 16)
        self.att1_img = self.attention(block_channel[0], block_channel[0] // 16)

        self.att4_lidar = self.attention(block_channel[3], block_channel[3] // 16)
        self.att3_lidar = self.attention(block_channel[2], block_channel[2] // 16)
        self.att2_lidar = self.attention(block_channel[1], block_channel[1] // 16)
        self.att1_lidar = self.attention(block_channel[0], block_channel[0] // 16)

        self.att4_thr = self.attention(block_channel[3], block_channel[3] // 16)
        self.att3_thr = self.attention(block_channel[2], block_channel[2] // 16)
        self.att2_thr = self.attention(block_channel[1], block_channel[1] // 16)
        self.att1_thr = self.attention(block_channel[0], block_channel[0] // 16)

        self.up1 = _UpProjection(num_input_features=block_channel[3]*2, num_output_features=block_channel[2])
        self.up2 = _UpProjection(num_input_features=block_channel[2]*3, num_output_features=block_channel[1])
        self.up3 = _UpProjection(num_input_features=block_channel[1]*3, num_output_features=block_channel[0])
        self.up4 = _UpProjection(num_input_features=block_channel[0]*3, num_output_features=block_channel[0] // 2)


        self.conv1 = nn.Conv2d(block_channel[3]*3,block_channel[3]*2,kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(block_channel[3]*2)

        self.conv2 = nn.Conv2d(block_channel[2]*4,block_channel[2]*3,kernel_size=5, stride=1, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(block_channel[2]*3)

        self.conv3 = nn.Conv2d(block_channel[1]*4,block_channel[1]*3,kernel_size=5, stride=1, padding=2, bias=False)
        self.bn3 = nn.BatchNorm2d(block_channel[1]*3)

        self.conv4 = nn.Conv2d(block_channel[0]*4,block_channel[0]*3,kernel_size=5, stride=1, padding=2, bias=False)
        self.bn4 = nn.BatchNorm2d(block_channel[0]*3)

    def attention(self, features1, features2):
        prior = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        conv1 = nn.Conv2d(features1, features2, kernel_size=1, bias=False)
        # bn = nn.BatchNorm2d(features)
        relu = nn.ReLU()
        conv2 = nn.Conv2d(features2, features1, kernel_size=1, bias=False)
        sigmoid = nn.Sigmoid()
        return nn.Sequential(prior, conv1, relu, conv2, sigmoid)

    def forward(self, x_img_1, x_img_2, x_img_3, x_img_4, x_lidar_1, x_lidar_2, x_lidar_3, x_lidar_4, x_thr_1, x_thr_2, x_thr_3, x_thr_4):
        x_att1_img = self.att1_img(x_img_1)
        x_att2_img = self.att2_img(x_img_2)
        x_att3_img = self.att3_img(x_img_3)
        x_att4_img = self.att4_img(x_img_4)

        x_att1_lidar = self.att1_lidar(x_lidar_1)
        x_att2_lidar = self.att2_lidar(x_lidar_2)
        x_att3_lidar = self.att3_lidar(x_lidar_3)
        x_att4_lidar = self.att4_lidar(x_lidar_4)

        x_att1_thr = self.att1_thr(x_thr_1)
        x_att2_thr = self.att2_thr(x_thr_2)
        x_att3_thr = self.att3_thr(x_thr_3)
        x_att4_thr = self.att4_thr(x_thr_4)

        x1 = torch.cat((x_lidar_4 * x_att4_lidar,x_img_4 * x_att4_img, x_thr_4 * x_att4_thr),1)
        x1 = F.relu(self.bn1(self.conv1(x1)))
        x_d1 = self.up1(x1, [x_lidar_3.size(2), x_lidar_3.size(3)])

        x2 = torch.cat((x_d1, x_lidar_3 * x_att3_lidar,x_img_3 * x_att3_img,x_thr_3 * x_att3_thr),1)
        x2 = F.relu(self.bn2(self.conv2(x2)))
        x_d2 = self.up2(x2,  [x_lidar_2.size(2), x_lidar_2.size(3)])

        x3 = torch.cat((x_d2, x_lidar_2 * x_att2_lidar, x_img_2 * x_att2_img, x_thr_2 * x_att2_thr),1)
        x3 = F.relu(self.bn3(self.conv3(x3)))
        x_d3 = self.up3(x3,  [x_lidar_1.size(2), x_lidar_1.size(3)])

        x4 = torch.cat((x_d3, x_lidar_1 * x_att1_lidar,x_img_1 * x_att1_img,x_thr_1 * x_att1_thr),1)
        x4 = F.relu(self.bn4(self.conv4(x4)))
        x_d4 = self.up4(x4,  [x_lidar_1.size(2)*2, x_lidar_1.size(3)*2])

        return x_d4



class R_um(nn.Module):

    def __init__(self, num_features=32, num_output_features=1):
        super(R_um, self).__init__()
        self.conv0 = nn.Conv2d(num_features, num_features,kernel_size=5, stride=1, padding=2, bias=False)
        self.bn0 = nn.BatchNorm2d(num_features)

        self.conv1 = nn.Conv2d(num_features, num_output_features, kernel_size=5, stride=1, padding=2, bias=True)

    def forward(self, x):
        x0 = self.conv0(x)
        x0 = self.bn0(x0)
        x0 = F.relu(x0)

        out = self.conv1(x0)

        return out




