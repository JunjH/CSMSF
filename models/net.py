import torch
import torch.nn.functional as F
import torch.nn as nn
import copy
import numpy as np
from .modules import *
import random


class model_EF(nn.Module):
    def __init__(self, Encoder, block_channel):
        super(model_EF, self).__init__()
        self.E = Encoder
        self.D = D(block_channel,num_features=block_channel[3])
        self.R1 = R_um(num_features=block_channel[0]//2)
        self.R2 = R_um(num_features=block_channel[0]//2)

    def forward(self, rgb,lidar,thermal):
        x_e1, x_e2, x_e3, x_e4 = self.E(rgb,lidar,thermal)
        x_d4, xd3, x_d2, x_d1 = self.D(x_e1, x_e2, x_e3, x_e4)
        out = self.R1(x_d4)
        uncetainty = self.R2(x_d4)

        return out, uncetainty

class IF(nn.Module):
    def __init__(self, Encoder_img, Encoder_lidar, Encoder_thr, block_channel):

        super(IF, self).__init__()
        self.E_img = Encoder_img
        self.E_lidar = Encoder_lidar
        self.E_thr = Encoder_thr

        self.D_fusion = fusion_layer2(block_channel,num_features=block_channel[3])

        self.R1_fusion = R_um(num_features=block_channel[0]//2)
        self.R2_fusion = R_um(num_features=block_channel[0]//2)

    def forward(self, input_img, input_lidar, input_thr):
        x1, x2, x3, x4 = self.E_img(input_img)
        x1_, x2_, x3_, x4_ = self.E_lidar(input_lidar)
        x1__, x2__, x3__, x4__ = self.E_thr(input_thr)

        x_decoder = self.D_fusion(x1, x2, x3, x4, x1_, x2_, x3_, x4_,x1__, x2__, x3__, x4__)
        out_fusion = self.R1_fusion(x_decoder)
        uncetainty_fusion = self.R2_fusion(x_decoder)

        return out_fusion, uncetainty_fusion

class model_LF(nn.Module):
    def __init__(self, Encoder_img, Encoder_lidar, Encoder_thr, block_channel):
        super(model_LF, self).__init__()
        self.E_img = Encoder_img
        self.E_lidar = Encoder_lidar
        self.E_thr = Encoder_thr

        self.D_img = D(block_channel,num_features=block_channel[3])
        self.D_lidar = D(block_channel,num_features=block_channel[3])
        self.D_thr = D(block_channel,num_features=block_channel[3])
        self.R1 = R_um(num_features=block_channel[0]//2)
        self.R2 = R_um(num_features=block_channel[0]//2)

    def forward(self, rgb,lidar,thermal):
        x1, x2, x3, x4 = self.E_img(rgb)
        x1_, x2_, x3_, x4_ = self.E_lidar(lidar)
        x1__, x2__, x3__, x4__ = self.E_thr(thermal)

        x_d4_img, _,_,_ = self.D_img(x1, x2, x3, x4)
        x_d4_lidar, _,_,_ = self.D_lidar(x1_, x2_, x3_, x4_)
        x_d4_thr, _,_,_ = self.D_thr(x1__, x2__, x3__, x4__)
        out = self.R1(x_d4_img + x_d4_lidar + x_d4_thr)
        uncetainty = self.R2(x_d4_img + x_d4_lidar + x_d4_thr)

        return out, uncetainty



class model(nn.Module): #model for single sensor
    def __init__(self, Encoder, block_channel):
        super(model, self).__init__()
        self.E = Encoder
        self.D = D(block_channel,num_features=block_channel[3])
        self.R1 = R_um(num_features=block_channel[0]//2)
        self.R2 = R_um(num_features=block_channel[0]//2)

    def forward(self, input):
        x_e1, x_e2, x_e3, x_e4 = self.E(input)
        x_d4, xd3, x_d2, x_d1 = self.D(x_e1, x_e2, x_e3, x_e4)
        out = self.R1(x_d4)
        uncetainty = self.R2(x_d4)

        return out, uncetainty, x_e1, x_e2, x_e3, x_e4



class fusion_2sensors(nn.Module):
    def __init__(self, block_channel):
        super(fusion_2sensors, self).__init__()
        self.D_fusion = fusion_layer(block_channel,num_features=block_channel[3])

        self.R1_fusion = R_um(num_features=block_channel[0]//2)
        self.R2_fusion = R_um(num_features=block_channel[0]//2)

    def forward(self, x1, x2, x3, x4, x1_, x2_, x3_, x4_):
        x_decoder = self.D_fusion(x1, x2, x3, x4, x1_, x2_, x3_, x4_)
        out_fusion = self.R1_fusion(x_decoder)
        uncetainty_fusion = self.R2_fusion(x_decoder)

        return out_fusion, uncetainty_fusion



class fusion_3sensors(nn.Module):
    def __init__(self, block_channel):
        super(fusion_3sensors, self).__init__()

        self.D_fusion_all = fusion_layer2(block_channel,num_features=block_channel[3])
        self.R1_fusion_all = R_um(num_features=block_channel[0]//2)
        self.R2_fusion_all = R_um(num_features=block_channel[0]//2)

    def forward(self, x1, x2, x3, x4, x1_, x2_, x3_, x4_, x1__, x2__, x3__, x4__):
        x_decoder_all = self.D_fusion_all(x1, x2, x3, x4, x1_, x2_, x3_, x4_,  x1__, x2__, x3__, x4__)
        out_all = self.R1_fusion_all(x_decoder_all)
        uncetainty_all = self.R2_fusion_all(x_decoder_all)

        return out_all, uncetainty_all 



