import numpy as np
from PIL import Image 
import os 
import cv2
import torch
from utils import utils

noise_file = "./utils/lf_noise_1.jpg"
shape = (256, 640)

##########code in this file is partly taken from https://github.com/brunolinux/TV-DIP/blob/master/create_noisy_data.py
def create_low_frequency_noise(file_path, shape, max_strength = 0.5):
    noise = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    noise = cv2.resize(noise, dsize=(shape[1], shape[0]), interpolation=cv2.INTER_AREA)
    noise = np.array(noise, dtype=np.float32)
    noise = noise / 256 * max_strength
    return noise

def create_column_strip(random_generator: np.random.Generator, shape, sigma_white, sigma_strip):
    white_noise = random_generator.random(shape) * sigma_white
    strip_column = random_generator.random((1, shape[1])) * sigma_strip
    strip = np.repeat(strip_column, shape[0], axis=0)
    noise = strip + white_noise
    return noise



def create_image_noise(noise_type, sigma_strip):
    rng = np.random.default_rng(1998) # 2022
    sigma_white = 0.1
    lf_max_strength = 0.4

    if noise_type == "lpn":
        noise = create_low_frequency_noise(noise_file, shape, lf_max_strength)
    elif noise_type == "fpn":
        noise = create_column_strip(rng, shape, sigma_white, sigma_strip)
    elif noise_type == "combined":
        noise_lf = create_low_frequency_noise(noise_file, shape, lf_max_strength)
        noise_fpn = create_column_strip(rng, shape, sigma_white, sigma_strip)
        noise = noise_lf + noise_fpn
    else:
        print("noise type {} is not supported".format(noise_type))
        return

    return noise

def generate_noise(thr, sigma_strip=0.5):
    if(sigma_strip>0):
        ori_thr = utils.denormalize(thr)
        noise = create_image_noise("combined",sigma_strip)
        noise = torch.from_numpy(noise).cuda().unsqueeze(0).unsqueeze(0).float() 

        thr = torch.cat((noise*0.299, noise*0.587, noise*0.114),1) + ori_thr
        thr = utils.normalize(thr).cuda()
    return thr