import numpy as np
import torch
from utils import utils

def generate_noise_img(img, mean = 0, std = 0.01):
    if std==0:
        return img
    img = utils.denormalize(img)
    noise = torch.randn_like(img) * std + mean
    noisy_image = img + noise

    return utils.normalize(noisy_image) 

def generate_noise_lidar(img, mean = 0, std = 0.01):
    if std==0:
        return img
    std = img.max()*std
    noise = torch.randn_like(img) * std + mean
    noisy_image = img + noise

    return noisy_image
  
