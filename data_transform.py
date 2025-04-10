
import torch
import pandas as pd
# from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image, ImageOps
import collections
try:
    import accimage
except ImportError:
    accimage = None
import random
import math
import numbers
import types
import scipy.ndimage as ndimage

import pdb


def _is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


class RandomCrop(object):
    """Crop the given PIL.Image at a random location.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is 0, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively.
    """

    def __init__(self, size, size_depth):
       
        self.size = size
        self.size_depth = size_depth


    def crop(self, img1):
        w, h = img1.size
        tw, th = self.size
        i = random.randint(0, w - tw)


        j = h - th
        img1 = img1.crop((i, j, i + tw, j + th))
      
        return img1

    def __call__(self, sample):

        img, lidar_img_sd, lidar_img_gt, thr, lidar_thr_sd, lidar_thr_gt = sample['img'], sample['lidar_img_sd'], sample['lidar_img_gt'], sample['thr'], sample['lidar_thr_sd'], sample['lidar_thr_gt']


        img = self.crop(img)
        lidar_img_sd = self.crop(lidar_img_sd)
        lidar_img_gt = self.crop(lidar_img_gt)

        thr = self.crop(thr)
        lidar_thr_sd = self.crop(lidar_thr_sd)
        lidar_thr_gt = self.crop(lidar_thr_gt)

        return {'img': img, 'lidar_img_sd': lidar_img_sd, 'lidar_img_gt': lidar_img_gt, 'thr': thr, 'lidar_thr_sd': lidar_thr_sd, 'lidar_thr_gt': lidar_thr_gt}


class RandomHorizontalFlip(object):

    def __call__(self, sample):
        img, lidar_img_sd, lidar_img_gt, thr, lidar_thr_sd, lidar_thr_gt = sample['img'], sample['lidar_img_sd'], sample['lidar_img_gt'], sample['thr'], sample['lidar_thr_sd'], sample['lidar_thr_gt']

        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            lidar_img_sd = lidar_img_sd.transpose(Image.FLIP_LEFT_RIGHT)
            lidar_img_gt = lidar_img_gt.transpose(Image.FLIP_LEFT_RIGHT)

            thr = thr.transpose(Image.FLIP_LEFT_RIGHT)
            lidar_thr_sd = lidar_img_sd.transpose(Image.FLIP_LEFT_RIGHT)
            lidar_thr_gt = lidar_img_gt.transpose(Image.FLIP_LEFT_RIGHT)

        return {'img': img, 'lidar_img_sd': lidar_img_sd, 'lidar_img_gt': lidar_img_gt, 'thr': thr, 'lidar_thr_sd': lidar_thr_sd, 'lidar_thr_gt': lidar_thr_gt}


class ToTensor(object):


    def __call__(self, sample):
        img, lidar_img_sd, lidar_img_gt, thr, lidar_thr_sd, lidar_thr_gt = sample['img'], sample['lidar_img_sd'], sample['lidar_img_gt'], sample['thr'], sample['lidar_thr_sd'], sample['lidar_thr_gt']

        img = self.to_tensor(img)
        lidar_img_sd = self.to_tensor(lidar_img_sd).float().div_(256.0) 
        lidar_img_gt = self.to_tensor(lidar_img_gt).float().div_(256.0)  

        thr = self.to_tensor(thr).float()
        thr = self.Raw2Celsius(thr).div_(255.0) 
        lidar_thr_sd = self.to_tensor(lidar_thr_sd).float().div_(256.0)
        lidar_thr_gt = self.to_tensor(lidar_thr_gt).float().div_(256.0) 
        # depth = torch.true_divide(depth, 256)
        # depth_gt = self.to_tensor(depth_gt) 
        # depth_gt = torch.true_divide(depth_gt, 256)
       
        return {'img': img, 'lidar_img_sd': lidar_img_sd, 'lidar_img_gt': lidar_img_gt, 'thr': thr, 'lidar_thr_sd': lidar_thr_sd, 'lidar_thr_gt': lidar_thr_gt}
       

    def to_tensor(self, pic):
        if not(_is_pil_image(pic) or _is_numpy_image(pic)):
            raise TypeError(
                'pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img.float()


        if accimage is not None and isinstance(pic, accimage.Image):
            nppic = np.zeros([pic.channels, pic.height, pic.width], dtype=np.float32)
            pic.copyto(nppic)
            return torch.from_numpy(nppic)

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)
        # put it from HWC to CHW format
        # yikes, this transpose takes 80% of the loading time/CPU
        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float().div(255)
        else:
            return img


    # Raw thermal radiation value to tempearture 
    def Raw2Celsius(self, Raw):
        R = 380747
        B = 1428
        F = 1
        O = -88.539
        Celsius = B / torch.log(R / (Raw - O) + F) - 273.15;
        return Celsius

class Lighting(object):

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, sample):
        img, lidar_img_sd, lidar_img_gt, thr, lidar_thr_sd, lidar_thr_gt = sample['img'], sample['lidar_img_sd'], sample['lidar_img_gt'], sample['thr'], sample['lidar_thr_sd'], sample['lidar_thr_gt']
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone()\
            .mul(alpha.view(1, 3).expand(3, 3))\
            .mul(self.eigval.view(1, 3).expand(3, 3))\
            .sum(1).squeeze()

        img = img.add(rgb.view(3, 1, 1).expand_as(img))

        return {'img': img, 'lidar_img_sd': lidar_img_sd, 'lidar_img_gt': lidar_img_gt, 'thr': thr, 'lidar_thr_sd': lidar_thr_sd, 'lidar_thr_gt': lidar_thr_gt}

class Grayscale(object):

    def __call__(self, img):
        gs = img.clone()
        gs[0].mul_(0.299).add_(0.587, gs[1]).add_(0.114, gs[2])
        gs[1].copy_(gs[0])
        gs[2].copy_(gs[0])
        return gs


class Saturation(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = Grayscale()(img)
        alpha = random.uniform(-self.var, self.var)
        return img.lerp(gs, alpha)



class Brightness(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = img.new().resize_as_(img).zero_()
        alpha = random.uniform(-self.var, self.var)
        # print('brightness:',img,gs,alpha,img.lerp(gs, alpha))
        return img.lerp(gs, alpha)


class Contrast(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = Grayscale()(img)
        gs.fill_(gs.mean())

        alpha = random.uniform(-self.var, self.var)
        return img.lerp(gs, alpha)


class RandomOrder(object):
    """ Composes several transforms together in random order.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        img, lidar_img_sd, lidar_img_gt, thr, lidar_thr_sd, lidar_thr_gt = sample['img'], sample['lidar_img_sd'], sample['lidar_img_gt'], sample['thr'], sample['lidar_thr_sd'], sample['lidar_thr_gt']
        
        thr = torch.cat((thr,thr,thr),0)
        if self.transforms is None:
            return {'img': img, 'lidar_img_sd': lidar_img_sd, 'lidar_img_gt': lidar_img_gt, 'thr': thr, 'lidar_thr_sd': lidar_thr_sd, 'lidar_thr_gt': lidar_thr_gt}
        order = torch.randperm(len(self.transforms))
        
        for i in order:
            img = self.transforms[i](img)
            thr = self.transforms[i](thr)

        return {'img': img, 'lidar_img_sd': lidar_img_sd, 'lidar_img_gt': lidar_img_gt, 'thr': thr, 'lidar_thr_sd': lidar_thr_sd, 'lidar_thr_gt': lidar_thr_gt}


class ColorJitter(RandomOrder):

    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4):
        self.transforms = []
        if brightness != 0:
            self.transforms.append(Brightness(brightness))
        if contrast != 0:
            self.transforms.append(Contrast(contrast))
        if saturation != 0:
            self.transforms.append(Saturation(saturation))


class Normalize(object):
    """Normalize an tensor image with mean and standard deviation.
    Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std
    Args:
        mean (sequence): Sequence of means for R, G, B channels respecitvely.
        std (sequence): Sequence of standard deviations for R, G, B channels
            respecitvely.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        img, lidar_img_sd, lidar_img_gt, thr, lidar_thr_sd, lidar_thr_gt = sample['img'], sample['lidar_img_sd'], sample['lidar_img_gt'], sample['thr'], sample['lidar_thr_sd'], sample['lidar_thr_gt']

        img = self.normalize(img, self.mean, self.std)
        thr = self.normalize(thr, self.mean, self.std)

        return {'img': img, 'lidar_img_sd': lidar_img_sd, 'lidar_img_gt': lidar_img_gt, 'thr': thr, 'lidar_thr_sd': lidar_thr_sd, 'lidar_thr_gt': lidar_thr_gt}

    def normalize(self, tensor, mean, std):
        for t, m, s in zip(tensor, mean, std):
            t.sub_(m).div_(s)
        return tensor
