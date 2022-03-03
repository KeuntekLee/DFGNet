#-*- coding:utf-8 _*-  
""" 
@author: LiuZhen
@license: Apache Licence 
@file: utils.py 
@time: 2020/06/30
@contact: liuzhen.pwd@gmail.com
@site:  
@software: PyCharm 
"""
import numpy as np
import os, glob
import cv2
import imageio
from math import log10
import torch
import torch.nn as nn
import torch.nn.init as init
#from skimage.measure.simple_metrics import compare_psnr
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
imageio.plugins.freeimage.download()


def list_all_files_sorted(folderName, extension=""):
    return sorted(glob.glob(os.path.join(folderName, "*" + extension)))


def ReadExpoTimes(fileName):
    return np.power(2.0, np.loadtxt(fileName))


def ReadImages(fileNames):
    imgs = []
    for imgStr in fileNames:
        if imgStr.split(".")[-1] == "hdr":
            #print("No", imgStr)
            img = cv2.imread(imgStr, -1)
        else:
            #print("Yes", imgStr)
            img = cv2.imread(imgStr)
        #img = cv2.imread(imgStr)

        # equivalent to im2single from Matlab
        #img = img / 2 ** 16
        #img = np.float32(img)
        img = img.astype(np.float32)/255.

        #img.clip(0, 1)

        imgs.append(img)
    return np.array(imgs)


def ReadLabel(fileName):
    #label = imageio.imread(os.path.join(fileName, 'HDRImg.hdr'), 'hdr')
    #label = label[:, :, [2, 1, 0]]  ##cv2
    label = cv2.imread(os.path.join(fileName, 'HDRImg.hdr'), -1)
    label = label.astype(np.float32)
    return label


def LDR_to_HDR(imgs, expo, gamma):
    return np.power(imgs, gamma) / expo


def range_compressor(x):
    return (np.log(1 + 5000 * x)) / np.log(1 + 5000)


def range_compressor_tensor(x):
    const_1 = torch.from_numpy(np.array(1.0)).cuda()
    const_5000 = torch.from_numpy(np.array(5000.0)).cuda()
    return (torch.log(const_1 + const_5000 * x)) / torch.log(const_1 + const_5000)


def set_random_seed(seed):
    """Set random seed for reproduce"""
    # random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

