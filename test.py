
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#import tensorflow.compat.v1 as tf
#import tensorflow as tf
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset.dataset import Dynamic_Scenes_Dataset
from DFGNet_model import DFGNet
from utils.utils import *
from disentanglenet import Exposure_Encoder, Spaital_Encoder

data_root = '/data1/keuntek/HDR_KALANTRI'

test_dataset = Dynamic_Scenes_Dataset(root_dir=data_root, is_training=False, transform=None, crop=False)
test_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
ee = Exposure_Encoder(nFeat=64,outdim=1024)
es = Spaital_Encoder(nFeat=64)
#ee.load_state_dict(torch.load("./models/ee_35000.pth"))
#es.load_state_dict(torch.load("./models/es_35000.pth"))
model.load_state_dict(torch.load("./models/DFGNet.pth"))
ee.load_state_dict(torch.load("./models/ee.pth"))
es.load_state_dict(torch.load("./models/es.pth"))
#model = HDR_UNET_DEFC(6, 5, 64, 32).cuda()
model = DFGNet(6, 64)
'''
model = torch.nn.DataParallel(model)
ee = torch.nn.DataParallel(ee)
'''
model.cuda()
ee.cuda()
es.cuda()

model.eval()
es.eval()
ee.eval()
def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook
es.Inconv.register_forward_hook(get_activation('feat_64'))
es.model[0].register_forward_hook(get_activation('feat_128'))
es.model[1].register_forward_hook(get_activation('feat_256'))

requires_grad(ee, False)
requires_grad(es, False)
requires_grad(model, False)


    
total_psnr=0.

for i in val_loader:
    #print(i['input0'].shape)
    ldr1 = i['input0'].cuda()
    ldr2 = i['input1'].cuda()
    ldr3 = i['input2'].cuda()
    ldr1_tm = i['input0_tm'].cuda()
    ldr2_tm = i['input1_tm'].cuda()
    ldr3_tm = i['input2_tm'].cuda()
    gt = i['label'].cuda()

    batch_size,ch,H,W = ldr1.shape
    #print(H,W)
    ldr1 = ldr1*2 -1
    ldr2 = ldr2*2 -1
    ldr3 = ldr3*2 -1

    ldr1_tm = ldr1_tm*2 -1
    ldr2_tm = ldr2_tm*2 -1
    ldr3_tm = ldr3_tm*2 -1

    ldr1_cat = torch.cat([ldr1,ldr1_tm], dim=1)
    ldr2_cat = torch.cat([ldr2,ldr2_tm], dim=1)
    ldr3_cat = torch.cat([ldr3,ldr3_tm], dim=1)

    lumi1 = ee(ldr1)
    lumi2 = ee(ldr2)
    lumi3 = ee(ldr3)
    activation={}
    _ = es(ldr1).detach()
    ldr1_64 = activation['feat_64']
    ldr1_128 = activation['feat_128']
    ldr1_256 = activation['feat_256']
    activation={}
    _ = es(ldr2).detach()
    ldr2_64 = activation['feat_64']
    ldr2_128 = activation['feat_128']
    ldr2_256 = activation['feat_256']
    activation={}
    _ = es(ldr3).detach()
    ldr3_64 = activation['feat_64']
    ldr3_128 = activation['feat_128']
    ldr3_256 = activation['feat_256']
    #print(interpolated1.shape)
    pred = model(ldr1_cat, ldr2_cat, ldr3_cat, [lumi1,lumi2,lumi3],
                    [ldr1_64, ldr2_64, ldr3_64],
                    [ldr1_128, ldr2_128, ldr3_128],
                    [ldr1_256, ldr2_256, ldr3_256]).detach()
    #pred = model(ldr1_cat, ldr2_cat, ldr3_cat)
    pred = pred[0,:,:,:]#.astype(np.float32)
    pred = (pred+1.)/2.
    pred = range_compressor_tensor(pred)
    #pred = torch.clamp(pred, 0., 1.)
    #gt = (gt+1)/2.
    gt = range_compressor_tensor(gt)
    #loss = criterion(pred, gt)

    #psnr = batch_PSNR(gt, pred, 1.0)
    mse = torch.mean((gt-pred)**2)
    psnr = -10.*math.log10(mse)
    total_psnr+=psnr
    #print(psnr)
if total_psnr/len(val_loader) > best_psnr:
    best_psnr = total_psnr/len(val_loader)
print("Test PSNR: ",total_psnr/len(val_loader))

