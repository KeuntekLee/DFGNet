import torch
import numpy as np
from torch.utils.data import Dataset
import os
from utils.utils import *
import cv2


IMG_HEIGHT = 960
IMG_WIDTH = 1440

class Dynamic_Scenes_Dataset(Dataset):
    def __init__(self, root_dir, transform=None, is_training=True, crop=True, crop_size=None):
        self.root_dir = root_dir
        self.is_training = is_training
        self.crop = crop
        self.crop_size = crop_size

        # scenes dir
        if not self.is_training:
            self.scenes_dir_extra = os.path.join(self.root_dir, 'Test/EXTRA')
            #print(self.scenes_dir)
            self.scenes_dir_extra_list = os.listdir(self.scenes_dir_extra)
            #print(self.scenes_dir_extra_list)
            self.scenes_dir_paper = os.path.join(self.root_dir, 'Test/PAPER')
            self.scenes_dir_paper_list = os.listdir(self.scenes_dir_paper)
            #print(self.scenes_dir_paper_list)
            #print(self.scenes_dir_list)
        else:
            self.scenes_dir = os.path.join(self.root_dir, 'Training')
            self.scenes_dir_list = os.listdir(self.scenes_dir)

        self.image_list = []
        if not self.is_training:
            for scene in range(len(self.scenes_dir_extra_list)):
                exposure_file_path = os.path.join(self.scenes_dir_extra, self.scenes_dir_extra_list[scene], 'exposure.txt')
                #print(exposure_file_path)
                ldr_file_path = list_all_files_sorted(os.path.join(self.scenes_dir_extra, self.scenes_dir_extra_list[scene]), '.tif')
                #print(ldr_file_path)
                label_path = os.path.join(self.scenes_dir_extra, self.scenes_dir_extra_list[scene])
                self.image_list += [[exposure_file_path, ldr_file_path, label_path]]

            for scene in range(len(self.scenes_dir_paper_list)):
                exposure_file_path = os.path.join(self.scenes_dir_paper, self.scenes_dir_paper_list[scene], 'exposure.txt')
                #print(exposure_file_path)
                ldr_file_path = list_all_files_sorted(os.path.join(self.scenes_dir_paper, self.scenes_dir_paper_list[scene]), '.tif')
                #print(ldr_file_path)
                label_path = os.path.join(self.scenes_dir_paper, self.scenes_dir_paper_list[scene])
                self.image_list += [[exposure_file_path, ldr_file_path, label_path]]
        else:
            for scene in range(len(self.scenes_dir_list)):
                exposure_file_path = os.path.join(self.scenes_dir, self.scenes_dir_list[scene], 'exposure.txt')
                #print(exposure_file_path)
                ldr_file_path = list_all_files_sorted(os.path.join(self.scenes_dir, self.scenes_dir_list[scene]), '.tif')
                #print(ldr_file_path)
                label_path = os.path.join(self.scenes_dir, self.scenes_dir_list[scene])
                self.image_list += [[exposure_file_path, ldr_file_path, label_path]]
        #print(ldr_file_path)
        self.expoTimes_list = []
        self.ldr_images_list = []
        self.label_list = []
        for i in self.image_list:
            self.expoTimes_list.append(ReadExpoTimes(i[0]))
            img = ReadImages(i[1])
            #print(img.shape)
            self.ldr_images_list.append(img)
            '''
            if not self.is_training:
                self.ldr_images_list.append(ReadImages(i[1])[116:884,110:1390,:])
            else:
                self.ldr_images_list.append(ReadImages(i[1]))
            '''
            self.label_list.append(ReadLabel(i[2]))
        self.transform=transform
    def __getitem__(self, index):
        # Read exposure times in one scene
        #expoTimes = ReadExpoTimes(self.image_list[index][0])
        # Read LDR image in one scene
        #ldr_images = ReadImages(self.image_list[index][1])
        # Read HDR label
        #label = ReadLabel(self.image_list[index][2])
        # ldr images process
        expoTimes = self.expoTimes_list[index]
        ldr_images = self.ldr_images_list[index]
        label = self.label_list[index]
        #print(ldr_images[0].shape)

        pre_img0 = ldr_images[0]
        pre_img1 = ldr_images[1]
        pre_img2 = ldr_images[2]
        pre_img0 = pre_img0[:,:,::-1]
        pre_img1 = pre_img1[:,:,::-1]
        pre_img2 = pre_img2[:,:,::-1]
        pre_img0 = cv2.resize(pre_img0, (IMG_WIDTH, IMG_HEIGHT))
        pre_img1 = cv2.resize(pre_img1, (IMG_WIDTH, IMG_HEIGHT))
        pre_img2 = cv2.resize(pre_img2, (IMG_WIDTH, IMG_HEIGHT))
        #label = label
        label = label[:,:,::-1]
        label = cv2.resize(label, (IMG_WIDTH, IMG_HEIGHT))
        #pre_img0 = cv2.cvtColor(pre_img0,cv2.COLOR_BGR2RGB)
        #pre_img1 = cv2.cvtColor(pre_img1,cv2.COLOR_BGR2RGB)
        #pre_img2 = cv2.cvtColor(pre_img2,cv2.COLOR_BGR2RGB)
        pre_img0_tm = LDR_to_HDR(pre_img0, expoTimes[0], 2.2)
        pre_img1_tm = LDR_to_HDR(pre_img1, expoTimes[1], 2.2)
        pre_img2_tm = LDR_to_HDR(pre_img2, expoTimes[2], 2.2)

        if self.crop:
            
            H, W, _ = ldr_images[0].shape
            x = np.random.randint(0, H - self.crop_size[0] - 1)
            y = np.random.randint(0, W - self.crop_size[1] - 1)

            img0 = pre_img0[x:x + self.crop_size[0], y:y + self.crop_size[1]].astype(np.float32)#.transpose(2, 0, 1)
            img1 = pre_img1[x:x + self.crop_size[0], y:y + self.crop_size[1]].astype(np.float32)#.transpose(2, 0, 1)
            img2 = pre_img2[x:x + self.crop_size[0], y:y + self.crop_size[1]].astype(np.float32)#.transpose(2, 0, 1)
            label = label[x:x + self.crop_size[0], y:y + self.crop_size[1]].astype(np.float32)#.transpose(2, 0, 1)
            img0_tm = pre_img0_tm[x:x + self.crop_size[0], y:y + self.crop_size[1]].astype(np.float32)#.transpose(2, 0, 1)
            img1_tm = pre_img1_tm[x:x + self.crop_size[0], y:y + self.crop_size[1]].astype(np.float32)#.transpose(2, 0, 1)
            img2_tm = pre_img2_tm[x:x + self.crop_size[0], y:y + self.crop_size[1]].astype(np.float32)#.transpose(2, 0, 1)
            
        else:
            img0 = pre_img0.astype(np.float32)#.transpose(2, 0, 1)
            img1 = pre_img1.astype(np.float32)#.transpose(2, 0, 1)
            img2 = pre_img2.astype(np.float32)#.transpose(2, 0, 1)
            label = label.astype(np.float32)#.transpose(2, 0, 1)
            img0_tm = pre_img0_tm.astype(np.float32)
            img1_tm = pre_img1_tm.astype(np.float32)
            img2_tm = pre_img2_tm.astype(np.float32)
        '''
        img0 = torch.from_numpy(img0)
        img1 = torch.from_numpy(img1)
        img2 = torch.from_numpy(img2)
        label = torch.from_numpy(label)
        '''
        if self.transform != None:
            img0=self.transform(img0)
            img1=self.transform(img1)
            img2=self.transform(img2)
            label=self.transform(label)
            img0_tm=self.transform(img0_tm)
            img1_tm=self.transform(img1_tm)
            img2_tm=self.transform(img2_tm)
        else:
            img0 = torch.from_numpy(img0.transpose(2, 0, 1))
            img1 = torch.from_numpy(img1.transpose(2, 0, 1))
            img2 = torch.from_numpy(img2.transpose(2, 0, 1))
            label = torch.from_numpy(label.transpose(2, 0, 1))
            img0_tm=torch.from_numpy(img0_tm.transpose(2, 0, 1))
            img1_tm=torch.from_numpy(img1_tm.transpose(2, 0, 1))
            img2_tm=torch.from_numpy(img2_tm.transpose(2, 0, 1))
            
        sample = {'input0': img0, 'input1': img1, 'input2': img2, 'label': label, 'input0_tm': img0_tm, 'input1_tm': img1_tm, 'input2_tm': img2_tm, 'expo0': expoTimes[0]}

        return sample

    def __len__(self):
        if not self.is_training:
            return len(self.scenes_dir_extra_list+self.scenes_dir_paper_list)
        else:
            return len(self.scenes_dir_list)
        

