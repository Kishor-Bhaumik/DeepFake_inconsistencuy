import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import random, natsort
import cv2, os
import numpy as np
import pdb
import glob

rootdir='/home/agency/xai/home/data/DeepFake/'
bs= 2
frame_size = 80
img_per_frame = 8
im_size =224

def get_paths(rootdir,part):
    files=glob.glob(rootdir+part) 
    aList=natsort.natsorted(files)
    label = 0 if "low_fake" in part else 1
    
    gap = frame_size
    final = []
    for g in range(0, len(aList)-gap+1, gap):
        all_images = aList[g:g+gap]

        for r in range(0,len(all_images)-img_per_frame+1, img_per_frame):
            taken_images = all_images[r:r+img_per_frame]
            final.append(taken_images+[label])
    return final


def real_fake(fake_path,real_path):
    
    all_path= real_path+fake_path
    random.shuffle(all_path)
    return all_path



train_fake_path= get_paths(rootdir, "train/low_fake/*")
train_real_path= get_paths(rootdir, "train/low_real/*")
test_fake_path= get_paths(rootdir, "test/low_fake/*")
test_real_path= get_paths(rootdir, "test/low_real/*")
val_fake_path= get_paths(rootdir, "val/low_fake/*")
val_real_path= get_paths(rootdir, "val/low_real/*")


train_data_paths= real_fake(train_fake_path,train_real_path)
test_data_paths= real_fake(test_fake_path, test_real_path)
val_data_paths= real_fake(val_fake_path,val_real_path)


class DeepFakeDataset(Dataset):
    def __init__(self, path, transform):
        self.images_path = path
        self.T =transform 
        
    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, idx):
        
        data_paths = self.images_path[idx]
        image_paths = data_paths[:-1]
        label =data_paths[-1]     
        outx= [] 
        for img in image_paths:
            image = cv2.imread(img)
            image = self.T(image)
            outx.append(image)
            
        outx= torch.stack(outx,0)
            
        return outx, label


train_transform = transforms.Compose([transforms.ToTensor(),transforms.Resize(im_size),
                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225] )])

val_transform = transforms.Compose([transforms.ToTensor(),transforms.Resize(im_size),
                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225] )])
test_transform = transforms.Compose([transforms.ToTensor(),transforms.Resize(im_size),
                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225] )])

train_dataset = DeepFakeDataset(train_data_paths,train_transform)
test_dataset = DeepFakeDataset(test_data_paths,test_transform)
val_dataset = DeepFakeDataset(val_data_paths,val_transform)

train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, drop_last=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False, drop_last=False, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False, drop_last=True, num_workers=4)


# for data,label in train_loader:
#     print(data.shape)
#     break


