from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import random #, natsort
import torch,  cv2, os, pdb, glob, json, natsort
import numpy as np

im_size= 224
BATCH_SIZE=  256
bs=BATCH_SIZE


### this creates the faceSwap_test_simp.json file from faceSwap_test.json file


f = open('../faceSwap_test.json')
data = json.load(f)

L=len(data['train'])

train=[]
for i in range(L):

    lab = data['train'][i][-1]
    ss=data['train'][i][:-1]
    for frames in ss:
        train.append((frames,int(lab)))


test=[]
L=len(data['test'])
for i in range(L):

    lab = data['test'][i][-1]
    ss=data['test'][i][:-1]
    for frames in ss:
        test.append((frames,int(lab)))     


dictionary= {"train": train, "test":test}

with open("faceSwap_test_simp.json", "w") as outfile:
    json.dump(dictionary, outfile)

print("writing finished....")





class DeepFakeDataset(Dataset):
    def __init__(self, path, transform):
        self.images_path = path
        self.T =transform 
        
    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, idx):
        
        data_paths = self.images_path[idx]
        img = data_paths[0]
        label =data_paths[1]     
        image = cv2.imread(img)
        image = self.T(image)
            
        return image, int(label)


train_transform = transforms.Compose([transforms.ToPILImage(),transforms.Resize(im_size), transforms.ToTensor(),
                                 transforms.Normalize([0.5321, 0.4070, 0.3695] ,[0.2312, 0.1846, 0.1768] )])

val_transform = transforms.Compose([transforms.ToPILImage(),transforms.Resize(im_size), transforms.ToTensor(),
                                 transforms.Normalize([0.5148, 0.3918, 0.3571],[0.2299, 0.1805, 0.1753] )])

test_transform = transforms.Compose([transforms.ToPILImage(),transforms.Resize(im_size), transforms.ToTensor(),
                                 transforms.Normalize([0.5304, 0.3940, 0.3660] , [0.2283, 0.1762, 0.1690] )])

f = open('../faceSwap_test_simp.json')
data = json.load(f)

train_dataset = DeepFakeDataset(data['train'],train_transform)
test_dataset = DeepFakeDataset(data['test'],test_transform)
#val_dataset = DeepFakeDataset(data['val'],val_transform)

train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, drop_last=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False, drop_last=False, num_workers=4)

# for image,label in test_loader:
#     print(image.shape)
#     break

