
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import random, natsort
import glob
import numpy as np
import datetime
import os

def get_mean_std(folder):
 
    means = torch.zeros(3)
    stds = torch.zeros(3)

    _data = datasets.ImageFolder(root = folder,transform = transforms.ToTensor())
    data_len = len(_data)

    for img, label in _data:
        means += torch.mean(img, dim = (1,2))
        stds += torch.std(img, dim = (1,2))

    means /= data_len
    stds /= data_len
    
    return means, stds
    

# val_means , val_stds = get_mean_std('/home/data/DeepFake/val')
# train_means , train_stds = get_mean_std('/home/data/DeepFake/train')
# print(train_means , train_stds)
# print(val_means , val_stds)
# test_means, test_stds = get_mean_std('/home/data/DeepFake/test')
# print(test_means, test_stds)


def random_choose(gap,List, img_per_fram,last_portion=False):
    
    length= len(List)
    new_path=[]
    label = 0 if "low_fake" in List[0] else 1
    #pdb.set_trace()
     
    if not last_portion: 
        for v in range(0, length-img_per_fram +1, img_per_fram ):
            aList = List[v:v+img_per_fram ]
            temp_path =[]
            for g in range(0, len(aList)-gap+1, gap):
                temp_path+= random.choices( aList[g:g+gap], k=1)   
            new_path.append(temp_path+[label])

    if last_portion:
        for g in range(0, length-gap+1, gap):
            new_path+= random.choices( List[g:g+gap], k=1)   
        new_path+=[label]

    return new_path


def path_organize(path,gap,img_per_fram):
    "image will be selected from frames "
    path=natsort.natsorted(path)
    j=0
    new_path=[]
    length= len(path)
    remndr= length % img_per_fram

    if (remndr != 0):
        taken_samples_per_frame = int(img_per_fram / gap)
        upto = length-remndr
        new_path= random_choose(gap,path[:upto],img_per_fram)
        lastportion = path[upto:]

        if remndr> taken_samples_per_frame or remndr == taken_samples_per_frame:

            if (taken_samples_per_frame*2 <= remndr):
                new_gap = int(remndr /taken_samples_per_frame)
                new_path.append( random_choose(new_gap,lastportion,img_per_fram,last_portion=True) )
            else:
                new_path.append(lastportion[:int(taken_samples_per_frame)])

        return new_path
    
    elif (remndr == 0):
        new_path= random_choose(gap,path)
        return new_path
    
    else: print("something is wrong!")



def get_paths(rootdir,part,img_per_fram,segment,skip):
    files=glob.glob(rootdir+part) 
    if segment: 
        path= path_organize(files ,skip,img_per_fram) 
    if not segment:
        path=natsort.natsorted(files)
        length= len(path)
        label= 0 if 'low_fake' in path[0] else 1
        p=[]
        for v in range(0, length-img_per_fram +1, img_per_fram ):
            aList = path[v:v+img_per_fram ]+[label]
            p.append(aList)
        path=p
        #path=path_organize(path, skip)
    return path




def separate(paths,temporal_window):
    
    ite= int(len(paths)/temporal_window)
    
    label = 0 if "low_fake" in paths[0] else 1
    rows =[]
    a,b   = 0,temporal_window
    for v in range(ite):   
        Npath = paths[a:b]+[label]
        #print(len(Npath))
        rows.append(Npath)
        a=b
        b+=temporal_window
    return rows
    

def real_fake(fake_path,real_path):
    
    all_path= real_path+fake_path
    random.shuffle(all_path)
    return all_path

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, length=0):
        self.length = length
        self.reset()

    def reset(self):
        if self.length > 0:
            self.history = []
        else:
            self.count = 0
            self.sum = 0.0
        self.val = 0.0
        self.avg = 0.0

    def update(self, val, num=1):
        if self.length > 0:
            # currently assert num==1 to avoid bad usage, refine when there are some explict requirements
            assert num == 1
            self.history.append(val)
            if len(self.history) > self.length:
                del self.history[0]

            self.val = self.history[-1]
            self.avg = np.mean(self.history)
        else:
            self.val = val
            self.sum += val*num
            self.count += num
            self.avg = self.sum / self.count


class write_logger(object):
    def __init__(self):
        if not os.path.isdir('results'):
            os.mkdir('results')
        now = datetime.datetime.now()
        self.filename_log = 'Results-'+str(now)+'.txt'
    
    def write(self, **kwargs):
        f = open('results/'+self.filename_log, "a")
        for key, value in kwargs.items():
            f.write(str(key) +": " +str(value)+ "\n")
        f.write("\n")
        f.close()

