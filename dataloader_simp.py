
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

pic_size=224
BATCH_SIZE = 8
rootdir='/home/agency/xai/home/data/DeepFake/'

train_means , train_stds = (0.485, 0.456, 0.406),(0.229, 0.224, 0.225)
val_means , val_stds     = (0.485, 0.456, 0.406),(0.229, 0.224, 0.225)
test_means, test_stds= (0.485, 0.456, 0.406),(0.229, 0.224, 0.225)

train_transforms = transforms.Compose([transforms.ToTensor(),transforms.Resize(pic_size),
                           transforms.Normalize(mean = train_means, 
                                                std = train_stds) ])

val_transforms = transforms.Compose([
                           transforms.ToTensor(),transforms.Resize(pic_size),
                           transforms.Normalize(mean = val_means, 
                                                std = val_stds) ])    

test_transforms = transforms.Compose([
                           transforms.ToTensor(),transforms.Resize(pic_size),
                           transforms.Normalize(mean = test_means, 
                                                std = test_means) ])


train_data = datasets.ImageFolder(root = rootdir+'train', 
                                  transform = train_transforms)
val_data = datasets.ImageFolder(root = rootdir+'val', 
                                 transform = val_transforms)
test_data = datasets.ImageFolder(root = rootdir+'test', 
                                 transform = test_transforms)



train_loader = DataLoader(train_data, 
                                 shuffle = True, 
                                 batch_size = BATCH_SIZE)

val_loader = DataLoader(val_data,  shuffle=False,
                                 batch_size = BATCH_SIZE)

test_loader = DataLoader(test_data, shuffle=True, batch_size = BATCH_SIZE)

