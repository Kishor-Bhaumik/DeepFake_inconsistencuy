# from dataloader_simp import train_loader, test_loader, val_loader
from dataloader_simp import train_loader, test_loader, val_loader
from model.vip import vip_tiny, vip_base
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm 
import numpy as np
import utils
import torchmetrics

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
resume = False 

class dfmodel(nn.Module):
    def __init__(self):
        super(dfmodel, self).__init__()
        self.representation = vip_tiny()
        self.object_vector_length = 128
        self.linear = nn.Linear(self.object_vector_length*self.object_vector_length, 128)
        self.final = nn.Linear(128,2)
    def forward(self,x):
        #shape of x should be [Batch, Channel, H, W]
        B = x.shape[0]
        _, x,_ = self.representation(x)
        x = F.gelu(x)
        x_norm = F.normalize(x, dim=-1)
        weights = torch.matmul(x_norm, x_norm.transpose(-1,-2)) #similarity matrix
        # print(weights.shape)
        weights = F.softmax(weights, dim=-1)
        # x = torch.matmul(weights, x)
        x = weights
        x = x.view(B, -1)
        x = F.relu(self.linear(x))
        x = self.final(x)
        return x

model = dfmodel()
model.to(device) 



from timm.optim.optim_factory import create_optimizer
from timm.scheduler import create_scheduler



from argu import args
optimizer = create_optimizer(args, model)

lr_scheduler, num_epochs = create_scheduler(args, optimizer)


EPOCH = 0

best_val_acc = 0

if resume == True:
    checkpoint = torch.load('last_model_tiny.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    EPOCH = checkpoint['epoch']
    best_val_acc = checkpoint['best_acc']

    


criterion = nn.CrossEntropyLoss()
accuracy = torchmetrics.Accuracy().to(device)

write_log = utils.write_logger()



start_epoch = 0
for epoch in range(EPOCH,num_epochs):
    model.train()

    train_loss= utils.AverageMeter()
    train_acc = utils.AverageMeter()

    for inp, label in tqdm(train_loader):
        inp, label = inp.to(device), label.to(device)
        optimizer.zero_grad()
        out = model(inp)
        loss = criterion(out, label.long().detach()) 

        loss.backward()

        optimizer.step()

        train_accuracy = accuracy(out, label)

        train_loss.update(loss)
        train_acc.update(train_accuracy)
        
        
        

    with torch.no_grad():
        val_acc = utils.AverageMeter()
        val_loss = utils.AverageMeter()
        model.eval()

        for inp, label in tqdm(test_loader):
            inp, label = inp.to(device), label.to(device)
            out = model(inp)
            loss = criterion(out, label.long().detach()) 
            
            val_accuracy = accuracy(out, label)
            val_loss.update(loss)
            val_acc.update(val_accuracy)

    print("Epoch ", epoch +1, "Done ")
    

    if val_acc.avg.detach().cpu().item() >= best_val_acc:
        best_val_acc = val_acc.avg.detach().cpu()
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc': best_val_acc
            }, 'best_model_tiny.pth')
    else:
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc': best_val_acc
            }, 'last_model_tiny.pth')

    logs = {'epoch': epoch+1, 'Train Loss':train_loss.avg.detach().cpu().item(), 'Train Acc': train_acc.avg.detach().cpu().item(), 
            'Validaton Loss': val_loss.avg.detach().cpu().item(), "Validation Acc": val_acc.avg.detach().cpu().item(),
            'Best Validation Acc': best_val_acc}

    write_log.write(**logs)

    lr_scheduler.step(start_epoch)
    



