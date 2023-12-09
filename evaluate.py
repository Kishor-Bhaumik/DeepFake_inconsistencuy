import torch
import torch.nn.functional as F
import torch.nn as nn
from dataloader_cons import train_loader, test_loader, val_loader
from model.vip import vip_tiny, vip_mobile, vip_small, vip_base
import torch.optim as optim
from tqdm import tqdm 
import utils
import torchmetrics
from sklearn.metrics import roc_auc_score

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
resume = False

class df_final(nn.Module):
    def __init__(self):
        super(df_final, self).__init__()
        self.object_numbers = 128
        self.frames = 8
        self.object_vector = 1024
        self.object_model = vip_base()
        self.intra_linear = nn.Linear(self.frames*self.object_numbers*self.object_numbers, 128)
        self.inter_linear = nn.Linear(self.object_numbers*self.frames*self.frames, 128)
        self.final = nn.Linear(128*2,2)
    def forward(self,x, device):
        B = x.shape[0]
        fr = inp.shape[1]
        
        reps = [] #torch.empty(B,fr,self.object_numbers,self.object_vector).to(device)
        for i in range(B):
            _, out_rep,_ = self.object_model(x[i])
            reps.append(out_rep)
        
        reps = torch.stack(reps, 0)
        x = F.gelu(reps)
        x_norm = F.normalize(x, dim=-1)

        weights1 = torch.matmul(x_norm, x_norm.transpose(-1,-2)) #similarity matrix
        # print(weights1.shape)
        weights1 = F.softmax(weights1, dim=-1)


        x_norm = x_norm.transpose(1,2)
        weights2 = torch.matmul(x_norm, x_norm.transpose(-1,-2)) #similarity matrix
        # print(weights2.shape)
        weights2 = F.softmax(weights2, dim=-1)


        intra_frame = F.relu(self.intra_linear(weights1.view(B, -1)))
        inter_frame = F.relu(self.inter_linear(weights2.view(B, -1)))



        x = torch.cat([intra_frame, inter_frame], dim=-1)
        x = x.view(B, -1)
        x = self.final(x)
        return x

model = df_final()
model.to(device)



from timm.optim.optim_factory import create_optimizer
from timm.scheduler import create_scheduler


from argu import args
optimizer = create_optimizer(args, model)

lr_scheduler, num_epochs = create_scheduler(args, optimizer)

criterion = nn.CrossEntropyLoss()

EPOCH = 0

best_val_acc = 0

if resume == True:
    checkpoint = torch.load('best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    EPOCH = checkpoint['epoch']
    best_val_acc = checkpoint['best_acc']


accuracy = torchmetrics.Accuracy().to(device)
auc = torchmetrics.AUROC(task = "binary").to(device)

write_log = utils.write_logger()


        

    with torch.no_grad():

        for inp, label in tqdm(test_loader):
            inp, label = inp.to(device), label.to(device)
            B = inp.shape[0]
            fr = inp.shape[1]
            
            out = model(inp, device)
            loss = criterion(out, label.long().detach()) 
            
            out_soft = F.softmax(out, dim=-1)
            
            for i in range(B):
                out_auc.append(out_soft[i][1].tolist())
                label_auc.append(label[i].tolist())
            
        
            val_accuracy = accuracy(out, label)
            val_loss.update(loss)
            val_acc.update(val_accuracy)
            

    print("Epoch ", epoch +1, "Done ")
    # print("AUC", roc_auc_score(label_auc, out_auc))
    auc_score_final = auc(torch.tensor(out_auc), torch.tensor(label_auc)).tolist()
    
    # print("AUC2", auc(torch.tensor(out_auc), torch.tensor(label_auc)))
    # print("AUC", sum(auc_val) / len(auc_val) )

    if val_acc.avg.detach().cpu().item() >= best_val_acc:
        best_val_acc = val_acc.avg.detach().cpu()
    
    if auc_score_final > best_auc:
        best_auc = auc_score_final
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc': best_val_acc
            }, 'best_model.pth')
    else:
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc': best_val_acc
            }, 'last_model.pth')

    logs = {'epoch': epoch+1, 'Train Loss':train_loss.avg.detach().cpu().item(), 'Train Acc': train_acc.avg.detach().cpu().item(), 
            'Validaton Loss': val_loss.avg.detach().cpu().item(), "Validation Acc": val_acc.avg.detach().cpu().item(),
            "AUC": auc_score_final, "BEST AUC": best_auc}

    write_log.write(**logs)


    lr_scheduler.step(start_epoch)
