import numpy as np
import torch, random
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from dataloader import build
from tqdm import tqdm
from binning_fn_torch import canonical_ECE

def save_on_master(*args, **kwargs):
    torch.save(*args, **kwargs)

def get_wmat(size):
    matrix = torch.full((size, size), 0.5)
    indices = torch.arange(size)
    matrix[indices, indices] = 1.0
    return matrix

class LDACE_Loss(nn.Module):
    def __init__(self, num_classes):
        super(LDACE_Loss, self).__init__()
        self.deno = (num_classes * (num_classes+1)) / 2
        self.wm = get_wmat(num_classes) 

    def forward(self, y_pred, y_true):
        y_pred = y_pred.sigmoid()
        y_pred = torch.clamp(y_pred, min=1e-7, max=1-1e-7)
        y_pred = y_pred @ y_pred.transpose(1, 2)
         
        y_true = y_true.unsqueeze(2) @ y_true.unsqueeze(1)
        wgts = self.wm.unsqueeze(0).repeat(y_pred.shape[0],1,1)       
        loss = - (y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred))
        loss = loss * wgts.to(loss.device)
        return torch.sum(loss)/(loss.shape[0]*self.deno)

class CCL_Loss(nn.Module):
    def __init__(self, num_labels, num_bins, num_classes):
        super(CCL_Loss,self).__init__()
        self.num_labels= num_labels
        self.num_bins= num_bins
        
    def forward(self, y_pred, labels):
        y_pred = y_pred.sigmoid()
        y_pred = torch.clamp(y_pred, min=1e-7, max=1-1e-7)
        y_pred = y_pred @ y_pred.transpose(1, 2)
        
        indices = torch.arange(y_pred.shape[-1])
        y_pred = y_pred[:, indices, indices] 
        
        canonical_ECE_loss= canonical_ECE(labels=labels.transpose(1,0), predictions=y_pred.transpose(1,0), num_bins= self.num_bins)
        return canonical_ECE_loss
        
# define configs and create config
config = {
    # optimization configs
    'seed': 42,
    'epoch': 100,  
    'num_features':1,
    'lambda': 100,
    'max_norm': 0.1,
    'batch_size': 32,
    'eval_batch_size': 32,
    'test_batch_size': 1,

    # dataset configs
    'dataset': 'ChestMNIST',
    'num_classes': 14,
}

# fix the seed for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

train_transform = transforms.Compose([
                      transforms.RandomHorizontalFlip(),
                      transforms.RandomVerticalFlip(),
                      transforms.RandomRotation(degrees=(-10, 10)),
                      transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                      transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
                      transforms.ToTensor()
                  ])
eval_transform = transforms.Compose([transforms.Resize((224,224)),
                                      transforms.ToTensor()
                 ])                  
                 
train_dataset = build('train', config, train_transform)
val_dataset = build('val', config, eval_transform)

# Create DataLoader instances for training and testing
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config['eval_batch_size'], shuffle=False)

# Load ResNet-50 model
# model = models.resnet50(pretrained=False)
# model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
# model.fc = nn.Linear(2048, config['num_classes']*config['num_features'])

# Load ViT b 32 model
model = models.vit_b_32()
model.conv_proj = nn.Conv2d(1, 768, kernel_size=(32, 32), stride=(32, 32))
model.heads = nn.Sequential(nn.Linear(in_features=768, out_features=config['num_classes']*config['num_features'], bias=True))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
criterion = LDACE_Loss(config['num_classes']) 
criterion_ece = CCL_Loss(config['batch_size'], torch.tensor(1), config['num_classes'])
best_val_loss = 9999
for epoch in range(config['epoch']):
    print('Epoch:',epoch,'/',config['epoch'])
    model.train()
    tr_loss, tr_ecel, tr_dcl, val_loss, val_ecel, val_dcl = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    for inputs, labels in tqdm(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        bs = inputs.shape[0]
        
        optimizer.zero_grad()
        outputs = model(inputs.float())
        outputs_rs = outputs.view(bs, config['num_classes'], config['num_features'])
        
        ccl_loss = criterion_ece(outputs_rs, labels)
        ldace_loss = criterion(outputs_rs, labels) 
        loss = ldace_loss + ccl_loss 
        
        assert not torch.isnan(loss)
        tr_ecel += ccl_loss.item()
        tr_dcl += ldace_loss.item()
        tr_loss += loss.item()
        loss.backward()
        if config['max_norm'] > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_norm'])
        optimizer.step()
       
    # Validation loop (optional)
    model.eval()
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            bs = inputs.shape[0]
            
            outputs = model(inputs.float())
            outputs_rs = outputs.view(bs, config['num_classes'], config['num_features'])
            
            ccl_loss = criterion_ece(outputs_rs, labels)
            ldace_loss = criterion(outputs_rs, labels) 
            loss = ldace_loss + ccl_loss
            val_ecel += ccl_loss.item()
            val_dcl += ldace_loss.item()  
            val_loss += loss.item() 
    
    val_loss = val_loss / len(val_loader)  
    
    print(f"Epoch {epoch+1}/{config['epoch']}, Loss_train: {tr_loss/len(train_loader)}, Loss_val: {val_loss}, LDACE_loss_train: {tr_dcl/len(train_loader)}, LDACE_loss_val: {val_dcl/len(val_loader)}, CCL_loss_train: {tr_ecel/len(train_loader)}, CCL_loss_val: {val_ecel/len(val_loader)}")
    
    save_on_master({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'args': config,
        'val_acc': val_loss,
    }, 'saved_data/latest_model.pth')       
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss    
        save_on_master({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'args': config,
            'val_acc': val_loss,
        }, 'saved_data/best_model.pth')        

    

