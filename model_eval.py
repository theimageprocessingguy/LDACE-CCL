import numpy as np
import torch, random
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from dataloader import build
from sklearn.metrics import hamming_loss

def save_on_master(*args, **kwargs):
    torch.save(*args, **kwargs)

# define configs and create config
config = {
    # optimization configs
    'seed': 42,
    'epoch': 100, 
    'num_features':1,
    'batch_size': 32,
    'eval_batch_size': 32,
    'test_batch_size': 1,

    # dataset configs
    'dataset': 'ChestMNIST', 
    'num_classes': 14, 
    
    # saved model
    'checkpoint_path' : 'saved_data/best_model.pth',
}

# fix the seed for reproducibility
seed = config['seed']
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

eval_transform = transforms.Compose([transforms.Resize((224, 224)),
                                      transforms.ToTensor()
                 ])                  

val_dataset = build('val', config, eval_transform)                 
test_dataset = build('test', config, eval_transform)

# Create DataLoader instances for training and testing
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Load pre-trained ResNet-50 model
# model = models.resnet50(pretrained=False)
# model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
# model.fc = nn.Linear(2048, config['num_classes'])

# Load ViT b 32 model
model = models.vit_b_32()
model.conv_proj = nn.Conv2d(1, 768, kernel_size=(32, 32), stride=(32, 32))
model.heads = nn.Sequential(nn.Linear(in_features=768, out_features=config['num_classes']*config['num_features'], bias=True))

model.to(device)

checkpoint = torch.load(config['checkpoint_path'])
model.load_state_dict(checkpoint['model'])
model.eval()

hl, idx, ct = 0, 0, 0
pred_vals, gt_vals = [], []
nzros = []
with torch.no_grad():
    for inputs, labels in val_loader:
        idx += 1
        print(idx, '/', len(val_loader), '>')
        inputs, labels = inputs.to(device), labels.to(device)
        bs = inputs.shape[0]
            
        outputs = model(inputs.float())
        outputs_rs = outputs.view(bs, config['num_classes'], config['num_features'])
        outputs_rs = outputs_rs.sigmoid()
        y_pred = torch.clamp(outputs_rs, min=1e-7, max=1-1e-7)
        y_pred = y_pred @ y_pred.transpose(1, 2)
           
        indices = torch.arange(y_pred.shape[-1])
        outputs_f = y_pred[:, indices, indices] 
        
        pred = torch.max(outputs.softmax(-1), dim=-1)
        hl += (pred.indices == labels)
        pred_vals.append(outputs_f.squeeze().cpu().data.numpy())
        gt_vals.append(labels.squeeze().cpu().data.numpy())
        
acc = hl/len(val_loader)
print('Accuracy:',acc, 'Zeros:', ct)
print(hl, len(val_loader))
print(np.array(pred_vals).shape, np.array(gt_vals).shape)
np.save('ChestMNIST_raw_42/val_224_pred_s42_cbce_ece_vit.npy', np.array(pred_vals))
np.save('ChestMNIST_raw_42/val_224_gt_s42_cbce_ece_vit.npy', np.array(gt_vals)) 
