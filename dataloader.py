import numpy as np
from PIL import Image

class DataGen:
    def __init__(self, data, transforms=None):
        self.imgs = getattr(data,'imgs')
        self.lbls = getattr(data,'labels').astype(np.float32)
        self.transform = transforms
        
    def __getitem__(self, idx):
        img, lbl = Image.fromarray(self.imgs[idx]), self.lbls[idx]
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img, lbl

    def __len__(self):
        return len(self.imgs)
        
def build(image_set, args, transforms=None):
    if args['dataset'] == 'ChestMNIST':
        from medmnist import ChestMNIST
        data = ChestMNIST(split=image_set, download=True, size=224)
        
    dataset = DataGen(data, transforms)    

    return dataset
