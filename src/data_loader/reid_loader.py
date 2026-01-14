import os
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import torch

class MarketDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.img_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.jpg')]
        
        # Lấy ID và map sang index
        self.pids = [int(os.path.basename(p).split('_')[0]) for p in self.img_paths]
        unique_pids = sorted(list(set(self.pids)))
        self.pid2label = {pid: i for i, pid in enumerate(unique_pids)}
        self.num_classes = len(unique_pids)

    def __len__(self): return len(self.img_paths)

    def __getitem__(self, idx):
        path = self.img_paths[idx]
        label = self.pid2label[self.pids[idx]]
        try:
            img = Image.open(path).convert('RGB')
            if self.transform: img = self.transform(img)
            return img, label
        except:
            # Trả về ảnh đen nếu lỗi
            return torch.zeros(3, 256, 128), label

def get_dataloaders(cfg):
    # Transform
    train_tfm = transforms.Compose([
        transforms.Resize((cfg['data']['height'], cfg['data']['width'])),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Pad(10),
        transforms.RandomCrop((cfg['data']['height'], cfg['data']['width'])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_tfm = transforms.Compose([
        transforms.Resize((cfg['data']['height'], cfg['data']['width'])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load Full Dataset
    full_dataset = MarketDataset(cfg['data']['root_dir'], transform=train_tfm)
    num_classes = full_dataset.num_classes
    
    # Split Train/Val
    val_size = int(len(full_dataset) * cfg['data']['val_split'])
    train_size = len(full_dataset) - val_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size])
    
    # Áp dụng transform riêng cho Val
    val_set.dataset.transform = val_tfm 

    train_loader = DataLoader(
        train_set, 
        batch_size=cfg['train']['batch_size'], 
        shuffle=True, 
        num_workers=cfg['data']['workers'], 
        pin_memory=True,
        drop_last=True  
    )
    
    val_loader = DataLoader(
        val_set, 
        batch_size=cfg['train']['batch_size'], 
        shuffle=False, 
        num_workers=cfg['data']['workers'], 
        pin_memory=True,
        drop_last=True  
    )
    
    return train_loader, val_loader, num_classes