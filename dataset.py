import os
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# 1. Data Augmentation
def get_transforms(mode='train', input_size=64):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    if mode == 'train':
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

# 2. Dataset
class EuroSATDataset(Dataset):
    def __init__(self, txt_file, root_dir, transform=None):
        self.transform = transform
        self.root_dir = Path(root_dir)
        
        if not os.path.exists(txt_file):
            raise FileNotFoundError(f"File not found: {txt_file}. Please run src/make_splits.py first!")

        with open(txt_file, "r") as f:
            self.img_list = [line.strip() for line in f.readlines()]
            
        self.classes = sorted(list(set([p.split('/')[1] for p in self.img_list])))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

    def __getitem__(self, idx):
        rel_path = self.img_list[idx]
        img_path = self.root_dir / rel_path 
        
        try:
            img = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            print(f"Error loading: {img_path}")
            raise

        cls_name = rel_path.split('/')[1]
        label = self.class_to_idx[cls_name]

        if self.transform:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.img_list)

# 3. Loader
def get_dataloaders(data_root=".", split_dir="train_val_test", batch_size=64, num_workers=2):
    
    split_path = Path(split_dir)

    train_loader = DataLoader(
        EuroSATDataset(split_path / "train.txt", data_root, transform=get_transforms('train')),
        batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    
    val_loader = DataLoader(
        EuroSATDataset(split_path / "val.txt", data_root, transform=get_transforms('eval')),
        batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    
    test_loader = DataLoader(
        EuroSATDataset(split_path / "test.txt", data_root, transform=get_transforms('eval')),
        batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader