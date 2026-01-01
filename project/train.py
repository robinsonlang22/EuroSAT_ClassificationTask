import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataset import get_dataloaders
from model import SatelliteResNet

try:
    from config import RGB_DATASET_ROOT, SPLITS_ROOT, SEED
except ImportError:
    RGB_DATASET_ROOT = "EuroSAT_RGB"
    SPLITS_ROOT = "splits"
    SEED = 3719704

# 1. Setup Utilities
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seed set to {seed}")

# 2. Train Loop
def train_one_epoch(model, loader, criterion, optimizer, device, epoch_idx):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    loop = tqdm(loader, desc=f"Train Ep {epoch_idx}", leave=False)
    
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        loop.set_postfix(loss=loss.item())
        
    return running_loss / total, correct / total

# 3. Validation Loop
def validate(model, loader, criterion, device, epoch_idx, class_names):
    model.eval()
    running_loss = 0.0
    total_samples = 0
    
    num_classes = len(class_names)
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    
    with torch.no_grad():
        loop = tqdm(loader, desc=f"Val Ep {epoch_idx}", leave=False)
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            total_samples += labels.size(0)
            
            _, predicted = outputs.max(1)
            
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i].item()
                class_correct[label] += c[i].item() if c.dim() > 0 else c.item()
                class_total[label] += 1

    avg_loss = running_loss / total_samples
    total_acc = sum(class_correct) / total_samples
    
    # accuracy per class
    per_class_acc = {}
    for i in range(num_classes):
        acc = class_correct[i] / class_total[i] if class_total[i] > 0 else 0.0
        per_class_acc[class_names[i]] = acc

    return avg_loss, total_acc, per_class_acc

# 4. Main Function
def main(args):
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    save_dir = os.path.join("checkpoints", args.exp_name)
    os.makedirs(save_dir, exist_ok=True)
    
    writer = SummaryWriter(log_dir=f"runs/{args.exp_name}")

    print(f"Loading data... (Augmentation: {args.aug_strength})")
    
    train_loader, val_loader, _ = get_dataloaders(
        data_root=args.data_root,
        split_dir=args.split_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        aug_strength=args.aug_strength
    )

    class_names = val_loader.dataset.classes
    print(f"Classes: {class_names}")

    print("Initializing model...")
    model = SatelliteResNet(num_classes=10).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # using history to plot graph
    history = {name: [] for name in class_names}

    best_val_loss = float('inf')

    print(f"Starting training for {args.epochs} epochs...")
    
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        val_loss, val_acc, per_class_acc = validate(model, val_loader, criterion, device, epoch, class_names)
        
        for cls, acc in per_class_acc.items():
            history[cls].append(acc)
            writer.add_scalar(f'Accuracy_Class/{cls}', acc, epoch)

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Ep {epoch} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        # TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val_total', val_acc, epoch)

        # save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
            print(f"  --> Saved Best Model (Val Loss: {val_loss:.4f})")
    
    writer.close()
    
    # === Per-Class Accuracy ===
    print("Generating Plots...")
    plt.figure(figsize=(12, 8))
    epochs_range = range(1, args.epochs + 1)
    
    for cls in class_names:
        plt.plot(epochs_range, history[cls], label=cls, marker='.')
    
    plt.title(f"Per-Class Validation Accuracy\n(Augmentation: {args.aug_strength})")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.ylim(0.6, 1.05)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path = os.path.join(save_dir, f"val_acc_per_class_{args.aug_strength}.png")
    plt.savefig(plot_path)
    print(f"Plot saved to: {plot_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # config root
    parser.add_argument('--data_root', type=str, default=RGB_DATASET_ROOT)
    parser.add_argument('--split_dir', type=str, default=SPLITS_ROOT)
    
    # experiment
    parser.add_argument('--exp_name', type=str, default="temp_run")
    parser.add_argument('--aug_strength', type=str, default="mild")
    
    # hyper-parameter
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--num_workers', type=int, default=2)

    args = parser.parse_args()
    
    experiments = [
        ("mild",   "mild_run"),
        ("strong", "strong_run")
    ]

    print(f" Plan: Running {len(experiments)} experiments sequentially")

    for strength, exp_name in experiments:
        print(f"\n>>> [STARTING] Experiment: {exp_name} | Augmentation: {strength}")
        
        args.aug_strength = strength
        args.exp_name = exp_name
     
        main(args)
        
        print(f">>> [FINISHED] Experiment: {exp_name} completed.\n")

    print(" All experiments completed successfully!")