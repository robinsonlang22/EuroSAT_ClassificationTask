import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import get_dataloaders
from model import SatelliteResNet

# --------------------------------------------------------
# 1. Setup Utilities
# --------------------------------------------------------
def set_seed(seed=369):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seed set to {seed}")

# --------------------------------------------------------
# 2. Train Loop
# --------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, device, epoch_idx):
    model.train()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    loop = tqdm(loader, desc=f"Train Ep {epoch_idx}", leave=False)
    
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)
        
        # Forward
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Stats
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        loop.set_postfix(loss=loss.item())
        
    return running_loss / total, correct / total

# --------------------------------------------------------
# 3. Validation Loop
# --------------------------------------------------------
def validate(model, loader, criterion, device, epoch_idx):
    model.eval()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        loop = tqdm(loader, desc=f"Val Ep {epoch_idx}", leave=False)
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return running_loss / total, correct / total

# --------------------------------------------------------
# 4. Main Function
# --------------------------------------------------------
def main(args):
    # --- Setup ---
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(args.save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=f"runs/{args.exp_name}")

    # --- Data Loading ---
    print("Loading data...")
    
    train_loader, val_loader, _ = get_dataloaders(
        data_root=args.data_root,
        split_dir=args.split_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # --- Model Initialization ---
    print("Initializing model (Custom ResNet18 for 64x64)...")
    model = SatelliteResNet(num_classes=10).to(device)

    # --- Optimizer & Scheduler ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Decay LR by factor of 0.1 every 7 epochs
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # --- Training Loop ---
    best_val_loss = float('inf')
    print(f"🔥 Starting training for {args.epochs} epochs...")
    
    for epoch in range(1, args.epochs + 1):
        # Train & Validate
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_acc = validate(model, val_loader, criterion, device, epoch)
        
        # Step Scheduler
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        # Console Log
        print(f"Epoch [{epoch}/{args.epochs}] "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
              f"LR: {current_lr:.1e}")
        
        # TensorBoard Log
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('Learning Rate', current_lr, epoch)

        # Checkpointing (Best Model Only)
        if val_loss < best_val_loss:
            print(f"New best result ({best_val_loss:.4f} -> {val_loss:.4f}). Saving...")
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.save_dir, "best_model.pth"))
    
    writer.close()
    print(f"Training finished. Best model at: {os.path.join(args.save_dir, 'best_model.pth')}")

# --------------------------------------------------------
# 5. Argument Parsing
# --------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ResNet on EuroSAT")
    
    # Paths
    parser.add_argument('--data_root', type=str, default=".", help='Root dir containing the EuroSAT folder')
    parser.add_argument('--split_dir', type=str, default="train_val_test", help='Dir containing .txt splits')
    parser.add_argument('--save_dir', type=str, default="checkpoints", help='Save dir for .pth files')
    parser.add_argument('--exp_name', type=str, default="resnet18_run1", help='TensorBoard experiment name')
    
    # Hyperparameters
    parser.add_argument('--epochs', type=int, default=20, help='Total training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='Weight decay for AdamW')
    
    # System
    parser.add_argument('--seed', type=int, default=369, help='Random seed for reproducibility')
    parser.add_argument('--num_workers', type=int, default=2, help='DataLoader workers (0 for Win, 4 for Linux)')

    args = parser.parse_args()
    
    main(args)