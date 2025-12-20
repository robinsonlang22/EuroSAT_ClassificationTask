import torch
import torch.nn as nn
import torch.optim as optim
import itertools
import pandas as pd

from model import SatelliteResNet
from dataset import get_dataloaders

def tuning_runner(data_root, split_dir, device):
    
    # grid-search space
    search_space = {
        "lr": [1e-3, 5e-4, 1e-4],
        "weight_decay": [1e-4, 1e-3, 1e-2], 
        "batch_size": [64, 128] # argument is recieved by func get_dataloaders
    }

    # groups
    keys, values = zip(*search_space.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"Total configurations: {len(combinations)}")
    
    results = []

    # core loop to find the best model
    for i, config in enumerate(combinations):
        print(f"--- Experiment {i+1}/{len(combinations)}: {config} ---")
    
        train_loader, val_loader, _ = get_dataloaders(
            data_root=data_root,
            split_dir=split_dir,
            batch_size=config['batch_size'],
            num_workers=2
        )

        # reset model in loop
        model = SatelliteResNet(num_classes=10).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=config['lr'], 
            weight_decay=config['weight_decay']
        )

        # 5 Epoch training
        best_val_loss_this_run = float('inf')
        
        for epoch in range(5): 
            # Train
            model.train()
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            # Val
            model.eval()
            val_loss = 0.0
            total_samples = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    # 累加 Loss
                    val_loss += loss.item() * inputs.size(0)
                    total_samples += inputs.size(0)
            
            avg_val_loss = val_loss / total_samples
            
            if avg_val_loss < best_val_loss_this_run:
                best_val_loss_this_run = avg_val_loss

        # record result
        record = config.copy()
        record['best_val_loss'] = best_val_loss_this_run
        results.append(record)
        print(f"    --> Best Val Loss: {best_val_loss_this_run:.4f}")

    # Report
    print("\n\n🏆 ================= TUNING REPORT ================= 🏆")
    df = pd.DataFrame(results)
    df_sorted = df.sort_values(by='best_val_loss', ascending=True)
    print(df_sorted.to_string(index=False))
    
    best_config = df_sorted.iloc[0].to_dict()
    return best_config

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    best_params = tuning_runner(
        data_root=".", 
        split_dir="train_val_test", 
        device=device
    )