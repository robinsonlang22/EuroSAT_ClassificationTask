import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from tqdm import tqdm

from model import SatelliteResNet
from dataset import get_dataloaders

try:
    from config import RGB_DATASET_ROOT, SPLITS_ROOT, SEED
except ImportError:
    RGB_DATASET_ROOT = "EuroSAT_RGB"
    SPLITS_ROOT = "splits"
    SEED = 3719704

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seed set to {seed}")

def denormalize(tensor):
    img = tensor.cpu().numpy().transpose((1, 2, 0))
    img = img * STD + MEAN
    img = np.clip(img, 0, 1)
    return img

def collect_results(model, loader, class_names):
    model.eval()
    correct_samples = []
    incorrect_samples = []
    
    print(f" Analyzing {len(loader.dataset)}  Valset pictures...")
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Inferencing"):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            max_probs, preds = torch.max(probs, dim=1)
            
            for i in range(len(images)):
                conf = max_probs[i].item()
                pred_idx = preds[i].item()
                true_idx = labels[i].item()
                
                info = {
                    "conf": conf,
                    "img": images[i].cpu(),
                    "pred_name": class_names[pred_idx],
                    "true_name": class_names[true_idx]
                }
                
                if pred_idx == true_idx:
                    correct_samples.append(info)
                else:
                    incorrect_samples.append(info)
                    
    return correct_samples, incorrect_samples

def plot_top_bottom(samples, title, filename_suffix):
    num = min(5, len(samples))
    if num == 0:
        print(f"There is no sample likes {title} ")
        return

    plt.figure(figsize=(15, 4))
    
    for i in range(num):
        info = samples[i]
        ax = plt.subplot(1, 5, i + 1)
        
        img = denormalize(info['img'])
        plt.imshow(img, interpolation='nearest')
        plt.axis('off')
        
        is_correct = info['pred_name'] == info['true_name']
        color = 'green' if is_correct else 'red'
        
        label_text = (f"True: {info['true_name']}\n"
                      f"Pred: {info['pred_name']}\n"
                      f"Conf: {info['conf']:.4f}")
        
        plt.title(label_text, color=color, fontsize=10, fontweight='bold')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    save_name = f"analysis_val_{filename_suffix}.png"
    plt.savefig(save_name)
    print(f" Picture saved: {save_name}") 

def main(args):

    set_seed(SEED)
    
    # 1. Loading Data
    print(" Loading Test Data...")
    _, _, test_loader = get_dataloaders(
        data_root=args.data_root,
        split_dir=args.split_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    class_names = test_loader.dataset.classes
    print(f" Detected Classes: {class_names}")

    # 2. Loading Model
    print(f" Loading Model from {args.model_path}...")
    model = SatelliteResNet(num_classes=10).to(device)
    
    if not os.path.exists(args.model_path):
        print(f" Error: No such file {args.model_path}")
        return

    state_dict = torch.load(args.model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)

    # 3. Collect Results
    correct, incorrect = collect_results(model, test_loader, class_names)

    # 4. NEW: Per-Class Top/Bottom
    target_classes = ['Forest', 'Highway', 'River'] 
    
    print(f"\n Generating Plots for specific classes: {target_classes}...")

    for cls_name in target_classes:
        cls_correct = [x for x in correct if x['true_name'] == cls_name]
        cls_incorrect = [x for x in incorrect if x['true_name'] == cls_name]
        
        # Sort
        cls_correct.sort(key=lambda x: x['conf'], reverse=True)
        cls_incorrect.sort(key=lambda x: x['conf'], reverse=True)
        
        # Plot
        if len(cls_correct) > 0:
            plot_top_bottom(cls_correct[:5], f"Top 5 Correct ({cls_name})", f"{cls_name}_top5")
        
        if len(cls_incorrect) > 0:
            plot_top_bottom(cls_incorrect[:5], f"Bottom 5 Failures ({cls_name})", f"{cls_name}_bottom5")
        else:
            print(f"Good job! No failures found for class {cls_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default=RGB_DATASET_ROOT)
    parser.add_argument('--split_dir', type=str, default=SPLITS_ROOT)
    parser.add_argument('--model_path', type=str, default="checkpoints/best_model.pth", help='Path to .pth')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=2)

    args = parser.parse_args()
    main(args)