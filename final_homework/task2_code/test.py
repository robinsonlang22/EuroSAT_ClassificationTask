import torch
import os
import argparse
from tqdm import tqdm
from model import SatelliteResNet
from dataset import get_dataloaders

try:
    from config import RGB_DATASET_ROOT, SPLITS_ROOT
except ImportError:
    RGB_DATASET_ROOT = "EuroSAT_RGB"
    SPLITS_ROOT = "splits"

def generate_logits(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" Generating logits using device: {device}")

    # 1. loading test set
    _, _, test_loader = get_dataloaders(
        data_root=args.data_root,
        split_dir=args.split_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # 2. loading model
    print(f" Loading model from {args.model_path}...")
    model = SatelliteResNet(num_classes=10).to(device)
    
    # loading weights
    state_dict = torch.load(args.model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    
    model.eval()

    all_logits = []

    # 3. Inference
    print(" Starting inference on Test set...")
    with torch.no_grad(): 
        for images, _ in tqdm(test_loader, desc="Inferencing"):
            images = images.to(device)
            outputs = model(images)
            all_logits.append(outputs.cpu())

    # 4. concat results
    all_logits = torch.cat(all_logits, dim=0)
    print(f" Generated logits shape: {all_logits.shape}")

    # 5. saving
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, "test_logits.pt")
    torch.save(all_logits, save_path)
    print(f" Logits saved to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_root', type=str, default=RGB_DATASET_ROOT)
    parser.add_argument('--split_dir', type=str, default=SPLITS_ROOT)
    parser.add_argument('--model_path', type=str, default="checkpoints/best_model.pth", help='Path to trained model')
    parser.add_argument('--output_dir', type=str, default="submission", help='Where to save logits')
    
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=2)

    args = parser.parse_args()
    generate_logits(args)