import torch
import os
import argparse
from model import SatelliteResNet
from dataset import get_dataloaders
from tqdm import tqdm

try:
    from config import RGB_DATASET_ROOT, SPLITS_ROOT
except ImportError:
    RGB_DATASET_ROOT = "EuroSAT_RGB"
    SPLITS_ROOT = "splits"

def check_reproducibility(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. checking  Logits
    if not os.path.exists(args.saved_logits):
        print(f" Error: Saved logits not found at {args.saved_logits}")
        print(" Please run test.py first!")
        return

    # 2. loading Logits
    print(f" Loading saved logits from {args.saved_logits}...")
    saved_logits = torch.load(args.saved_logits, map_location="cpu")

    print(" Re-running inference to check consistency...")
    
    # loading test set
    _, _, test_loader = get_dataloaders(
        data_root=args.data_root,
        split_dir=args.split_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # loading model
    model = SatelliteResNet(num_classes=10).to(device)
    state_dict = torch.load(args.model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    new_logits = []
    with torch.no_grad():
        for images, _ in tqdm(test_loader, desc="Re-Inferencing"):
            images = images.to(device)
            outputs = model(images)
            new_logits.append(outputs.cpu())

    new_logits = torch.cat(new_logits, dim=0)

    # 4. compare result by torch.allclose
    is_match = torch.allclose(saved_logits, new_logits, atol=1e-5)
    diff = (saved_logits - new_logits).abs().max().item()

    print("\n" + "="*40)
    print(f" Reproducibility Report")
    print("="*40)
    print(f"Max difference between runs: {diff:.8f}")
    
    if is_match:
        print(" SUCCESS: The results are reproducible!")
        print("   (Saved logits match the re-calculated logits)")
    else:
        print(" FAILURE: Results do not match!")
        print("   Check if seed was set correctly or if shuffle=True was used accidentally.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default=RGB_DATASET_ROOT)
    parser.add_argument('--split_dir', type=str, default=SPLITS_ROOT)
    parser.add_argument('--model_path', type=str, default="checkpoints/mild_run/best_model.pth")
    parser.add_argument('--saved_logits', type=str, default="submission/test_logits.pt")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=2)

    args = parser.parse_args()
    check_reproducibility(args)