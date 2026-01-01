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
    
    print(" Setting up model and dataloader...")
    _, _, test_loader = get_dataloaders(
        data_root=args.data_root,
        split_dir=args.split_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    test_txt_path = os.path.join(args.split_dir, "test.txt")
    current_filenames = []
    with open(test_txt_path, "r") as f:
        for line in f:
            if line.strip():
                rel_path = line.strip().split()[0]
                current_filenames.append(os.path.basename(rel_path))

    model = SatelliteResNet(num_classes=10).to(device)
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        return
        
    state_dict = torch.load(args.model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    print(" Running inference...")
    new_logits = []
    
    with torch.no_grad():
        for images, _ in tqdm(test_loader, desc="Inference"):
            images = images.to(device)
            outputs = model(images)
            new_logits.append(outputs.cpu())

    new_logits = torch.cat(new_logits, dim=0)

    # Switch Logic
    
    # CASE A: GENERATE MODE (Switch is ON)
    if args.generate:
        print(f"\n[MODE] Generation")
        os.makedirs(os.path.dirname(args.saved_logits), exist_ok=True)
        
        # Save Logits
        torch.save(new_logits, args.saved_logits)
        
        # Save Filenames
        with open(args.saved_filenames, "w") as f:
            for name in current_filenames:
                f.write(f"{name}\n")
                
        print(f" SUCCESS: Logits saved to {args.saved_logits}")
        print(f" SUCCESS: Filenames saved to {args.saved_filenames}")

    # CASE B: CHECK MODE (Default)
    else:
        print(f"\n[MODE] Verification")
        if not os.path.exists(args.saved_logits) or not os.path.exists(args.saved_filenames):
            print(f" Error: Saved files not found.")
            print(" Please run with --generate first.")
            return

        print(f" Loading saved data...")
        saved_logits = torch.load(args.saved_logits, map_location="cpu")
        
        with open(args.saved_filenames, "r") as f:
            saved_filenames = [line.strip() for line in f]

        # Check 1: Filenames Order
        if saved_filenames != current_filenames:
            print(" FAILURE: Filename order mismatch!")
            print(" The test set is not being loaded in the same order as before.")
            return
        else:
            print(" CHECK: Filename order matches.")

        # Check 2: Values
        if saved_logits.shape != new_logits.shape:
             print(f" FAILURE: Shape mismatch! Saved: {saved_logits.shape}, New: {new_logits.shape}")
             return

        is_match = torch.allclose(saved_logits, new_logits, atol=1e-5)
        diff = (saved_logits - new_logits).abs().max().item()

        print("\n" + "="*40)
        print(f" Reproducibility Report")
        print("="*40)
        print(f"Max difference: {diff:.8f}")
        
        if is_match:
            print(" SUCCESS: The results are reproducible!")
        else:
            print(" FAILURE: Logits do not match!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default=RGB_DATASET_ROOT)
    parser.add_argument('--split_dir', type=str, default=SPLITS_ROOT)
    parser.add_argument('--model_path', type=str, default="checkpoints/mild_run/best_model.pth")
    
    parser.add_argument('--saved_logits', type=str, default="submission/test_logits.pt")
    parser.add_argument('--saved_filenames', type=str, default="submission/test_filenames.txt")
    
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--generate', action='store_true', help='Switch to save logits instead of comparing')

    args = parser.parse_args()
    check_reproducibility(args)