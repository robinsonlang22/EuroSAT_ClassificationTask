import torch
import os
import argparse
from model import SatelliteResNet
from dataset import get_dataloaders
from tqdm import tqdm

def check_reproducibility(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 检查文件是否存在
    if not os.path.exists(args.saved_logits):
        print(f" Error: Saved logits not found at {args.saved_logits}")
        print(" Please run generate_submission.py first!")
        return

    # 2. 加载之前保存的 Logits
    print(f" Loading saved logits from {args.saved_logits}...")
    saved_logits = torch.load(args.saved_logits, map_location="cpu") # 放在 CPU 对比即可

    # 3. 重新跑一遍模型 (Live Inference)
    print("🔄 Re-running inference to check consistency...")
    
    # 加载数据
    _, _, test_loader = get_dataloaders(
        data_root=args.data_root,
        split_dir=args.split_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # 加载模型
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

    # 4. 核心对比：使用 torch.allclose
    # atol=1e-5 表示允许小数点后5位的误差（浮点数计算会有微小误差，这是正常的）
    is_match = torch.allclose(saved_logits, new_logits, atol=1e-5)
    
    # 计算最大误差
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
    parser.add_argument('--data_root', type=str, default=".")
    parser.add_argument('--split_dir', type=str, default="train_val_test")
    parser.add_argument('--model_path', type=str, default="checkpoints/best_model.pth")
    parser.add_argument('--saved_logits', type=str, default="submission/test_logits.pt")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=2)

    args = parser.parse_args()
    check_reproducibility(args)