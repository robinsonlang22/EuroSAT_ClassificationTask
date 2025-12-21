import torch
import os
import argparse
from tqdm import tqdm
from model import SatelliteResNet
from dataset import get_dataloaders

def generate_logits(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" Generating logits using device: {device}")

    # 1. 准备数据 (只需要 Test Loader)
    # 注意：Test Loader 的 shuffle 必须是 False (在 get_dataloaders 里已经设好了)
    # 这样生成的 logits 顺序才能和 test.txt 里的文件名顺序一一对应
    _, _, test_loader = get_dataloaders(
        data_root=args.data_root,
        split_dir=args.split_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # 2. 加载模型
    print(f" Loading model from {args.model_path}...")
    model = SatelliteResNet(num_classes=10).to(device)
    
    # 【关键】加载权重
    # map_location 确保在 CPU 上也能加载 GPU 训练的模型
    state_dict = torch.load(args.model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    
    # 【关键】开启评估模式 (关闭 Dropout, BatchNorm 锁定)
    model.eval()

    all_logits = []

    # 3. 推理 (Inference)
    print(" Starting inference on Test set...")
    with torch.no_grad(): # 这一步很关键，省显存，不计算梯度
        for images, _ in tqdm(test_loader, desc="Inferencing"):
            images = images.to(device)
            outputs = model(images)
            all_logits.append(outputs.cpu()) # 把结果拿回 CPU

    # 4. 拼接结果
    # 现在的形状是 [Batch1, Batch2, ...] -> [Total_Test_Images, 10]
    all_logits = torch.cat(all_logits, dim=0)
    print(f" Generated logits shape: {all_logits.shape}")

    # 5. 保存
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, "test_logits.pt")
    torch.save(all_logits, save_path)
    print(f" Logits saved to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 路径参数
    parser.add_argument('--data_root', type=str, default=".", help='Root dir')
    parser.add_argument('--split_dir', type=str, default="train_val_test", help='Split dir')
    parser.add_argument('--model_path', type=str, default="checkpoints/best_model.pth", help='Path to trained model')
    parser.add_argument('--output_dir', type=str, default="submission", help='Where to save logits')
    
    # 系统参数
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=2)

    args = parser.parse_args()
    generate_logits(args)