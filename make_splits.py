import random
from pathlib import Path
from sklearn.model_selection import train_test_split

def generate_splits(dataset_root, output_root="train_val_test", seed=369, 
                    train_ratio=0.5, val_ratio=0.2, test_ratio=0.3, max_images=2000):
    
    dataset_root = Path(dataset_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    
    classes = sorted([d.name for d in dataset_root.iterdir() if d.is_dir()])
    train_files, val_files, test_files = [], [], []

    # 1. set seed as 369
    random.seed(seed)
    print(f"🔨 Processing splits from {dataset_root}...")

    for cls in classes:
        class_dir = dataset_root / cls
        # construct relative path 
        files = sorted([str(f.relative_to(dataset_root.parent)) for f in class_dir.glob("*.jpg")])

        # 2. shuffle & limit 2000 pic per cls
        random.shuffle(files)
        if len(files) > max_images:
            files = files[:max_images]

        labels = [cls] * len(files)

        # 3. segmentation
        train_f, temp_f, train_l, temp_l = train_test_split(
            files, labels, test_size=(1 - train_ratio), stratify=labels, random_state=seed
        )
        val_f, test_f, _, _ = train_test_split(
            temp_f, temp_l, test_size=test_ratio / (val_ratio + test_ratio),
            stratify=temp_l, random_state=seed
        )

        train_files.extend(train_f)
        val_files.extend(val_f)
        test_files.extend(test_f)

    # 4. assert to check overlapping
    assert len(set(train_files) & set(val_files)) == 0, "Error: Train/Val overlap"
    assert len(set(train_files) & set(test_files)) == 0, "Error: Train/Test overlap"
    assert len(set(val_files) & set(test_files)) == 0,  "Error: Val/Test overlap"

    # 5. write in txt
    (output_root / "train.txt").write_text("\n".join(train_files))
    (output_root / "val.txt").write_text("\n".join(val_files))
    (output_root / "test.txt").write_text("\n".join(test_files))
    
    print(f"Success! Splits saved to: {output_root}")
    print(f"Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")

if __name__ == "__main__":
    generate_splits(dataset_root="EuroSAT_RGB", output_root="train_val_test")
