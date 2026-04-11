import os
import glob
import shutil
import random

# 定义源路径和目标路径
source_dir = r"f:\npfcode\data\chip-surface-defect-dataset"
dest_dir = r"f:\npfcode\data\yolo_dataset"

# YOLO标准数据集结构要求：
# yolo_dataset/
# ├── images/
# │   ├── train/
# │   └── val/
# └── labels/
#     ├── train/
#     └── val/

# 创建目录
for folder in ['images', 'labels']:
    for split in ['train', 'val']:
        os.makedirs(os.path.join(dest_dir, folder, split), exist_ok=True)

# 寻找所有yolo_label目录
yolo_label_dirs = glob.glob(os.path.join(source_dir, "**", "yolo_label"), recursive=True)

all_samples = []
for yl_dir in yolo_label_dirs:
    # 对应的image目录在同级
    parent_dir = os.path.dirname(yl_dir)
    img_dir = os.path.join(parent_dir, "image")
    
    if not os.path.exists(img_dir):
        continue
        
    for txt_file in glob.glob(os.path.join(yl_dir, "*.txt")):
        basename = os.path.splitext(os.path.basename(txt_file))[0]
        # 寻找对应的图片文件
        img_candidates = glob.glob(os.path.join(img_dir, basename + ".*"))
        if len(img_candidates) > 0:
            img_file = img_candidates[0]
            # 只有当txt文件大小>0(有标注)或不包含物体时都保存
            all_samples.append((img_file, txt_file))

print(f"共找到 {len(all_samples)} 对图像和标签。")

# 随机打乱并按照 9:1 分割训练集和验证集
random.seed(42)
random.shuffle(all_samples)
split_idx = int(len(all_samples) * 0.9)
train_samples = all_samples[:split_idx]
val_samples = all_samples[split_idx:]

def copy_samples(samples, split_name):
    for img_path, txt_path in samples:
        # 使用copy2保留元数据，如果想节省空间在Linux下可用软链接，Windows下建议直接复制
        shutil.copy2(img_path, os.path.join(dest_dir, "images", split_name, os.path.basename(img_path)))
        shutil.copy2(txt_path, os.path.join(dest_dir, "labels", split_name, os.path.basename(txt_path)))

print("正在复制训练集数据...")
copy_samples(train_samples, "train")
print("正在复制验证集数据...")
copy_samples(val_samples, "val")

# 读取类别并生成 dataset.yaml
classes_txt = os.path.join(source_dir, "classes.txt")
classes = []
if os.path.exists(classes_txt):
    with open(classes_txt, "r", encoding="utf-8") as f:
        classes = [line.strip() for line in f.readlines() if line.strip()]

yaml_path = os.path.join(dest_dir, "dataset.yaml")
with open(yaml_path, "w", encoding="utf-8") as f:
    f.write(f"path: {dest_dir}\n")
    f.write(f"train: images/train\n")
    f.write(f"val: images/val\n\n")
    f.write(f"names:\n")
    for i, c in enumerate(classes):
        f.write(f"  {i}: {c}\n")

print(f"数据集划分完成！可以直接用于YOLO训练。\n配置文件已保存到: {yaml_path}")
