import argparse
import os
import warnings
import time
from pathlib import Path

from sklearn.model_selection import train_test_split
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import models_vit as models
from timm.models.layers import trunc_normal_
from config import device

warnings.simplefilter(action='ignore', category=FutureWarning)

# 图像预处理
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.Resize(255),
    transforms.CenterCrop(255),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

test_transform = transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(255),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

class_tabel = ('A', 'C', 'CSC', 'D', 'G', 'N', 'RP' , 'RVO')
class_tabel_with_level = ('A', 'C', 'CSC', 'D0','D1','D2','D3','D4', 'G', 'N', 'RP' , 'RVO')

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.tiff', '.TIFF'}

# 自定义数据集类
class FundusDataset(Dataset):
    def __init__(self, img_paths, labels, transform=None):
        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.labels[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

def load_data(args):
    all_img_paths, all_labels = [], []

    root = Path(args.data_path)

    for i, class_name in enumerate(class_tabel):
        # 递归查找所有目录名为 class_name 的目录
        for class_dir in root.rglob(class_name):
            if class_dir.is_dir():
                # 在该目录下递归查找图像文件
                for img_path in class_dir.rglob('*'):
                    if img_path.suffix.lower() in IMG_EXTS and img_path.is_file():
                        all_img_paths.append(str(img_path))
                        all_labels.append(i)

    # 划分训练集和验证集
    train_img_paths, val_img_paths, train_labels, val_labels = train_test_split(
        all_img_paths, all_labels, test_size=args.val_ratio, random_state=42)

    # 构建数据集
    train_dataset = FundusDataset(train_img_paths, train_labels, transform=train_transform)
    val_dataset = FundusDataset(val_img_paths, val_labels, transform=test_transform)

    # 构建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    return train_loader, val_loader

def load_model(args, device):
    if args.model == 'RETFound_mae':
        model = models.__dict__[args.model](
            img_size=args.input_size,
            num_classes=args.nb_classes,
            drop_path_rate=args.drop_path,
            global_pool=args.global_pool,
        )
    else:
        model = models.__dict__[args.model](
            num_classes=args.nb_classes,
            drop_path_rate=args.drop_path,
            args=args,
        )

    checkpoint_path = args.models_path
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    checkpoint_model = checkpoint['model']

    checkpoint_model = {k.replace("backbone.", ""): v for k, v in checkpoint_model.items()}
    checkpoint_model = {k.replace("mlp.w12.", "mlp.fc1."): v for k, v in checkpoint_model.items()}
    checkpoint_model = {k.replace("mlp.w3.", "mlp.fc2."): v for k, v in checkpoint_model.items()}

    state_dict = model.state_dict()
    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]

    # load pre-trained model
    model.load_state_dict(checkpoint_model, strict=False)

    trunc_normal_(model.head.weight, std=2e-5)

    # 初始化分类头权重
    if hasattr(model, 'head') and isinstance(model.head, nn.Linear):
        trunc_normal_(model.head.weight, std=2e-5)
        if model.head.bias is not None:
            nn.init.constant_(model.head.bias, 0)

    model.to(device, dtype=torch.float32)

    # 冻结除分类头外的所有参数
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("head")

    return model

def load_finetuned_model(args, device):
    if args.model == 'RETFound_mae':
        model = models.__dict__[args.model](
            img_size=args.input_size,
            num_classes=args.nb_classes,
            drop_path_rate=args.drop_path,
            global_pool=args.global_pool,
        )
    else:
        model = models.__dict__[args.model](
            num_classes=args.nb_classes,
            drop_path_rate=args.drop_path,
            args=args,
        )

    # 加载保存的微调权重
    checkpoint_path = os.path.join(args.save_path, 'best_model.pth')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"找不到微调权重文件: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # 尝试加载权重，记录不匹配项
    model.load_state_dict(checkpoint, strict=False)

    model.to(device, dtype=torch.float32)

    # 冻结除分类头和 Adapter 外的所有参数
    for name, param in model.named_parameters():
        if name.startswith("head") or "adapter" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    return model

def load_model_finetuned(args, device):
    if args.model == 'RETFound_mae':
        model = models.__dict__[args.model](
            img_size=args.input_size,
            num_classes=args.nb_classes,
            drop_path_rate=args.drop_path,
            global_pool=args.global_pool,
        )
    else:
        model = models.__dict__[args.model](
            num_classes=args.nb_classes,
            drop_path_rate=args.drop_path,
            args=args,
        )

    # 加载保存的微调权重
    checkpoint_path = args.models_path
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"找不到微调权重文件: {checkpoint_path}")

    # 加载 checkpoint，不加载 head
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.copy()
    for k in list(state_dict.keys()):
        if k.startswith('head.'):
            del state_dict[k]

    model.load_state_dict(state_dict, strict=False)
    model.to(device, dtype=torch.float32)

    # 重新初始化分类头
    if hasattr(model, 'head') and isinstance(model.head, nn.Linear):
        trunc_normal_(model.head.weight, std=2e-5)
        if model.head.bias is not None:
            nn.init.constant_(model.head.bias, 0)

    # 冻结除分类头和 Adapter 外的所有参数
    for name, param in model.named_parameters():
        if name.startswith("head") or "adapter" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    return model

def train(model, train_loader, val_loader, args, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=0.05)
    best_val_acc = 88

    # 计算总迭代次数
    total_iterations = len(train_loader) * args.epochs
    print(f"🚀 开始训练，总轮数: {args.epochs}, 每轮迭代数: {len(train_loader)}, 总迭代数: {total_iterations}")
    print(f"📊 学习率: {args.lr}, 批次大小: {args.batch_size}")
    print("=" * 80)

    for epoch in range(args.epochs):
        start_time = time.time()
        model.train()
        train_loss = 0
        
        print(f"🔄 Epoch {epoch+1}/{args.epochs} 开始训练...")

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device, dtype=torch.float32), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            # 每10个批次输出一次进度和损失
            if (batch_idx + 1) % 10 == 0:
                avg_loss = train_loss / (batch_idx + 1)
                progress = (epoch * len(train_loader) + batch_idx + 1) / (len(train_loader) * args.epochs) * 100
                print(f"  📍 Epoch {epoch+1}/{args.epochs} | Batch {batch_idx+1}/{len(train_loader)} | "
                      f"Progress: {progress:.1f}% | Current Loss: {loss.item():.4f} | Avg Loss: {avg_loss:.4f}")

        # 计算平均训练损失
        avg_train_loss = train_loss / len(train_loader)
        
        # 评估阶段
        print(f"🔍 Epoch {epoch+1}/{args.epochs} 开始验证...")
        model.eval()
        val_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device, dtype=torch.float32), labels.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_acc = 100. * correct / total
        avg_val_loss = val_loss / len(val_loader)
        elapsed = time.time() - start_time
        
        # 计算总体进度
        overall_progress = (epoch + 1) / args.epochs * 100

        print(f"📈 Epoch {epoch+1}/{args.epochs} 完成 | 总体进度: {overall_progress:.1f}%")
        print(f"   🎯 训练损失: {avg_train_loss:.4f} | 验证损失: {avg_val_loss:.4f} | 验证准确率: {val_acc:.2f}%")
        print(f"   ⏱️  耗时: {elapsed:.2f}s | 最佳准确率: {best_val_acc:.2f}%")
        
        # 损失变化趋势
        if epoch > 0:
            loss_change = avg_train_loss - prev_train_loss
            loss_trend = "↗️" if loss_change > 0 else "↘️" if loss_change < 0 else "➡️"
            print(f"   📊 损失变化: {loss_change:+.4f} {loss_trend}")
        
        prev_train_loss = avg_train_loss
        print("-" * 80)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(args.save_path, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.save_path, 'checkpoint-best.pth'))
            print(f"✅ 保存最佳模型，准确率: {best_val_acc:.2f}%")
            print(f"💾 模型已保存到: {os.path.join(args.save_path, 'checkpoint-best.pth')}")
    
    print("🎉 训练完成！")
    print(f"🏆 最佳验证准确率: {best_val_acc:.2f}%")

def convert_path_to_unix_style(path):
    if os.path.sep == '\\':
        return path
    return path.replace('\\', '/')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--models_path', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--input_size', default=255, type=int)
    parser.add_argument('--nb_classes', default=8, type=int)
    parser.add_argument('--global_pool', default='token')
    parser.add_argument('--drop_path', type=float, default=0.2)
    parser.add_argument('--val_ratio', type=float, default=0.2)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--save_path', type=str, required=True)
    args = parser.parse_args()

    # 兼容 Unix 路径
    if os.name == 'posix':
        args.models_path = convert_path_to_unix_style(args.models_path)
        args.data_path = convert_path_to_unix_style(args.data_path)
        args.save_path = convert_path_to_unix_style(args.save_path)

    model = load_model_finetuned(args, device)
    train_loader, val_loader = load_data(args)
    train(model, train_loader, val_loader, args, device)