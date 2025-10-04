import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import json
from sklearn.model_selection import train_test_split
from models import get_model

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class WeatherDataset(Dataset):
    """天气分类数据集"""
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # 加载图片
        image = Image.open(img_path).convert('RGB')
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        
        return image, label


def load_dataset(data_dir, class_names, test_size=0.2, random_state=42):
    """
    加载数据集
    
    Args:
        data_dir: 数据目录
        class_names: 类别名称列表
        test_size: 测试集比例
        random_state: 随机种子
    
    Returns:
        train_paths, train_labels, test_paths, test_labels, class_to_idx
    """
    all_paths = []
    all_labels = []
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    
    print(f"加载数据集，类别: {class_names}")
    
    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"警告: 类别目录不存在: {class_dir}")
            continue
        
        # 获取该类别的所有图片
        images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        class_label = class_to_idx[class_name]
        
        for img_name in images:
            img_path = os.path.join(class_dir, img_name)
            all_paths.append(img_path)
            all_labels.append(class_label)
        
        print(f"  {class_name}: {len(images)} 张图片")
    
    print(f"总共: {len(all_paths)} 张图片")
    
    # 划分训练集和测试集
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        all_paths, all_labels, test_size=test_size, random_state=random_state, stratify=all_labels
    )
    
    print(f"训练集: {len(train_paths)} 张, 测试集: {len(test_paths)} 张")
    
    return train_paths, train_labels, test_paths, test_labels, class_to_idx


def get_transforms(input_size=224):
    """获取数据变换"""
    train_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, test_transform


def train_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="训练中")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 统计
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # 更新进度条
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100 * correct / total:.2f}%'})
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """验证/测试"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="测试中")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # 统计
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 更新进度条
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100 * correct / total:.2f}%'})
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc


def plot_history(history, save_path):
    """绘制训练历史"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss曲线
    ax1.plot(history['train_loss'], label='训练Loss', marker='o')
    ax1.plot(history['val_loss'], label='测试Loss', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss曲线')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy曲线
    ax2.plot(history['train_acc'], label='训练Acc', marker='o')
    ax2.plot(history['val_acc'], label='测试Acc', marker='s')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('准确率曲线')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"训练曲线已保存到: {save_path}")
    plt.close()


def train(args):
    """训练主函数"""
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 类别名称
    class_names = ['sunny', 'snow', 'rain', 'lightning', 'fogsmog', 'cloudy']
    num_classes = len(class_names)
    
    # 加载数据集
    train_paths, train_labels, test_paths, test_labels, class_to_idx = load_dataset(
        args.data_dir, class_names, test_size=args.test_size, random_state=args.seed
    )
    
    # 数据变换
    train_transform, test_transform = get_transforms(args.input_size)
    
    # 创建数据集和数据加载器
    train_dataset = WeatherDataset(train_paths, train_labels, transform=train_transform)
    test_dataset = WeatherDataset(test_paths, test_labels, transform=test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                             num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, 
                            num_workers=args.num_workers, pin_memory=True)
    
    # 创建模型
    print(f"\n创建模型: {args.model}")
    model = get_model(args.model, num_classes=num_classes)
    model = model.to(device)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, 
                                                     patience=5, verbose=True)
    
    # 训练历史
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # 最佳模型
    best_acc = 0.0
    best_epoch = 0
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 开始训练
    print(f"\n开始训练，共 {args.epochs} 个epoch\n")
    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print("-" * 50)
        
        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # 验证
        val_loss, val_acc = validate(model, test_loader, criterion, device)
        
        # 更新学习率
        scheduler.step(val_acc)
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # 打印结果
        print(f"\n训练 - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"测试 - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}\n")
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch + 1
            
            # 保存模型
            save_path = os.path.join(args.save_dir, f'{args.model}_best.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'class_to_idx': class_to_idx,
                'class_names': class_names
            }, save_path)
            print(f"✓ 保存最佳模型到: {save_path} (Acc: {best_acc:.4f})\n")
    
    # 训练结束
    print("=" * 50)
    print(f"训练完成！")
    print(f"最佳准确率: {best_acc:.4f} (Epoch {best_epoch})")
    
    # 保存训练历史
    history_path = os.path.join(args.save_dir, f'{args.model}_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
    print(f"训练历史已保存到: {history_path}")
    
    # 绘制训练曲线
    plot_path = os.path.join(args.save_dir, f'{args.model}_training_curves.png')
    plot_history(history, plot_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='天气分类模型训练')
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, default='image', help='数据目录')
    parser.add_argument('--test_size', type=float, default=0.2, help='测试集比例')
    parser.add_argument('--input_size', type=int, default=224, help='输入图片大小')
    
    # 模型参数
    parser.add_argument('--model', type=str, default='vit',
                       choices=['dinov3', 'clip', 'resnet34','vit','resnet152'], help='模型名称')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=3, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载线程数')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    # 保存参数
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='模型保存目录')
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 开始训练
    train(args)

