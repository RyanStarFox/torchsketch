#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练单个模型的脚本
支持多种模型类型：resnet18, mobilenetv2, inceptionv3, tcn, bilstm, gat, mgt等
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

try:
    import torchsketch.networks.cnn as cnns
    import torchsketch.networks.rnn as rnns
    import torchsketch.networks.tcn as tcns
    import torchsketch.networks.gnn as gnns
    TORCHSKETCH_AVAILABLE = True
except ImportError:
    TORCHSKETCH_AVAILABLE = False
    print("警告: torchsketch未安装，将使用torchvision模型")


class CoordinateDataset(Dataset):
    """加载坐标数据的Dataset"""
    def __init__(self, data_dir, class_mapping_file, transform=None, use_images=False):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.use_images = use_images
        
        # 加载类别映射
        with open(class_mapping_file, 'r') as f:
            self.class_mapping = json.load(f)
        
        self.label_to_idx = self.class_mapping['label_to_idx']
        self.n_classes = self.class_mapping['n_classes']
        
        # 收集所有数据文件
        self.samples = []
        coordinate_dir = self.data_dir / 'coordinate_files'
        
        for label, idx in self.label_to_idx.items():
            label_dir = coordinate_dir / label
            if label_dir.exists():
                for npy_file in label_dir.glob('*.npy'):
                    self.samples.append((str(npy_file), idx, label))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        npy_path, label_idx, label_name = self.samples[idx]
        
        # 加载坐标数据
        data = np.load(npy_path, allow_pickle=True)
        
        # 如果数据是数组，直接使用；如果是字典，提取坐标
        if isinstance(data, np.ndarray):
            coordinates = data
        elif isinstance(data, np.lib.npyio.NpzFile):
            # 从npz文件中提取坐标
            keys = list(data.keys())
            coordinates = data[keys[0]] if keys else np.array([])
        else:
            coordinates = np.array(data)
        
        # 转换为tensor
        if len(coordinates.shape) == 2:
            # 形状: [n_points, 2] 或 [n_points, 3]
            coordinates = torch.FloatTensor(coordinates)
        else:
            # 如果是其他形状，尝试reshape
            coordinates = torch.FloatTensor(coordinates.flatten()[:1000].reshape(-1, 2))
        
        # 如果使用图片，加载图片
        if self.use_images:
            # 这里可以添加图片加载逻辑
            pass
        
        if self.transform:
            coordinates = self.transform(coordinates)
        
        return coordinates, label_idx


def get_model(model_name, n_classes, input_dim=2, **kwargs):
    """根据模型名称创建模型"""
    if not TORCHSKETCH_AVAILABLE:
        # 使用torchvision模型作为fallback
        if model_name == 'resnet18':
            from torchvision.models import resnet18
            model = resnet18(pretrained=False)
            model.fc = nn.Linear(model.fc.in_features, n_classes)
            return model
        elif model_name == 'mobilenetv2':
            from torchvision.models import mobilenet_v2
            model = mobilenet_v2(pretrained=False)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, n_classes)
            return model
        elif model_name == 'inceptionv3':
            from torchvision.models import inception_v3
            model = inception_v3(pretrained=False, aux_logits=False)
            model.fc = nn.Linear(model.fc.in_features, n_classes)
            return model
    
    # 使用torchsketch模型
    if model_name == 'resnet18':
        return cnns.resnet18(num_classes=n_classes)
    elif model_name == 'mobilenetv2':
        return cnns.mobilenet_v2(num_classes=n_classes)
    elif model_name == 'inceptionv3':
        return cnns.inception_v3(num_classes=n_classes)
    elif model_name == 'tcn':
        return tcns.tcn(input_size=input_dim, num_classes=n_classes, **kwargs)
    elif model_name == 'bilstm' or model_name.startswith('bilstm'):
        hidden_size = kwargs.get('hidden_size', 128)
        num_layers = kwargs.get('num_layers', 2)
        if '3_layer' in model_name:
            num_layers = 3
        return rnns.bilstm(input_size=input_dim, hidden_size=hidden_size, 
                          num_layers=num_layers, num_classes=n_classes)
    elif model_name == 'gat':
        return gnns.gat(num_classes=n_classes, **kwargs)
    elif model_name == 'gcn':
        return gnns.gcn(num_classes=n_classes, **kwargs)
    elif model_name == 'mgt' or model_name == 'multigraph_transformer':
        return gnns.multigraph_transformer(num_classes=n_classes, **kwargs)
    else:
        raise ValueError(f"不支持的模型类型: {model_name}")


def train_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        
        # 处理不同形状的输入
        if len(data.shape) == 3:
            # [batch, seq_len, features]
            pass
        elif len(data.shape) == 2:
            # [batch, features] - 需要reshape
            data = data.unsqueeze(1)  # [batch, 1, features]
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """验证模型"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            
            if len(data.shape) == 2:
                data = data.unsqueeze(1)
            
            output = model(data)
            loss = criterion(output, target)
            
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def main():
    parser = argparse.ArgumentParser(description='训练单个模型')
    parser.add_argument('--dataset', type=str, required=True, help='数据集名称（如sampling_1）')
    parser.add_argument('--model', type=str, required=True, 
                       choices=['resnet18', 'mobilenetv2', 'inceptionv3', 'tcn', 
                               'bilstm', 'bilstm_30_epoch', 'bilstm_3_layer_30_epoch',
                               'gat', 'gcn', 'mgt', 'multigraph_transformer'],
                       help='模型类型')
    parser.add_argument('--epochs', type=int, default=30, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--device', type=str, default='auto', 
                       choices=['auto', 'cuda', 'mps', 'cpu'], help='设备')
    parser.add_argument('--early_stop', type=int, default=5, 
                       help='Early stopping patience')
    parser.add_argument('--train_split', type=float, default=0.7, help='训练集比例')
    parser.add_argument('--val_split', type=float, default=0.15, help='验证集比例')
    
    args = parser.parse_args()
    
    # 设置设备
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    
    print(f"使用设备: {device}")
    
    # 数据路径
    data_dir = Path(f"data/{args.dataset}")
    class_mapping_file = data_dir / "class_mapping.json"
    
    if not class_mapping_file.exists():
        raise FileNotFoundError(f"找不到class_mapping.json: {class_mapping_file}")
    
    # 加载数据集
    dataset = CoordinateDataset(data_dir, class_mapping_file)
    n_classes = dataset.n_classes
    
    # 划分数据集
    train_size = int(args.train_split * len(dataset))
    val_size = int(args.val_split * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}, 测试集: {len(test_dataset)}")
    
    # 创建模型
    model = get_model(args.model, n_classes, input_dim=2)
    model = model.to(device)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 训练历史
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    # 保存最佳模型
    model_dir = Path(f"model/{args.model}")
    model_dir.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0
    patience_counter = 0
    
    print(f"\n开始训练 {args.model}...")
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    for epoch in range(args.epochs):
        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # 验证
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f"Epoch [{epoch+1}/{args.epochs}] - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), model_dir / "best_model.pth")
            print(f"  -> 保存最佳模型 (Val Acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= args.early_stop:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # 测试
    print("\n在测试集上评估...")
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"测试集准确率: {test_acc:.2f}%")
    
    # 保存训练历史
    history = {
        'train_loss': train_losses,
        'train_acc': train_accs,
        'val_loss': val_losses,
        'val_acc': val_accs,
        'test_acc': test_acc,
        'best_val_acc': best_val_acc
    }
    
    import pickle
    with open(model_dir / "training_history.pkl", 'wb') as f:
        pickle.dump(history, f)
    
    print(f"\n训练完成！最佳验证准确率: {best_val_acc:.2f}%")
    print(f"模型保存在: {model_dir / 'best_model.pth'}")


if __name__ == '__main__':
    main()


