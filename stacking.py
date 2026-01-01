#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stacking模型训练脚本
使用8个基础模型进行交叉验证，训练meta-learner
"""

import os
import json
import argparse
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from train_model import CoordinateDataset, get_model


# 8个基础模型（从stacking/cache目录推断）
BASE_MODELS = [
    'bilstm_3_layer_30_epoch',
    'bilstm_30_epoch',
    'gat',
    'inceptionv3',
    'mgt',
    'mobilenetv2',
    'resnet18',
    'tcn'
]


def load_model(model_name, n_classes, device, model_dir=None):
    """加载训练好的模型"""
    if model_dir is None:
        model_dir = Path(f"model/{model_name}")
    
    model = get_model(model_name, n_classes, input_dim=2)
    model_path = model_dir / "best_model.pth"
    
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"加载模型: {model_name}")
    else:
        print(f"警告: 找不到模型文件 {model_path}，使用未训练的模型")
    
    model = model.to(device)
    model.eval()
    return model


def get_predictions(model, dataloader, device):
    """获取模型预测结果"""
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for data, target in dataloader:
            data = data.to(device)
            
            if len(data.shape) == 2:
                data = data.unsqueeze(1)
            
            output = model(data)
            probs = torch.softmax(output, dim=1)
            
            all_probs.append(probs.cpu().numpy())
            all_labels.append(target.numpy())
    
    return np.vstack(all_probs), np.hstack(all_labels)


def train_stacking_cv(dataset, base_models, n_classes, device, n_folds=5, cache_dir=None):
    """使用交叉验证训练stacking模型"""
    if cache_dir is None:
        cache_dir = Path("model/stacking/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取所有标签用于分层交叉验证
    all_labels = [dataset[i][1] for i in range(len(dataset))]
    all_labels = np.array(all_labels)
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # 存储所有fold的OOF预测
    oof_predictions = {model_name: [] for model_name in base_models}
    oof_labels = []
    
    print(f"\n开始 {n_folds}-fold 交叉验证...")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(dataset)), all_labels)):
        print(f"\nFold {fold + 1}/{n_folds}")
        
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        
        train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)
        
        val_labels = [dataset[i][1] for i in val_idx]
        oof_labels.extend(val_labels)
        
        # 对每个基础模型进行预测
        for model_name in base_models:
            print(f"  处理模型: {model_name}")
            
            # 加载模型
            model = load_model(model_name, n_classes, device)
            
            # 在验证集上预测
            val_probs, _ = get_predictions(model, val_loader, device)
            
            # 存储OOF预测
            oof_predictions[model_name].append(val_probs)
            
            del model
            torch.cuda.empty_cache() if device.type == 'cuda' else None
    
    # 合并所有fold的预测
    print("\n合并OOF预测...")
    stacked_features = []
    
    for model_name in base_models:
        model_oof = np.vstack(oof_predictions[model_name])
        stacked_features.append(model_oof)
        
        # 保存OOF预测
        oof_file = cache_dir / f"oof_predictions_{model_name}_n{n_folds}.npy"
        np.save(oof_file, model_oof)
        print(f"  保存 {model_name} 的OOF预测: {oof_file}")
    
    # 堆叠特征: [n_samples, n_models * n_classes]
    X_meta = np.hstack(stacked_features)
    y_meta = np.array(oof_labels)
    
    print(f"\nMeta特征形状: {X_meta.shape}")
    print(f"Meta标签形状: {y_meta.shape}")
    
    # 训练meta-learner（逻辑回归）
    print("\n训练meta-learner...")
    meta_learner = LogisticRegression(max_iter=1000, random_state=42)
    meta_learner.fit(X_meta, y_meta)
    
    # 评估meta-learner
    meta_pred = meta_learner.predict(X_meta)
    meta_acc = accuracy_score(y_meta, meta_pred)
    print(f"Meta-learner OOF准确率: {meta_acc:.4f}")
    
    # 保存meta-learner
    meta_learner_path = Path("model/stacking/meta_learner.pkl")
    meta_learner_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(meta_learner_path, 'wb') as f:
        pickle.dump(meta_learner, f)
    
    print(f"Meta-learner已保存: {meta_learner_path}")
    
    return meta_learner, X_meta, y_meta


def predict_stacking(base_models, meta_learner, dataloader, n_classes, device):
    """使用stacking模型进行预测"""
    # 获取所有基础模型的预测
    base_predictions = []
    
    for model_name in base_models:
        model = load_model(model_name, n_classes, device)
        probs, _ = get_predictions(model, dataloader, device)
        base_predictions.append(probs)
        del model
        torch.cuda.empty_cache() if device.type == 'cuda' else None
    
    # 堆叠特征
    X_meta = np.hstack(base_predictions)
    
    # Meta-learner预测
    meta_pred = meta_learner.predict(X_meta)
    meta_probs = meta_learner.predict_proba(X_meta)
    
    return meta_pred, meta_probs, base_predictions


def main():
    parser = argparse.ArgumentParser(description='训练Stacking模型')
    parser.add_argument('--dataset', type=str, required=True, help='数据集名称')
    parser.add_argument('--n_folds', type=int, default=5, help='交叉验证折数')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'mps', 'cpu'], help='设备')
    parser.add_argument('--base_models', type=str, nargs='+', default=BASE_MODELS,
                       help='基础模型列表')
    
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
    print(f"基础模型: {args.base_models}")
    
    # 加载数据集
    data_dir = Path(f"data/{args.dataset}")
    class_mapping_file = data_dir / "class_mapping.json"
    
    if not class_mapping_file.exists():
        raise FileNotFoundError(f"找不到class_mapping.json: {class_mapping_file}")
    
    dataset = CoordinateDataset(data_dir, class_mapping_file)
    n_classes = dataset.n_classes
    
    print(f"数据集大小: {len(dataset)}")
    print(f"类别数: {n_classes}")
    
    # 训练stacking模型
    meta_learner, X_meta, y_meta = train_stacking_cv(
        dataset, args.base_models, n_classes, device, args.n_folds
    )
    
    print("\n✅ Stacking模型训练完成！")


if __name__ == '__main__':
    main()
