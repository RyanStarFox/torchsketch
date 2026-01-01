#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试模型脚本
1. 数据预处理：将unprocessed_data处理成data格式
2. 在stacking模型上测试
"""

import os
import json
import argparse
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import shutil

import torch
from torch.utils.data import DataLoader
from train_model import CoordinateDataset, get_model
from stacking import load_model, predict_stacking, BASE_MODELS


def preprocess_data(unprocessed_dir, output_dir, dataset_name):
    """
    将unprocessed_data处理成data格式
    从.npz文件提取坐标数据，保存为.npy格式
    """
    unprocessed_path = Path(unprocessed_dir)
    output_path = Path(output_dir) / dataset_name
    coordinate_dir = output_path / "coordinate_files"
    picture_dir = output_path / "picture_files"
    
    coordinate_dir.mkdir(parents=True, exist_ok=True)
    picture_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"预处理数据: {unprocessed_dir} -> {output_path}")
    
    # 收集所有类别
    classes = []
    label_to_idx = {}
    sampling_to_quickdraw = {}
    
    # 遍历unprocessed_data目录
    for class_dir in sorted(unprocessed_path.iterdir()):
        if not class_dir.is_dir():
            continue
        
        class_name = class_dir.name
        classes.append(class_name)
        label_to_idx[class_name] = len(classes) - 1
        
        # 创建对应的输出目录
        class_coord_dir = coordinate_dir / class_name
        class_pic_dir = picture_dir / class_name
        class_coord_dir.mkdir(exist_ok=True)
        class_pic_dir.mkdir(exist_ok=True)
        
        print(f"  处理类别: {class_name}")
        
        # 处理该类别下的所有.npz文件
        npz_files = sorted(class_dir.glob("*.npz"))
        count = 0
        
        for npz_file in npz_files:
            try:
                # 加载npz文件
                data = np.load(npz_file, allow_pickle=True)
                
                # 提取坐标数据
                if isinstance(data, np.lib.npyio.NpzFile):
                    # 尝试常见的键名
                    keys = list(data.keys())
                    if 'drawing' in keys:
                        drawing = data['drawing']
                        # drawing通常是list of strokes，每个stroke是[n_points, 2]
                        if isinstance(drawing, (list, np.ndarray)):
                            # 合并所有strokes
                            if isinstance(drawing, list):
                                coordinates = np.vstack(drawing) if len(drawing) > 0 else np.array([[0, 0]])
                            else:
                                coordinates = drawing
                        else:
                            coordinates = np.array([[0, 0]])
                    elif len(keys) > 0:
                        # 使用第一个键
                        coordinates = data[keys[0]]
                        if isinstance(coordinates, (list, np.ndarray)):
                            if isinstance(coordinates, list):
                                coordinates = np.vstack(coordinates) if len(coordinates) > 0 else np.array([[0, 0]])
                            else:
                                coordinates = coordinates
                        else:
                            coordinates = np.array([[0, 0]])
                    else:
                        coordinates = np.array([[0, 0]])
                else:
                    coordinates = np.array(data)
                
                # 确保是2D数组 [n_points, 2]
                if len(coordinates.shape) == 1:
                    # 如果是1D，尝试reshape
                    n = len(coordinates) // 2
                    coordinates = coordinates[:n*2].reshape(n, 2)
                elif len(coordinates.shape) == 2 and coordinates.shape[1] > 2:
                    # 如果列数>2，只取前两列（x, y）
                    coordinates = coordinates[:, :2]
                elif len(coordinates.shape) > 2:
                    # 如果是3D或更高，flatten后reshape
                    coordinates = coordinates.flatten()[:1000].reshape(-1, 2)
                
                # 确保至少有一个点
                if len(coordinates) == 0:
                    coordinates = np.array([[0, 0]])
                
                # 保存为.npy文件
                output_file = class_coord_dir / f"{class_name}_{count}.npy"
                np.save(output_file, coordinates)
                count += 1
                
            except Exception as e:
                print(f"    警告: 处理 {npz_file} 时出错: {e}")
                continue
        
        print(f"    处理了 {count} 个文件")
    
    # 创建class_mapping.json
    class_mapping = {
        "dataset_name": dataset_name,
        "n_classes": len(classes),
        "label_to_idx": label_to_idx,
        "sampling_to_quickdraw_mapping": sampling_to_quickdraw  # 如果需要可以后续填充
    }
    
    mapping_file = output_path / "class_mapping.json"
    with open(mapping_file, 'w') as f:
        json.dump(class_mapping, f, indent=2)
    
    print(f"\n✅ 数据预处理完成！")
    print(f"  输出目录: {output_path}")
    print(f"  类别数: {len(classes)}")
    print(f"  类别映射: {mapping_file}")
    
    return output_path


def test_stacking(dataset_name, n_samples=None, device='auto'):
    """在stacking模型上测试"""
    # 设置设备
    if device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(device)
    
    print(f"\n使用设备: {device}")
    
    # 加载数据集
    data_dir = Path(f"data/{dataset_name}")
    class_mapping_file = data_dir / "class_mapping.json"
    
    if not class_mapping_file.exists():
        raise FileNotFoundError(f"找不到class_mapping.json: {class_mapping_file}")
    
    with open(class_mapping_file, 'r') as f:
        class_mapping = json.load(f)
    
    dataset = CoordinateDataset(data_dir, class_mapping_file)
    n_classes = dataset.n_classes
    label_to_idx = class_mapping['label_to_idx']
    idx_to_label = {v: k for k, v in label_to_idx.items()}
    
    # 如果指定了采样数量，随机采样
    if n_samples and n_samples < len(dataset):
        import random
        indices = random.sample(range(len(dataset)), n_samples)
        from torch.utils.data import Subset
        dataset = Subset(dataset, indices)
        print(f"采样 {n_samples} 个样本进行测试")
    
    test_loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # 加载meta-learner
    meta_learner_path = Path("model/stacking/meta_learner.pkl")
    if not meta_learner_path.exists():
        raise FileNotFoundError(f"找不到meta-learner: {meta_learner_path}")
    
    with open(meta_learner_path, 'rb') as f:
        meta_learner = pickle.load(f)
    
    print("加载meta-learner成功")
    
    # 获取所有样本的真实标签
    all_true_labels = []
    all_samples = []
    
    for i in range(len(dataset)):
        if isinstance(dataset, torch.utils.data.Subset):
            _, label = dataset.dataset[dataset.indices[i]]
            sample_info = dataset.dataset.samples[dataset.indices[i]]
        else:
            _, label = dataset[i]
            sample_info = dataset.samples[i]
        all_true_labels.append(label)
        all_samples.append(sample_info)
    
    # 使用stacking模型预测
    print("\n进行预测...")
    meta_pred, meta_probs, base_predictions = predict_stacking(
        BASE_MODELS, meta_learner, test_loader, n_classes, device
    )
    
    # 计算统计信息
    all_true_labels = np.array(all_true_labels)
    
    statistics = {}
    detailed_results = []
    
    # 计算每个基础模型的统计
    for i, model_name in enumerate(BASE_MODELS):
        base_probs = base_predictions[i]
        base_pred = np.argmax(base_probs, axis=1)
        
        # 计算准确率
        gt_accuracy = np.mean(base_pred == all_true_labels)
        pred_accuracy = np.mean(base_pred == meta_pred)
        agreement_rate = np.mean(base_pred == meta_pred)
        agreement_and_correct = np.mean((base_pred == meta_pred) & (base_pred == all_true_labels))
        
        statistics[model_name] = {
            "gt_accuracy": float(gt_accuracy),
            "pred_accuracy": float(pred_accuracy),
            "agreement_rate": float(agreement_rate),
            "agreement_and_correct_rate": float(agreement_and_correct)
        }
    
    # Stacking模型统计
    stacking_gt_acc = np.mean(meta_pred == all_true_labels)
    statistics["stacking"] = {
        "gt_accuracy": float(stacking_gt_acc),
        "pred_accuracy": float(stacking_gt_acc),  # 与ground truth一致
        "agreement_rate": 1.0,  # 自己与自己一致
        "agreement_and_correct_rate": float(stacking_gt_acc)
    }
    
    # 详细结果
    for idx, (sample_path, true_label_idx, true_label_name) in enumerate(all_samples):
        sample_id = Path(sample_path).stem
        
        result = {
            "sample_id": sample_id,
            "true_label": true_label_name,
            "true_label_idx": int(true_label_idx),
            "gt_predictions": {}
        }
        
        # 每个基础模型的预测
        for i, model_name in enumerate(BASE_MODELS):
            base_probs = base_predictions[i][idx]
            base_pred_idx = int(np.argmax(base_probs))
            
            result["gt_predictions"][model_name] = {
                "predicted_idx": base_pred_idx,
                "predicted_label": idx_to_label.get(base_pred_idx, "unknown"),
                "probs": base_probs.tolist()
            }
        
        # Stacking预测
        stacking_probs = meta_probs[idx]
        stacking_pred_idx = int(meta_pred[idx])
        
        result["gt_predictions"]["stacking"] = {
            "predicted_idx": stacking_pred_idx,
            "predicted_label": idx_to_label.get(stacking_pred_idx, "unknown"),
            "probs": stacking_probs.tolist()
        }
        
        detailed_results.append(result)
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = Path(f"result/{dataset_name}-compare-{timestamp}.json")
    result_file.parent.mkdir(parents=True, exist_ok=True)
    
    result_data = {
        "dataset_name": dataset_name,
        "data_root": f"data/{dataset_name}",
        "device": str(device),
        "timestamp": timestamp,
        "n_samples": len(detailed_results),
        "statistics": statistics,
        "detailed_results": detailed_results
    }
    
    with open(result_file, 'w') as f:
        json.dump(result_data, f, indent=2)
    
    print(f"\n✅ 测试完成！")
    print(f"结果保存在: {result_file}")
    print(f"\n统计信息:")
    for model_name, stats in statistics.items():
        print(f"  {model_name}:")
        print(f"    GT准确率: {stats['gt_accuracy']:.4f}")
        print(f"    预测准确率: {stats['pred_accuracy']:.4f}")
        print(f"    一致率: {stats['agreement_rate']:.4f}")
    
    return result_file


def main():
    parser = argparse.ArgumentParser(description='测试模型')
    parser.add_argument('--dataset', type=str, required=True, help='数据集名称')
    parser.add_argument('--unprocessed_dir', type=str, default=None,
                       help='未处理数据目录（如果提供，将先进行预处理）')
    parser.add_argument('--n_samples', type=int, default=None,
                       help='测试样本数量（None表示使用全部）')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'mps', 'cpu'], help='设备')
    parser.add_argument('--skip_preprocess', action='store_true',
                       help='跳过预处理，直接使用已有的data目录')
    
    args = parser.parse_args()
    
    # 数据预处理
    if args.unprocessed_dir and not args.skip_preprocess:
        print("=" * 60)
        print("步骤1: 数据预处理")
        print("=" * 60)
        preprocess_data(args.unprocessed_dir, "data", args.dataset)
    
    # 测试
    print("\n" + "=" * 60)
    print("步骤2: 模型测试")
    print("=" * 60)
    test_stacking(args.dataset, args.n_samples, args.device)


if __name__ == '__main__':
    main()
