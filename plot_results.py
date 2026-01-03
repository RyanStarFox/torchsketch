#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
作图函数
支持绘制训练曲线、模型对比、参数分析等多种可视化
"""

import os
import json
import argparse
import pickle
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端


def plot_training_curves(model_name, history_file=None, save_dir="figures"):
    """绘制训练曲线（loss和accuracy）"""
    if history_file is None:
        history_file = Path(f"model/{model_name}/training_history.pkl")
    
    if not history_file.exists():
        print(f"警告: 找不到训练历史文件 {history_file}")
        return
    
    with open(history_file, 'rb') as f:
        history = pickle.load(f)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Loss曲线
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title(f'{model_name} - Training and Validation Loss', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Accuracy曲线
    ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title(f'{model_name} - Training and Validation Accuracy', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    
    # 保存单独的图片
    plt.savefig(save_path / f"{model_name}_training_curves.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 分别保存loss和accuracy
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title(f'{model_name} - Loss', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path / "training_loss.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title(f'{model_name} - Accuracy', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path / "training_accuracy.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"训练曲线已保存到 {save_path}")


def plot_model_comparison(result_file, save_dir="figures"):
    """从result JSON文件绘制模型对比图"""
    with open(result_file, 'r') as f:
        result_data = json.load(f)
    
    statistics = result_data['statistics']
    
    models = list(statistics.keys())
    gt_accuracies = [statistics[m]['gt_accuracy'] for m in models]
    pred_accuracies = [statistics[m]['pred_accuracy'] for m in models]
    agreement_rates = [statistics[m]['agreement_rate'] for m in models]
    
    x = np.arange(len(models))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    bars1 = ax.bar(x - width, gt_accuracies, width, label='GT Accuracy', alpha=0.8)
    bars2 = ax.bar(x, pred_accuracies, width, label='Pred Accuracy', alpha=0.8)
    bars3 = ax.bar(x + width, agreement_rates, width, label='Agreement Rate', alpha=0.8)
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.1])
    
    # 添加数值标签
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    plt.savefig(save_path / "model_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"模型对比图已保存到 {save_path / 'model_comparison.png'}")


def plot_stacking_comparison(result_file, save_dir="figures"):
    """绘制stacking对比图"""
    with open(result_file, 'r') as f:
        result_data = json.load(f)
    
    statistics = result_data['statistics']
    
    # 分离基础模型和stacking
    base_models = [m for m in statistics.keys() if m != 'stacking']
    stacking_acc = statistics.get('stacking', {}).get('gt_accuracy', 0)
    
    base_accuracies = [statistics[m]['gt_accuracy'] for m in base_models]
    avg_base_acc = np.mean(base_accuracies)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 绘制基础模型准确率
    x_base = np.arange(len(base_models))
    bars = ax.bar(x_base, base_accuracies, alpha=0.7, color='skyblue', label='Base Models')
    
    # 绘制平均准确率
    ax.axhline(y=avg_base_acc, color='orange', linestyle='--', linewidth=2, 
               label=f'Average Base Model ({avg_base_acc:.3f})')
    
    # 绘制stacking准确率
    ax.axhline(y=stacking_acc, color='red', linestyle='-', linewidth=2, 
               label=f'Stacking ({stacking_acc:.3f})')
    
    ax.set_xlabel('Base Models', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Stacking vs Base Models Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x_base)
    ax.set_xticklabels(base_models, rotation=45, ha='right')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.1])
    
    # 添加数值标签
    for i, (bar, acc) in enumerate(zip(bars, base_accuracies)):
        ax.text(bar.get_x() + bar.get_width()/2., acc,
               f'{acc:.2f}',
               ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    plt.savefig(save_path / "stacking_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Stacking对比图已保存到 {save_path / 'stacking_comparison.png'}")


def plot_params_vs_accuracy(result_file, save_dir="figures"):
    """绘制参数数量vs准确率（如果可用）"""
    with open(result_file, 'r') as f:
        result_data = json.load(f)
    
    statistics = result_data['statistics']
    
    # 模型参数数量（近似值，需要从模型文件获取）
    # 这里使用默认值作为示例
    model_params = {
        'resnet18': 11.7e6,
        'mobilenetv2': 3.5e6,
        'inceptionv3': 27.2e6,
        'tcn': 0.5e6,
        'bilstm_30_epoch': 0.3e6,
        'bilstm_3_layer_30_epoch': 0.5e6,
        'gat': 0.2e6,
        'gcn': 0.2e6,
        'mgt': 1.0e6,
        'stacking': 0.1e6  # meta-learner参数很少
    }
    
    models = list(statistics.keys())
    accuracies = [statistics[m]['gt_accuracy'] for m in models]
    params = [model_params.get(m, 1e6) for m in models]  # 默认1M
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    scatter = ax.scatter(params, accuracies, s=100, alpha=0.6, c=range(len(models)), 
                        cmap='viridis', edgecolors='black', linewidth=1.5)
    
    # 添加模型标签
    for i, model in enumerate(models):
        ax.annotate(model, (params[i], accuracies[i]), 
                   xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax.set_xlabel('Number of Parameters', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Parameters vs Accuracy', fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    
    plt.colorbar(scatter, ax=ax, label='Model Index')
    plt.tight_layout()
    
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    plt.savefig(save_path / "params_vs_accuracy.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"参数vs准确率图已保存到 {save_path / 'params_vs_accuracy.png'}")


def plot_time_vs_accuracy(result_file, save_dir="figures"):
    """绘制训练时间vs准确率（如果可用）"""
    with open(result_file, 'r') as f:
        result_data = json.load(f)
    
    statistics = result_data['statistics']
    
    # 训练时间（近似值，需要从训练日志获取）
    # 这里使用默认值作为示例
    model_times = {
        'resnet18': 120,
        'mobilenetv2': 90,
        'inceptionv3': 180,
        'tcn': 60,
        'bilstm_30_epoch': 45,
        'bilstm_3_layer_30_epoch': 70,
        'gat': 80,
        'gcn': 70,
        'mgt': 100,
        'stacking': 200  # stacking需要训练所有基础模型
    }
    
    models = list(statistics.keys())
    accuracies = [statistics[m]['gt_accuracy'] for m in models]
    times = [model_times.get(m, 60) for m in models]  # 默认60分钟
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    scatter = ax.scatter(times, accuracies, s=100, alpha=0.6, c=range(len(models)), 
                        cmap='plasma', edgecolors='black', linewidth=1.5)
    
    # 添加模型标签
    for i, model in enumerate(models):
        ax.annotate(model, (times[i], accuracies[i]), 
                   xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax.set_xlabel('Training Time (minutes)', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Training Time vs Accuracy', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.colorbar(scatter, ax=ax, label='Model Index')
    plt.tight_layout()
    
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    plt.savefig(save_path / "time_vs_accuracy.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"时间vs准确率图已保存到 {save_path / 'time_vs_accuracy.png'}")


def plot_all_from_result(result_file, save_dir="figures"):
    """从result文件生成所有图表"""
    print(f"从 {result_file} 生成所有图表...")
    
    plot_model_comparison(result_file, save_dir)
    plot_stacking_comparison(result_file, save_dir)
    plot_params_vs_accuracy(result_file, save_dir)
    plot_time_vs_accuracy(result_file, save_dir)
    
    print(f"\n✅ 所有图表已保存到 {save_dir}")


def main():
    parser = argparse.ArgumentParser(description='绘制结果图表')
    parser.add_argument('--result_file', type=str, default=None,
                       help='结果JSON文件路径')
    parser.add_argument('--model_name', type=str, default=None,
                       help='模型名称（用于绘制训练曲线）')
    parser.add_argument('--save_dir', type=str, default='figures',
                       help='保存目录')
    parser.add_argument('--plot_type', type=str, default='all',
                       choices=['all', 'training', 'comparison', 'stacking', 
                               'params', 'time'],
                       help='绘图类型')
    
    args = parser.parse_args()
    
    if args.plot_type == 'training' and args.model_name:
        plot_training_curves(args.model_name, save_dir=args.save_dir)
    elif args.result_file:
        if args.plot_type == 'all':
            plot_all_from_result(args.result_file, args.save_dir)
        elif args.plot_type == 'comparison':
            plot_model_comparison(args.result_file, args.save_dir)
        elif args.plot_type == 'stacking':
            plot_stacking_comparison(args.result_file, args.save_dir)
        elif args.plot_type == 'params':
            plot_params_vs_accuracy(args.result_file, args.save_dir)
        elif args.plot_type == 'time':
            plot_time_vs_accuracy(args.result_file, args.save_dir)
    else:
        print("错误: 需要提供 --result_file 或 --model_name")
        parser.print_help()


if __name__ == '__main__':
    main()


