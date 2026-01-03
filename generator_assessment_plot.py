#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成器评估对比图
对比不同生成器（sketch pic2seq, sketchRNN）在不同数据量下的stacking模型表现
"""

import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import numpy as np


def load_result_data(result_file):
    """加载结果JSON文件"""
    with open(result_file, 'r') as f:
        data = json.load(f)
    return data


def extract_stacking_metrics(data):
    """提取stacking模型的指标"""
    statistics = data.get('statistics', {})
    stacking_stats = statistics.get('stacking', {})
    
    # 注意：JSON中使用的是agreement_rate，但用户可能想要agreement_accuracy
    # 如果存在agreement_accuracy就用它，否则用agreement_rate
    agreement_accuracy = stacking_stats.get('agreement_accuracy', 
                                            stacking_stats.get('agreement_rate', 0))
    agreement_and_correct_rate = stacking_stats.get('agreement_and_correct_rate', 0)
    
    return agreement_accuracy, agreement_and_correct_rate


def plot_generator_assessment(result_files, titles, save_path="figures/generator_assessment.png"):
    """
    绘制生成器评估对比图
    
    Args:
        result_files: 结果文件路径列表
        titles: 对应的标题列表
        save_path: 保存路径
    """
    # 加载数据
    agreement_accuracies = []
    agreement_and_correct_rates = []
    
    for result_file in result_files:
        data = load_result_data(result_file)
        agreement_acc, agreement_correct = extract_stacking_metrics(data)
        agreement_accuracies.append(agreement_acc)
        agreement_and_correct_rates.append(agreement_correct)
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # 设置柱状图位置
    x = np.arange(len(titles))
    width = 0.35
    
    # 绘制柱状图
    bars1 = ax.bar(x - width/2, agreement_accuracies, width, 
                   label='Agreement Rate', alpha=0.8, color='#2E86AB')
    bars2 = ax.bar(x + width/2, agreement_and_correct_rates, width,
                   label='Agreement and Correct Rate', alpha=0.8, color='#A23B72')
    
    # 设置标签和标题
    ax.set_xlabel('Generator', fontsize=14, fontweight='bold')
    ax.set_ylabel('Rate', fontsize=14, fontweight='bold')
    ax.set_title('Generator Assessment: Stacking Model Performance', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(titles, fontsize=12)
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, max(max(agreement_accuracies), max(agreement_and_correct_rates)) * 1.15])
    
    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   '{:.3f}'.format(height),
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ 图表已保存到: {}".format(save_path))
    print("\n数据摘要:")
    for title, acc, correct in zip(titles, agreement_accuracies, agreement_and_correct_rates):
        print("  {}:".format(title))
        print("    Agreement Rate: {:.4f}".format(acc))
        print("    Agreement and Correct Rate: {:.4f}".format(correct))


def main():
    parser = argparse.ArgumentParser(description='生成器评估对比图')
    parser.add_argument('--result_dir', type=str, default='result',
                       help='结果文件目录')
    parser.add_argument('--save_path', type=str, default='figures/generator_assessment.png',
                       help='保存路径')
    
    args = parser.parse_args()
    
    result_dir = Path(args.result_dir)
    
    # 定义结果文件和对应的标题
    result_configs = [
        {
            'file': 'sampling_1-compare-20251231_195627.json',
            'title': 'sketch pic2seq 62k'
        },
        {
            'file': 'sampling_dir_10k_m2-compare-20251231_195852.json',
            'title': 'sketchRNN 10k'
        },
        {
            'file': 'sampling_dir_62k_m1-compare-20251231_200910.json',
            'title': 'sketchRNN 62k'
        },
        {
            'file': 'sampling_dir_390k_m1-compare-20251231_201149.json',
            'title': 'sketchRNN 390k'
        }
    ]
    
    # 检查文件是否存在
    result_files = []
    titles = []
    
    for config in result_configs:
        file_path = result_dir / config['file']
        if file_path.exists():
            result_files.append(str(file_path))
            titles.append(config['title'])
        else:
            print("⚠️  警告: 找不到文件 {}".format(file_path))
    
    if len(result_files) == 0:
        print("❌ 错误: 没有找到任何结果文件")
        return
    
    print("找到 {} 个结果文件".format(len(result_files)))
    
    # 绘制图表
    plot_generator_assessment(result_files, titles, args.save_path)


if __name__ == '__main__':
    main()

