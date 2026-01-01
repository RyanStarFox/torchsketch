#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
上传模型到 Hugging Face Hub
使用方法：
1. pip install huggingface_hub
2. huggingface-cli login  # 登录你的 Hugging Face 账号
3. python upload_models_to_hf.py
"""

import os
from pathlib import Path
from huggingface_hub import HfApi, upload_file

# 配置
REPO_ID = "RyanStarFox/quickdraw-classifier-models"  # 改成你的用户名/仓库名
MODEL_DIR = Path("model")

def upload_models():
    """上传所有模型文件到 Hugging Face"""
    api = HfApi()
    
    # 创建仓库（如果不存在）
    try:
        api.create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True)
        print(f"✓ 仓库已创建/存在: {REPO_ID}")
    except Exception as e:
        print(f"⚠ 创建仓库时出错: {e}")
    
    # 查找所有模型文件
    model_files = list(MODEL_DIR.rglob("*.pth"))
    print(f"\n找到 {len(model_files)} 个模型文件\n")
    
    for model_file in model_files:
        # 计算相对路径作为仓库中的路径
        relative_path = model_file.relative_to(MODEL_DIR)
        repo_path = str(relative_path)
        
        print(f"上传: {model_file.name}")
        print(f"  路径: {repo_path}")
        
        try:
            upload_file(
                path_or_fileobj=str(model_file),
                path_in_repo=repo_path,
                repo_id=REPO_ID,
                repo_type="model",
            )
            print(f"  ✓ 上传成功\n")
        except Exception as e:
            print(f"  ✗ 上传失败: {e}\n")
    
    print(f"\n✅ 所有模型已上传到: https://huggingface.co/{REPO_ID}")

if __name__ == "__main__":
    upload_models()

