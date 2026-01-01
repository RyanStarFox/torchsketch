# 上传模型到 Hugging Face 指南

## 快速开始

### 1. 安装依赖
```bash
pip install huggingface_hub
```

### 2. 登录 Hugging Face
```bash
huggingface-cli login
```
会提示你输入 token，可以在 https://huggingface.co/settings/tokens 获取

### 3. 上传模型
```bash
python upload_models_to_hf.py
```

## 手动上传（如果脚本不工作）

### 使用网页界面
1. 访问 https://huggingface.co/new
2. 创建新仓库（类型选择 "Model"）
3. 在仓库页面点击 "Files" -> "Add file" -> "Upload files"
4. 拖拽上传 `model/` 目录下的所有 `.pth` 文件

### 使用命令行
```bash
# 安装
pip install huggingface_hub

# 登录
huggingface-cli login

# 上传单个文件
huggingface-cli upload RyanStarFox/quickdraw-classifier-models \
    model/resnet18/best_model.pth \
    resnet18/best_model.pth

# 或上传整个目录
huggingface-cli upload RyanStarFox/quickdraw-classifier-models \
    model/ \
    . \
    --repo-type model
```

## 下载模型

上传后，其他人可以通过以下方式下载：

```python
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="RyanStarFox/quickdraw-classifier-models",
    filename="resnet18/best_model.pth",
    repo_type="model"
)
```

或在 README 中提供下载链接：
```
https://huggingface.co/RyanStarFox/quickdraw-classifier-models/resolve/main/resnet18/best_model.pth
```

