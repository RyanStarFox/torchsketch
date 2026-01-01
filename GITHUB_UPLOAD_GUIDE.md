# GitHub上传指南

本指南说明如何将当前项目上传到非空的GitHub仓库：https://github.com/RyanStarFox/torchsketch

## 方法一：使用分支和Pull Request（推荐）

这是最安全的方法，不会直接修改主分支。

### 步骤1: 克隆远程仓库

```bash
# 克隆仓库
git clone https://github.com/RyanStarFox/torchsketch.git
cd torchsketch

# 查看当前分支
git branch -a
```

### 步骤2: 创建新分支

```bash
# 创建并切换到新分支
git checkout -b add-classifier-scripts

# 或者如果远程已有该分支，先拉取
# git fetch origin
# git checkout -b add-classifier-scripts origin/add-classifier-scripts
```

### 步骤3: 复制项目文件

将当前项目的以下文件复制到克隆的仓库根目录：

```bash
# 从当前项目目录复制文件
cp /path/to/classifier/train_model.py .
cp /path/to/classifier/stacking.py .
cp /path/to/classifier/test_model.py .
cp /path/to/classifier/plot_results.py .
```

或者手动复制以下文件：
- `train_model.py`
- `stacking.py`
- `test_model.py`
- `plot_results.py`

### 步骤4: 检查.gitignore

确保`.gitignore`文件已正确配置，忽略以下内容：
- 模型文件（`*.pth`, `*.pkl`）
- 数据文件（`data/`, `unprocessed_data/`）
- 结果文件（`result/`）
- 缓存文件（`__pycache__/`, `*.pyc`）

### 步骤5: 添加和提交文件

```bash
# 查看更改
git status

# 添加新文件
git add train_model.py stacking.py test_model.py plot_results.py

# 提交更改
git commit -m "添加分类器训练、stacking和测试脚本

- train_model.py: 支持多种模型类型的训练
- stacking.py: 使用8个基础模型进行stacking训练
- test_model.py: 数据预处理和模型测试
- plot_results.py: 结果可视化工具"
```

### 步骤6: 推送到远程

```bash
# 推送到远程分支
git push origin add-classifier-scripts
```

### 步骤7: 创建Pull Request

1. 访问 https://github.com/RyanStarFox/torchsketch
2. 点击 "Pull requests" 标签
3. 点击 "New pull request"
4. 选择 `add-classifier-scripts` 分支
5. 填写PR描述
6. 点击 "Create pull request"
7. 等待审核和合并

## 方法二：直接推送到master（不推荐）

⚠️ **警告**: 这种方法会直接修改主分支，建议只在有权限且确定的情况下使用。

```bash
# 克隆仓库
git clone https://github.com/RyanStarFox/torchsketch.git
cd torchsketch

# 复制文件（同上）

# 添加文件
git add train_model.py stacking.py test_model.py plot_results.py

# 提交
git commit -m "添加分类器脚本"

# 推送到master（需要权限）
git push origin master
```

## 方法三：Fork仓库（如果无法直接推送）

如果你没有直接推送权限，可以Fork仓库：

1. 访问 https://github.com/RyanStarFox/torchsketch
2. 点击右上角的 "Fork" 按钮
3. 在你的Fork中按照方法一或方法二操作
4. 从你的Fork创建Pull Request到原仓库

## 处理冲突

如果远程仓库已有同名文件，可能会遇到冲突：

```bash
# 拉取最新更改
git pull origin master

# 如果有冲突，手动解决冲突后
git add <解决冲突的文件>
git commit -m "解决冲突"
git push origin <分支名>
```

## 验证上传

上传后，访问以下URL验证文件是否存在：
- https://github.com/RyanStarFox/torchsketch/blob/master/train_model.py
- https://github.com/RyanStarFox/torchsketch/blob/master/stacking.py
- https://github.com/RyanStarFox/torchsketch/blob/master/test_model.py
- https://github.com/RyanStarFox/torchsketch/blob/master/plot_results.py

## 常见问题

### Q: 如何设置Git身份？

```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

### Q: 如何配置SSH密钥？

参考GitHub官方文档：https://docs.github.com/en/authentication/connecting-to-github-with-ssh

### Q: 如何撤销最后一次提交？

```bash
# 撤销提交但保留更改
git reset --soft HEAD~1

# 完全撤销提交和更改（谨慎使用）
git reset --hard HEAD~1
```

### Q: 如何查看提交历史？

```bash
git log --oneline
```

## 注意事项

1. **不要上传大文件**: 确保`.gitignore`正确配置，避免上传模型文件、数据文件等
2. **提交信息**: 使用清晰、描述性的提交信息
3. **代码质量**: 确保代码可以正常运行（虽然不需要测试，但应该没有明显的语法错误）
4. **权限**: 确保你有推送权限，或者使用Fork + PR的方式

## 快速命令总结

```bash
# 完整流程（方法一）
git clone https://github.com/RyanStarFox/torchsketch.git
cd torchsketch
git checkout -b add-classifier-scripts
# 复制文件...
git add train_model.py stacking.py test_model.py plot_results.py
git commit -m "添加分类器脚本"
git push origin add-classifier-scripts
# 然后在GitHub上创建PR
```

