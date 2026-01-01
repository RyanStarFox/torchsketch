#!/bin/bash
# Git 推送重试脚本 - 解决 HTTP 408 超时问题

cd "$(dirname "$0")"

echo "=== Git 推送配置优化 ==="

# 方案 1: 增加 HTTP 缓冲区大小
echo "1. 设置 Git HTTP 缓冲区为 500MB..."
git config http.postBuffer 524288000

# 方案 2: 设置低速度限制（允许慢速连接）
echo "2. 设置低速度限制..."
git config http.lowSpeedLimit 0
git config http.lowSpeedTime 999999

# 方案 3: 使用 HTTP/1.1（更稳定）
echo "3. 使用 HTTP/1.1..."
git config http.version HTTP/1.1

# 方案 4: 启用压缩
echo "4. 启用压缩..."
git config core.compression 9

echo ""
echo "=== 开始推送 ==="
echo "如果仍然失败，可以尝试："
echo "  - 使用 SSH: git remote set-url origin git@github.com:RyanStarFox/torchsketch.git"
echo "  - 分批推送: 先推送较早的提交"
echo ""

# 尝试推送
git push origin master

# 如果失败，提供其他选项
if [ $? -ne 0 ]; then
    echo ""
    echo "推送失败。其他解决方案："
    echo ""
    echo "方案 A: 使用 SSH（推荐）"
    echo "  git remote set-url origin git@github.com:RyanStarFox/torchsketch.git"
    echo "  git push origin master"
    echo ""
    echo "方案 B: 分批推送（先推送较早的提交）"
    echo "  git push origin 98bda21:master  # 推送第一个包含模型的提交"
    echo "  git push origin master          # 然后推送其余提交"
    echo ""
    echo "方案 C: 使用 Git LFS 管理大文件"
    echo "  git lfs install"
    echo "  git lfs track '*.pth'"
    echo "  git add .gitattributes"
    echo "  git commit -m 'Add LFS tracking'"
    echo "  git push origin master"
fi

