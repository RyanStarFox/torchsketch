#!/bin/bash
# 分批推送脚本

cd "$(dirname "$0")"

echo "=== 分批推送 Git 提交 ==="
echo ""

# 设置 Git 配置
git config http.postBuffer 524288000
git config http.lowSpeedLimit 0
git config http.lowSpeedTime 999999
git config http.version HTTP/1.1

# 获取所有需要推送的提交
COMMITS=($(git log --oneline origin/master..HEAD | awk '{print $1}'))

echo "需要推送 ${#COMMITS[@]} 个提交："
for i in "${!COMMITS[@]}"; do
    echo "  $((i+1)). ${COMMITS[$i]}"
done
echo ""

# 逐个推送（从最早到最新）
for i in "${!COMMITS[@]}"; do
    COMMIT=${COMMITS[$i]}
    echo "=========================================="
    echo "推送提交 $((i+1))/${#COMMITS[@]}: $COMMIT"
    echo "=========================================="
    
    # 推送单个提交
    if git push origin $COMMIT:master; then
        echo "✓ 提交 $COMMIT 推送成功"
        echo ""
    else
        echo "✗ 提交 $COMMIT 推送失败"
        echo ""
        echo "尝试推送所有剩余提交..."
        git push origin master
        exit 1
    fi
    
    # 等待一下，避免请求过快
    sleep 2
done

echo ""
echo "✅ 所有提交已成功推送！"
echo "现在推送最新状态..."
git push origin master


