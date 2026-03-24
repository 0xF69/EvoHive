#!/bin/bash
# EvoHive 一键安装运行脚本
# 使用方法: bash run.sh

set -e

echo ""
echo "=============================="
echo "  EvoHive 一键安装启动"
echo "=============================="
echo ""

# 检查Python
if ! command -v python3 &> /dev/null; then
    echo "[错误] 没有找到 Python3。"
    echo ""
    echo "请先安装Python:"
    echo "  Mac: 打开终端，输入 brew install python3"
    echo "  Windows: 去 https://www.python.org/downloads/ 下载安装"
    echo "  安装时勾选 'Add Python to PATH'"
    echo ""
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1)
echo "[OK] 找到 $PYTHON_VERSION"

# 检查API Key
if [ -z "$DEEPSEEK_API_KEY" ]; then
    echo ""
    echo "[需要设置] 你还没有设置 DeepSeek API Key。"
    echo ""
    echo "请输入你的 DeepSeek API Key（在 https://platform.deepseek.com/api_keys 获取）:"
    read -r api_key
    export DEEPSEEK_API_KEY="$api_key"
    echo "[OK] API Key 已设置"
fi

# 安装项目
echo ""
echo "[安装中] 正在安装 EvoHive..."
cd "$(dirname "$0")"
pip3 install -e . --quiet 2>&1 | tail -1
echo "[OK] 安装完成"

# 运行
echo ""
echo "=============================="
echo "  开始运行进化!"
echo "=============================="
echo ""

evohive -p "给一个面向独立开发者的AI代码审查工具设计定价策略" -n 10 -g 3
