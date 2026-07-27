#!/bin/bash
# 取得腳本所在的目錄，確保在任何地方點擊都能切換到正確的工作目錄
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

echo "========================================="
echo "        Auto Compress Video 啟動中       "
echo "========================================="

# 檢查是否安裝了 python3
if ! command -v python3 &> /dev/null
then
    echo "錯誤：未找到 Python 3！請確保已安裝 Python 3 並且已加入環境變數。"
    read -p "請按 Enter 鍵關閉視窗..."
    exit
fi

# 透過統一入口執行 Compress 模式
python3 -m auto_reframe_core compress

echo "========================================="
echo "                處理完成                 "
echo "========================================="
read -p "請按 Enter 鍵關閉視窗..."
