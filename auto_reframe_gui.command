#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

if ! command -v python3 &> /dev/null
then
    echo "錯誤：未找到 Python 3！請先安裝 Python 3。"
    read -p "請按 Enter 鍵關閉視窗..."
    exit 1
fi

python3 -m auto_reframe_core gui
