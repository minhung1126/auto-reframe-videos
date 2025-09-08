# 智慧影片垂直裁剪工具 (AI Vertical Video Cropper)

這是一個 Python 腳本，可以自動偵測橫向影片中的人物，並將其智慧裁剪為 9:16 的直向影片，非常適合用於製作社交媒體短片 (如 YouTube Shorts, TikTok, Instagram Reels)。

## ✨ 主要功能

- **全自動硬體加速**：程式會自動偵測您電腦中最佳的影片編碼器，並依 **NVIDIA > AMD > Intel > CPU** 的優先順序啟用，實現最高效能。
- **AI 人物偵測**：使用 MobileNet SSD 模型準確辨識影片中的人物，此部分也可由 CUDA 加速。
- **智慧追蹤裁剪**：鏡頭會平滑地跟隨畫面中主要人物的中心位置。
- **進度條顯示**：為每個影片處理過程提供即時進度條。
- **自動模型下載**：首次執行時會自動下載所需的 AI 模型。

## ⚙️ 環境需求與安裝

### 1. 安裝 FFmpeg (必須)

本腳本依賴 FFmpeg 來進行影片編碼，以確保最佳的效能與穩定性。

- **下載**：請前往 [FFmpeg 官網](https://ffmpeg.org/download.html) 或 [BtbN 的 Github 版本](https://github.com/BtbN/FFmpeg-Builds/releases) (推薦，提供已編譯好的版本) 下載。
- **安裝**：下載後解壓縮，並將其 `bin` 資料夾的路徑（例如 `C:\ffmpeg\bin`）新增到您 Windows 的**系統環境變數 `Path`** 中。
- **驗證**：開啟一個新的命令提示字元視窗，輸入 `ffmpeg -version`。如果能看到版本資訊，代表安裝成功。

### 2. 安裝 Python 套件

您需要安裝 Python 3.7 或更高版本。

在終端機或命令提示字元中執行以下指令來安裝必要的 Python 套件：

```bash
pip install -r requirements.txt
```

### 3. (選項) AI 偵測的 GPU 加速

如果您有支援 CUDA 的 NVIDIA 顯示卡，可以加速 AI 物件偵測的部分。

- **安裝 CUDA Toolkit & cuDNN**：確保您已安裝與驅動程式匹配的 [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive) 和 [cuDNN](https://developer.nvidia.com/cudnn)。
- **安裝 CUDA 版的 OpenCV**：`pip` 預設版不支援 CUDA。您需要解除安裝現有版本，並安裝一個社群編譯的 CUDA 版本，或自行編譯。

## 🚀 如何使用

1.  將您想要處理的橫向影片 (`.mp4`, `.mov` 等) 放入 `input_videos` 資料夾。
2.  執行主程式：
    ```bash
    python autocrop_videos.py
    ```
3.  程式會自動偵測並使用最佳的硬體加速方案，然後開始處理影片。
4.  處理完成的 `.mp4` 影片將會儲存在 `output_videos` 資料夾中。

## 📁 資料夾結構

```
.
├── README.md             # 本說明檔案
├── autocrop_videos.py    # 主要的執行腳本
├── input_videos/         # 存放您的原始橫向影片
├── output_videos/        # 存放處理完成的直向影片
└── models/               # 自動下載並存放 AI 模型的地方
```