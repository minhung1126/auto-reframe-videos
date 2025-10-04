# 專案分析：auto-reframe-videos (Google Colab 部署版)

這是一個 Python 專案，旨在自動將橫向影片 (16:9) 轉換為適合手機觀看的縱向影片 (9:16)。此版本特別為 **Google Colab** 環境進行了優化，讓任何使用者都能透過免費的雲端 GPU 資源，快速完成影片重構任務。

## 核心功能與特色

1.  **AI 人物偵測與智慧追蹤**：
    *   利用 AI 模型 (MobileNet-SSD 或 YOLOv4-tiny) 偵測影片中的「人物」，而非簡單地從畫面中央裁切。
    *   以偵測到的主要人物為中心，動態計算最佳裁切範圍，並透過平滑演算法確保鏡頭移動流暢自然，讓主角始終位於畫面焦點。

2.  **Google Colab GPU 加速**：
    *   **專為 Colab 優化**：腳本會優先利用 Colab 分配的 **NVIDIA GPU (Tesla T4, P100 等)** 進行硬體加速，無需手動設定。
    *   **端到端加速**：同時為 AI 運算 (CUDA) 和影片編碼 (NVENC) 提供加速，極大地縮短了處理時間，解決了此類任務最耗時的瓶頸。

3.  **自動化與易用性**：
    *   **一鍵環境設定**：在 Colab Notebook 中，只需執行幾個單元格即可自動安裝 FFmpeg 和所有 Python 依賴。
    *   **整合 Google Drive**：自動在您的 Google Drive 中建立 `input_videos`, `output_videos` 等資料夾，方便管理檔案。
    *   **自動模型下載**：首次執行時，腳本會自動下載所需的 AI 模型並存放在您的 Google Drive 中。
    *   **批次與並行處理**：您只需將影片放入 Google Drive 的 `input_videos` 資料夾，腳本會自動進行批次處理。同時，它會利用多核心 CPU 並行處理多個影片，最大化處理效率。

## 技術棧 (Technology Stack)

*   **執行環境**: Google Colab (Jupyter Notebook)
*   **檔案儲存**: Google Drive
*   **核心語言**: Python
*   **影片處理**: FFmpeg (透過 `apt-get` 安裝)
*   **AI 與影像**: OpenCV (`opencv-python`), NumPy
*   **其他工具**: Requests (下載模型), tqdm (進度條)

## 在 Colab 上的運作流程

1.  **前置設定**：
    *   在 Colab 中打開此專案的 Notebook (`.ipynb` 檔案)。
    *   在 `執行階段` -> `變更執行階段類型` 中，選擇 `GPU` 作為硬體加速器。
    *   執行第一個單元格，授權並掛載您的 Google Drive。腳本會在您的雲端硬碟根目錄下建立專案所需的資料夾。

2.  **安裝依賴**：
    *   執行安裝單元格，它會使用 `!apt-get install ffmpeg` 來安裝 FFmpeg，並使用 `!pip install -r requirements.txt` 安裝所有 Python 套件。

3.  **準備影片**：
    *   將您想要處理的橫向影片上傳到 Google Drive 中的 `input_videos` 資料夾。

4.  **執行處理**：
    *   執行主要的處理單元格。腳本會開始掃描 `input_videos` 資料夾，並對每個影片執行以下操作：
        *   **AI 分析**：逐幀讀取影片，使用 GPU 進行人物偵測，並計算出每一刻的最佳 9:16 裁切框。
        *   **硬體編碼**：將裁切後的影像即時傳送給 FFmpeg，使用 NVIDIA NVENC 編碼器高速產生無聲的暫存影片。
        *   **音訊合併**：將原始影片的音訊軌合併到裁切後的影片中。
        *   **最終輸出**：進行最終的壓縮與縮放 (1080x1920)，將成品儲存到 Google Drive 的 `output_videos` 資料夾中。

5.  **完成與檢視**：
    *   處理完成後，您可以直接在 Google Drive 的 `output_videos` 資料夾中找到所有轉換好的縱向影片。

總結來說，這個 Colab 版本讓強大的影片自動重構功能變得更加普及。使用者無需擁有昂貴的本地硬體，也無需處理複雜的環境設定，只需透過瀏覽器即可完成所有操作，完美解決了將傳統影片轉換為社群媒體格式的痛點。