# Auto Reframe Video v2.0

將橫向影片透過 FFmpeg 自動裁切、轉換為手機直式影片（9:16），支援硬體加速、平行處理、單次解碼多路輸出，並可在上下黑邊疊加自訂文字。

---

## 快速開始

### 一般執行 (Windows / macOS / Linux)

1. 將影片放入 `input/` 資料夾
2. （選填）編輯 `top_text.txt` 與 `bottom_text.txt`
3. 執行腳本：

```bash
python auto_reframe.py
```

### macOS 一鍵執行

專案內提供了 macOS 專用的啟動腳本：
- `auto_reframe.command`
- `auto_compress.command`

**如何使用：**
只需在 Finder 中雙擊這些 `.command` 檔案，系統就會自動開啟終端機並執行對應的程式。

**⚠️ 首次執行注意事項（權限問題）：**
如果雙擊檔案時出現「沒有權限」或「無法執行」的錯誤（常見於從 Windows 傳送檔案到 macOS 時遺失執行權限），請進行以下設定：
1. 打開 macOS 的「終端機」（Terminal）。
2. 輸入 `chmod +x `（注意 `+x` 後面有一個空白）。
3. 將無法執行的 `.command` 檔案從 Finder 拖曳到終端機視窗內（會自動輸入路徑）。
4. 按下 Enter 鍵。設定完成後即可正常雙擊執行。

輸出會依裁切比例與解析度分類存放至 `output/` 資料夾。

---

## 設定值說明（`ReframeConfig`）

所有設定值均在 `auto_reframe.py` 的 `ReframeConfig` 類別中定義。

### 輸入 / 輸出

| 設定值 | 預設值 | 說明 |
|---|---|---|
| `input_dir` | `"input"` | 來源影片資料夾路徑 |
| `output_dir` | `"output"` | 輸出影片資料夾路徑 |

### 輸出目標（`targets`）

| 設定值 | 預設值 | 說明 |
|---|---|---|
| `targets` | `[{'ratio': (4, 5), 'resolution': 'source', 'vcodec': h265}, {'ratio': (4, 5), 'resolution': '1080p', 'vcodec': h264}]` | 多目標輸出設定。每個目標需指定裁切比例 `ratio`、解析度上限 `resolution`、編碼器 `vcodec` |
| `final_ratio` | `(9, 16)` | 最終輸出影片的比例（寬:高）。不足的部分以黑邊補齊 |

> 例：`targets` 中可同時包含 `(4,5)` 與 `(1,1)`，每種比例都會輸出對應解析度與 codec 的版本。

### 上方文字設定

| 設定值 | 預設值 | 說明 |
|---|---|---|
| `top_text_file` | `"top_text.txt"` | 上方文字的來源檔案路徑 |
| `top_font_size` | `48` | 上方文字的基準字型大小（px，以 FHD 1920px 高為準，其他解析度等比縮放） |

### 下方文字設定

| 設定值 | 預設值 | 說明 |
|---|---|---|
| `bottom_text_file` | `"bottom_text.txt"` | 下方文字的來源檔案路徑 |
| `bottom_font_size` | `24` | 下方文字的基準字型大小（px，以 FHD 1920px 高為準，其他解析度等比縮放） |

### 字型設定

| 設定值 | 預設值 | 說明 |
|---|---|---|
| `font_path` | `"fonts/NotoSerifTC.ttf"` | 字型檔案路徑（相對於腳本位置） |
| `font_color` | `"white"` | 文字顏色，同時作為描邊顏色。接受 FFmpeg 顏色名稱或 `#RRGGBB` 格式 |
| `text_margin` | `20` | 文字與影片邊緣的間距（px，以 FHD 為準，其他解析度等比縮放） |
| `top_text_line_spacing_ratio` | `1.2` | 上方文字區的多行文字行距倍數（`1.0` = 無額外間距） |
| `bottom_text_line_spacing_ratio` | `1.2` | 下方文字區的多行文字行距倍數（`1.0` = 無額外間距） |

### 系統與平行化

| 設定值 | 預設值 | 說明 |
|---|---|---|
| `ffmpeg_path` | `"ffmpeg"` | FFmpeg 執行檔路徑。若未加入 PATH 請填寫完整路徑，例如 `"C:/ffmpeg/bin/ffmpeg.exe"` |
| `ffprobe_path` | `"ffprobe"` | FFprobe 執行檔路徑，同上 |
| `video_extensions` | `{".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm", ".ts", ".m4v"}` | 腳本會掃描的影片副檔名集合 |
| `skip_existing` | `True` | `True` 時若輸出檔案已存在則跳過，設為 `False` 可強制重新轉換 |
| `max_workers` | `0` | 同時平行處理的影片數量。`0` 代表自動判斷；目前上限為 macOS=4、其他平台=8 |
| `debug` | `False` | `True` 時會將 FFmpeg 的完整輸出記錄至腳本目錄下的 `ffmpeg_debug_<檔名>_<比例>.log` |

---

## 文字模板

### `top_text.txt`（上方文字）

顯示在影片**上方黑邊**中，支援多行。每一行對應一列文字，由上往下排列。**留空檔案則不顯示任何文字。**

```text {.line-numbers}
1: 2026.03.22
2: 富邦悍將 vs 統一獅 @ 亞太主球場
```

### `bottom_text.txt`（下方文字）

顯示在影片**下方黑邊**中，支援多行。排列方式與上方相同，從黑邊頂端起向下排列。**留空檔案則不顯示任何文字。**

```text {.line-numbers}
1: ©example.credit
```

### 注意事項

- 檔案編碼請使用 **UTF-8**（支援中文、日文等多位元組字元）
- 若文字中包含 `:` 或 `%` 符號，腳本會自動轉義，**不需手動處理**
- 若文字中包含單引號 `'`，腳本同樣會自動轉義
- 若檔案不存在，腳本啟動時會自動建立空白檔案

---

## 輸出目錄結構

```
output/
├── 4x5_FHD_h265/
│   └── video_name_4x5_FHD_h265.mp4
├── 4x5_FHD_h264/
│   └── video_name_4x5_FHD_h264.mp4
└── 1x1_HD_h265/
    └── video_name_1x1_HD_h265.mp4
```

---

## 硬體加速

腳本啟動時會自動偵測並優先使用以下編碼器：

| 優先順序 | 編碼器 | 加速方式 | 適用硬體 |
|---|---|---|---|
| 1 | `h264_nvenc` | CUDA | NVIDIA GPU |
| 2 | `h264_amf` | D3D11VA | AMD GPU |
| 3 | `h264_qsv` | QSV | Intel GPU |
| 4 | `libx264` | 軟體編碼 | 全平台（備援） |

---

## 依賴套件

```bash
pip install tqdm
```

> `tqdm` 為選擇性依賴，未安裝時會回退至傳統文字進度輸出。FFmpeg 與 FFprobe 需另行安裝並加入系統 PATH。
