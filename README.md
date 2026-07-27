# Auto Reframe Videos

將橫向影片透過 FFmpeg 自動裁切、轉換為手機直式影片（9:16），也可只做等比例壓縮。支援跨平台 GUI、PNG 浮水印、硬體加速、平行處理、單次解碼多路輸出，並可在上下黑邊疊加自訂文字。

---

## 快速開始

### 統一入口（建議）

1. 將影片放入 `input/`。
2. 如需浮水印，將一個或多個 PNG 放入小寫的 `watermark/`。
3. 透過統一入口啟動 GUI：

```bash
python -m auto_reframe_core
```

同一入口也可執行個別模式：

```bash
python -m auto_reframe_core reframe
python -m auto_reframe_core compress
python -m auto_reframe_core --help
```

也可直接雙擊平台啟動器；兩者都會轉送至上述統一入口：

- Windows：`run.bat`
- macOS：`run.command`

GUI 的輸入與輸出路徑固定為專案內的 `input/` 與 `output/`，不可編輯。程式每次啟動都會自動建立缺少的 `input/`、`output/` 與 `watermark/`。

GUI 以主頁簽區分功能：

- **比例裁切／直式重製**：維持固定的「上方文字／中央影片／下方文字」三層 9:16 版面。
- **影片壓縮／縮小解析度**：保留來源比例，只縮小與轉碼。
- **關於／更新**：檢查 GitHub 最新正式 Release，下載、驗證並安裝更新。

「裁切重製」內另有「輸出設定／編輯上下文／文字樣式」子頁簽，只有需要上下方文字的裁切工作會顯示上下文編輯器。「編輯上下文」提供 `©` 一鍵複製，以及直接插入上方或下方文字的按鈕。

每個輸出目標可獨立選擇：

- 解析度上限：原始、4K、2K / 1440p、Full HD 1080p、HD 720p、480p、360p。
- Codec：H.264 / AVC、H.265 / HEVC。
- 重製中央影片比例：4:5、1:1、4:3、16:9，或直接輸入自訂 `寬:高`。

可加入多列目標，一次輸出多個比例／解析度／codec 組合。解析度是「不大於指定值的最大標準解析度」，不會為了達到 4K 或 1080p 而插值放大。

### GUI 設定檔

- `config.json.example`：會加入 Git 的完整預設設定，也是 GUI 的預設值來源。
- `config.json`：按下 GUI 的「儲存設定」後產生，啟動時會自動載入。
- `config.json` 已加入 `.gitignore`，不會提交個人的工作設定。
- 按下「還原預設」會刪除 `config.json`，重新使用 `config.json.example`。

輸入／輸出路徑刻意不放進設定檔，永遠固定為專案內的 `input/` 與 `output/`。

### 軟體更新

GUI 的「關於／更新」頁籤可手動檢查與安裝更新。更新程式只接受
`minhung1126/auto-reframe-videos` 的最新正式 GitHub Release，並驗證 GitHub
Releases API 提供的 SHA-256 digest 與 Release 內的逐檔 manifest。

安裝前會在 `.update-backups/` 備份舊程式，且不會覆寫：

- `config.json`
- `input/`、`output/`
- `top_text.txt`、`bottom_text.txt`
- `watermark/`

成功更新後，下載與 staging 暫存檔會自動清除；`.update-backups/` 會保留，
方便需要時手動回復舊程式。

如果程式位於 Git 工作目錄，為避免破壞開發中的 tracked files，自動安裝會停用，
請改用 `git pull`。從 Release ZIP 解壓縮的版本才會啟用一鍵安裝。

### macOS 一鍵執行

專案內提供單一 macOS 啟動腳本 `run.command`，會開啟 GUI。

**如何使用：**
只需在 Finder 中雙擊 `run.command`，系統就會自動開啟終端機並啟動 GUI。

**⚠️ 首次執行注意事項（權限問題）：**
如果雙擊檔案時出現「沒有權限」或「無法執行」的錯誤（常見於從 Windows 傳送檔案到 macOS 時遺失執行權限），請進行以下設定：
1. 打開 macOS 的「終端機」（Terminal）。
2. 輸入 `chmod +x `（注意 `+x` 後面有一個空白）。
3. 將無法執行的 `.command` 檔案從 Finder 拖曳到終端機視窗內（會自動輸入路徑）。
4. 按下 Enter 鍵。設定完成後即可正常雙擊執行。

輸出會依裁切比例與解析度分類存放至 `output/` 資料夾。

---

## 專案結構

```text
auto_reframe_core/       # 所有應用程式實作與統一入口
run.bat                  # Windows GUI 啟動器
run.command              # macOS GUI 啟動器
config.json.example      # GUI 預設設定，納入 Git
tests/                   # 行為守衛與入口測試
scripts/                 # Release 建置與驗證
fonts/                   # 內建字型與授權
input/                   # 固定來源影片目錄（Git 忽略內容）
output/                  # 固定輸出目錄（Git 忽略內容）
watermark/               # 固定 PNG 浮水印目錄（Git 忽略內容）
```

`auto_reframe_core/` 內部職責：

| 模組 | 職責 |
|---|---|
| `__main__.py` / `cli.py` | GUI、Reframe、Compress 的統一啟動路由 |
| `gui.py` | Windows / macOS 共用 Tk GUI |
| `reframe.py` | `ReframeConfig` 與 Reframe 工作協調 |
| `compress.py` | `CompressConfig` 與 Compress 工作協調 |
| `video_utils.py` | FFprobe、bitrate、進度、平行化與取消 |
| `platform_profile.py` | Windows/macOS 平台判斷與 worker 上限 |
| `encoder_profiles.py` | H.264/H.265 硬體編碼器偵測與 encoder 參數 |
| `target_specs.py` | targets 驗證與正規化 |
| `reframe_geometry.py` | 裁切、補邊與 final canvas 尺寸計算 |
| `output_plans.py` | 輸出尺寸、命名、tmp/final 檔案規劃 |
| `ffmpeg_graphs.py` | FFmpeg command / filter graph 組裝 |
| `text_layout.py` | 固定 top/video/bottom 文字版面邏輯 |
| `watermark.py` | 共用 PNG 浮水印驗證與 overlay graph |
| `gui_options.py` | GUI 顯示選項、比例解析與 PNG 掃描 |
| `config_store.py` | `config.json` 版本驗證與原子儲存 |
| `batch_runner.py` | 掃描輸入、清 tmp、平行執行與任務總結 |

固定版面與 drawtext 行為由 `tests/test_behavior_guards.py` 鎖住。重構或修改濾鏡時請先閱讀 `AGENTS.md`。

---

## 設定值說明（`ReframeConfig`）

所有設定值均在 `auto_reframe_core/reframe.py` 的 `ReframeConfig` 類別中定義。

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
| `top_text_line_spacing_ratio` | `1.08` | 上方文字區的多行文字行距倍數（`1.0` = 無額外間距） |
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
1: ©your.credit
```

### 注意事項

- 檔案編碼請使用 **UTF-8**（支援中文、日文等多位元組字元）
- 若文字中包含 `:` 或 `%` 符號，腳本會自動轉義，**不需手動處理**
- 若文字中包含單引號 `'`，腳本同樣會自動轉義
- 若檔案不存在，腳本啟動時會自動建立空白檔案

---

## PNG 浮水印

- 資料夾名稱固定為小寫 `watermark/`，這對區分大小寫的 macOS 磁碟很重要。
- GUI 啟動或按下「重新整理」時會掃描所有 `.png`／`.PNG`，依檔名排序並預選第一個。
- 「蓋浮水印」可在重製與壓縮兩種模式中獨立開關。
- 預設錨點為下方中央。
- 預設寬度為輸出畫面寬度的 32%。
- 預設底部距離以 FHD 高度 56px 為基準，其他解析度等比縮放。
- 預設透明度為 85%，並保留 PNG 本身的 alpha。
- 浮水印會在每個解析度縮放完成後才疊加，再共用給相同尺寸的 H.264／H.265 輸出。
- 有浮水印的輸出名稱會加 `_wm`，避免與無浮水印版本因 `skip_existing=True` 互相跳過。

---

## 輸出目錄結構

```
output/
├── 4x5_FHD_h265_wm/
│   └── video_name_4x5_FHD_h265_wm.mp4
├── COMPRESS_4K_h265_wm/
│   └── video_name_COMPRESS_4K_h265_wm.mp4
└── previews/
    └── reframe_fhd_h264.jpg
```

---

## 硬體加速

腳本啟動時會自動偵測並優先使用以下編碼器：

| 優先順序 | 編碼器 | 加速方式 | 適用硬體 |
|---|---|---|---|
| 1 | `h264_nvenc` / `hevc_nvenc` | CUDA | NVIDIA GPU |
| 2 | `h264_amf` / `hevc_amf` | D3D11VA | AMD GPU |
| 3 | `h264_qsv` / `hevc_qsv` | QSV | Intel GPU |
| 4 | `h264_videotoolbox` / `hevc_videotoolbox` | VideoToolbox | macOS |
| 5 | `libx264` / `libx265` | 軟體編碼 | 全平台（備援） |

正式轉碼若硬體路徑失敗，會依序重試：

1. 停用硬體解碼，但保留硬體編碼。
2. 軟體解碼與 `libx264`／`libx265` 軟體編碼。

---

## 依賴套件

```bash
pip install tqdm
```

> `tqdm` 為選擇性依賴，未安裝時會回退至傳統文字進度輸出。FFmpeg 與 FFprobe 需另行安裝並加入系統 PATH。

Tkinter 通常隨 Windows 的 python.org 安裝程式提供。macOS 若使用精簡或 Homebrew Python 而無法 `import tkinter`，請安裝對應的 Tcl/Tk 套件，或改用包含 Tk 的 python.org 版本。

---

## 驗證

```bash
python -m compileall -q auto_reframe_core tests scripts
python -m auto_reframe_core --help
python3 -m unittest discover -s tests
git diff --check
```

---

## 建立 Release

`.github/workflows/release.yml` 提供手動 `workflow_dispatch`：

1. 更新 `auto_reframe_core/version.py` 的 `VERSION`。
2. 確認根目錄的 `LICENSE` 保留所有權利聲明仍符合發佈意圖。
3. 將通過測試的版本推送到 `main`。
4. 在 GitHub **Actions → Release → Run workflow** 選擇 `main`。
5. 輸入不含 `v` 的 `MAJOR.MINOR.PATCH`，例如 `2.4.0`。

Workflow 會先在 Windows、macOS、Linux 跑完整測試，再建立含逐檔 manifest 的
ZIP、SHA-256 checksum 與 GitHub Release。Action dependency 固定至完整 commit
SHA，Release job 只有最小的 `contents: write` 權限。

公開與首次發佈前，請完成
[`docs/PUBLIC_RELEASE_CHECKLIST.md`](docs/PUBLIC_RELEASE_CHECKLIST.md)，特別是啟用
GitHub Immutable Releases、private vulnerability reporting 與 branch protection。

## 授權

本專案採 **All Rights Reserved（保留所有權利）**，不是開放原始碼授權。
公開可見不等於取得使用、修改、建立衍生作品或再散布的權利；完整條款見根目錄
[`LICENSE`](LICENSE)。內建 Noto Serif TC 字型不受上述限制，而是依 SIL Open
Font License 1.1 授權，完整文字已隨附於 `fonts/LICENSE`，來源與固定雜湊詳見
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。

使用真實輸入的四格 smoke test（在系統暫存目錄擷取短片，結束後自動清理）：

```bash
python tests/input_smoke.py input/MVI_0156.MP4 --seconds 2
```
