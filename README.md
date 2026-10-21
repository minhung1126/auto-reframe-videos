# Auto Reframe Videos

以 Python、FFmpeg／FFprobe 與 Tkinter 建立的批次影片工具，提供 Windows／macOS GUI、H.264／H.265 轉碼、多目標輸出、上下方文字與 PNG 浮水印。

| 模式 | 處理方式 | 預設輸出目標 |
| --- | --- | --- |
| 裁切重製（Reframe） | 依指定比例從來源中央裁切，置中放入預設 9:16 畫布，上下補黑邊並可加文字 | 中央影片 4:5，原始解析度上限／H.265，以及 1080p 上限／H.264 |
| 影片壓縮（Compress） | 保留來源比例，只縮小與轉碼 | 原始解析度上限／H.265 |

兩種模式都可加入多列輸出目標，分別選擇解析度與 codec；裁切重製另可選擇中央影片比例。輸出為 MP4，有音訊時轉為 AAC。

## 安裝與啟動

### 桌面安裝版

前往 [GitHub Releases](https://github.com/minhung1126/auto-reframe-videos/releases) 選擇符合 OS／CPU 的安裝檔：

| 平台 | 成品 | 安裝位置 |
| --- | --- | --- |
| Windows x64 | `auto-reframe-videos-vX.Y.Z-windows-x64-Setup.exe` | 可選僅目前使用者或所有使用者 |
| macOS Apple Silicon | `auto-reframe-videos-vX.Y.Z-macos-arm64.dmg` | 拖曳至 `~/Applications` 或 `/Applications` |
| macOS Intel | `auto-reframe-videos-vX.Y.Z-macos-x64.dmg` | 拖曳至 `~/Applications` 或 `/Applications` |

安裝版隨附 Python、Tcl/Tk、拖曳元件 TkDND、FFmpeg／FFprobe、字型與 HTTPS CA 憑證，不需另裝 Python 或 FFmpeg。Windows 日常啟動不開啟終端機，僅「所有使用者」安裝需要管理員權限；各使用者的設定與浮水印獨立保存，升級及解除安裝保留使用者資料。

桌面成品固定為供自行使用的**未簽署安裝檔**。三種原生建置與自我測試已納入 CI；乾淨實機安裝、GPU、升級與資料保留、Windows DPI 等驗收仍有待完成項目。macOS 13.0 為最低版本候選，Windows 11 優先、Windows 10 22H2 待實機驗證；Windows ARM64 尚無原生安裝檔。詳見[桌面分發與驗收規範](https://github.com/minhung1126/auto-reframe-videos/blob/main/docs/DESKTOP_DISTRIBUTION.md)。

### 原始碼版

建議使用含 Tkinter 的 Python 3.12，並準備：

- Tcl/Tk 8.x，供 GUI 與 TkDND 使用。
- 外部 FFmpeg 與 FFprobe，加入 `PATH`，或於 GUI「進階設定」指定執行檔。FFmpeg 需具備文字繪製及所需編碼器，例如 `drawtext`、`libx264`、`libx265`。
- `requirements.txt` 中固定版本的 `tkinterdnd2`。

在專案目錄執行：

```bash
python -m pip install -r requirements.txt
python -m auto_reframe_core
```

省略模式會開啟 GUI，也可明確指定 `gui`。Windows 可雙擊 `run.bat`；macOS 可雙擊 `run.command`，若缺少執行權限，先執行 `chmod +x run.command`。macOS 的啟動器使用 `python3`，手動執行指令時也可將 `python` 換成 `python3`。

`tqdm` 是選用的 CLI 進度套件，可另以 `python -m pip install tqdm` 安裝。未安裝時仍可處理影片。

## GUI 使用流程

1. 在「待處理影片」新增多個影片或資料夾，也可將檔案／資料夾拖入視窗。資料夾預設只掃描當層，可勾選包含子資料夾；重複檔案會去重。
2. 選擇輸出位置。預設為各來源影片旁的 `auto-reframe/`，也可指定共用資料夾。
3. 切換至「裁切重製」或「影片壓縮」，加入需要的輸出目標；有需要時啟用該模式的浮水印。裁切重製可在「編輯上下文」與「文字樣式」設定上下方文字。
4. 按下開始處理。程式會在背景檢查來源與實際輸出，列出目的地；若有既有輸出，可選略過、覆寫、刪除列出的本次輸出後重做，或取消。確認後會再次驗證來源與輸出狀態。
5. 在「處理進度」查看每支影片的狀態、目前階段與進度，並從下方事件日誌查看詳細訊息。要保留選項與文字時，按下「儲存設定」。

GUI 共用同一份待處理影片清單，直接讀取原始檔。來源與輸出碰撞、跨來源同名輸出及寫入權限問題會在開始前檢查，原始影片受到覆寫保護。

可掃描的影片副檔名為 `.mp4`、`.mkv`、`.avi`、`.mov`、`.wmv`、`.flv`、`.webm`、`.ts`、`.m4v`；實際能否解碼取決於影片內容與 FFmpeg。

| 頁籤 | 功能 |
| --- | --- |
| 待處理影片 | 新增、拖曳、掃描、移除與清空影片，選擇輸出位置 |
| 裁切重製 | 輸出設定、編輯上下文、文字樣式與浮水印 |
| 影片壓縮 | 保留比例的多目標轉碼與浮水印 |
| 處理進度 | 逐片狀態與事件日誌，可調整表格和日誌的分隔比例 |
| 進階設定 | 工具路徑、平行工作數、跳過既有輸出與 FFmpeg 除錯日誌；桌面版的工具路徑固定使用隨附檔案 |
| 關於／更新 | 查看版本、開啟設定檔與 Release 頁面、檢查更新 |

目前 GUI 未提供影片預覽或轉碼中的停止按鈕；處理期間關閉視窗會提示等待完成。

## 輸出比例、解析度與檔案

裁切重製可選中央影片比例 `4:5`、`1:1`、`4:3`、`16:9`，或輸入自訂 `寬:高`。中央影片比例不得比最終畫布更窄高。

支援的解析度鍵值為 `source`、`4k`、`2k`、`1080p`／`fhd`、`720p`／`hd`、`480p`、`360p`，codec 為 `h264`、`h265`。GUI 會顯示對應的中文名稱。

解析度選項表示**不大於指定上限、且來源可容納的最大標準解析度**；選擇 4K 或 1080p 不會插值放大來源。`source` 保留可用來源尺寸，編碼寬高需要對齊時向下取偶數。輸出尺寸計算會採用影片旋轉後的顯示方向，檔名的解析度標籤依實際輸出尺寸產生。

輸出依中央比例、有效解析度與 codec 分類；浮水印版本的資料夾及檔名另加 `_wm`。例如：

```text
auto-reframe/
├── 4x5_FHD_h264/
│   └── clip_4x5_FHD_h264.mp4
├── 4x5_FHD_h265_wm/
│   └── clip_4x5_FHD_h265_wm.mp4
└── COMPRESS_4K_h265/
    └── clip_COMPRESS_4K_h265.mp4
```

完全相同的有效輸出可去重；同一路徑若對應不同輸出身分則報錯。未完成檔案先寫入 `.mp4.tmp`，同一輸出組全部成功後才轉為正式檔案；失敗會清理該次暫存檔，已完成的正式輸出保留。

## 上下方文字

Reframe 由上到下維持三層版面：

```text
┌────────────────────┐
│ 上方黑邊：上方文字  │
├────────────────────┤
│  置中的裁切影片    │
├────────────────────┤
│ 下方黑邊：下方文字  │
└────────────────────┘
```

文字水平置中。上方文字以最下面一行靠近影片上緣，增加行數時向上延伸；下方文字以第一行靠近影片下緣，後續行向下延伸。兩者與影片的距離由 `text_margin` 控制，留空即不顯示文字。

預設使用內附 Noto Serif TC 白字，上／下字級為 48／24、邊距 20、行距倍數 1.08／1.2。字級、間距與邊距以 1920px 高為基準，隨輸出高度縮放。GUI 支援選擇字型、文字顏色與複製／插入 `©`。

GUI 文字由編輯器提供並儲存於設定檔。CLI／Python API 預設讀取 `top_text.txt` 與 `bottom_text.txt`：相對路徑以 `input_dir` 的父目錄為基準，缺少時建立空白檔。建議使用 UTF-8；換行會正規化並保留中間空白行，反斜線、單引號、冒號與 `%` 可直接輸入。

## PNG 浮水印

在各模式的浮水印區按「開啟資料夾」，將 PNG 放入使用者資料目錄的 `watermark/`，再按「重新整理」。GUI 僅列出此資料夾的 `.png`／`.PNG` 並排序；兩種模式各自保存啟用狀態與選項。

- 位置支援九宮格錨點，預設下方中央。
- 裁切重製的預設等比例大小為 `0.15`（Lightroom 15），影片壓縮為 `0.10`（Lightroom 10）。
- 透明度預設 `0.8`（80%），保留 PNG 本身的 alpha；GUI 亦接受 `80` 或 `80%`。
- 垂直插入預設 `3`，比照 Lightroom 的插入值；PNG 自帶透明留白會保留並一起縮放。
- 浮水印套用於完成縮放與文字的畫面，再共用給相同尺寸的不同 codec 輸出，並覆蓋整段影片。

大小採用面積基準。若輸出為 `Wo × Ho`，PNG 完整畫布為 `Wm × Hm`，縮放倍率為：

```text
s = ratio × √((Wo × Ho) / (Wm × Hm))
浮水印尺寸 = (Wm × s) × (Hm × s)
下方錨點、插入值 3 的畫布邊距 = round(0.03 × √(Wo × Ho))
```

## 設定與日誌位置

原始碼版與桌面版都使用各使用者的資料目錄：

| 資料 | Windows | macOS |
| --- | --- | --- |
| 設定 | `%LOCALAPPDATA%\Auto Reframe Videos\config.json` | `~/Library/Application Support/Auto Reframe Videos/config.json` |
| 浮水印 | `%LOCALAPPDATA%\Auto Reframe Videos\watermark/` | `~/Library/Application Support/Auto Reframe Videos/watermark/` |
| 日誌 | `%LOCALAPPDATA%\Auto Reframe Videos\logs/` | `~/Library/Logs/Auto Reframe Videos/` |

[`config.json.example`](config.json.example) 是完整預設來源；使用者的 `config.json` 為版本化 JSON，透過 UTF-8 與暫存檔替換方式原子儲存。Reframe／Compress 的目標及浮水印設定分開保存。

「關於／更新 → 開啟設定檔」會開啟實際設定檔，不存在時先建立完整預設。手動編輯前請先關閉程式，修改會在下次啟動載入。「還原預設」會刪除使用者設定檔並重新套用預設。

影片清單由每次使用時加入；GUI 不建立固定的 `input/`、`output/` 或工作區設定檔。使用者影片、文字、浮水印、設定與日誌不應加入版本控制或分發成品。

## CLI 與 Python API

所有模式使用同一入口：

```bash
python -m auto_reframe_core --help
python -m auto_reframe_core --version
python -m auto_reframe_core reframe --input-dir ./videos --output-dir ./exports
python -m auto_reframe_core compress --input-dir ./videos --output-dir ./exports
```

CLI 掃描輸入資料夾當層；省略路徑時使用目前目錄下的 `input/` 與 `output/`。CLI 使用 `ReframeConfig`／`CompressConfig` 的預設值，與 GUI 儲存的設定分開；目前命令列提供模式與輸入／輸出路徑選項，進一步自訂目標請使用 GUI 或 Python API。CLI 有互動確認／結束提示，批次整合可直接呼叫 API。

例如，自訂 Reframe 的中央影片比例、解析度與文字：

```python
from auto_reframe_core.reframe import ReframeConfig, VideoReframer

config = ReframeConfig(
    input_dir="./videos",
    output_dir="./exports",
    targets=[
        {"ratio": (4, 5), "resolution": "1080p", "vcodec": "h264"},
        {"ratio": (1, 1), "resolution": "720p", "vcodec": "h265"},
    ],
    top_text_override="活動名稱\n拍攝日期",
    bottom_text_override="© Your Name",
)
success_count, failed_files = VideoReframer(config).run()
```

壓縮模式的 `targets` 不含 `ratio`：

```python
from auto_reframe_core.compress import CompressConfig, VideoCompressor

config = CompressConfig(
    input_dir="./videos",
    output_dir="./exports",
    targets=[{"resolution": "1080p", "vcodec": "h265"}],
)
success_count, failed_files = VideoCompressor(config).run()
```

兩者都可設定 `skip_existing`、`max_workers`、`debug` 與 `watermark_*`。Reframe 另有 `final_ratio` 與文字樣式設定；完整欄位見 [`reframe.py`](auto_reframe_core/reframe.py) 與 [`compress.py`](auto_reframe_core/compress.py)。

## 硬體加速與批次處理

程式會實際探測所選 codec 的可用硬體編碼器。macOS 優先 VideoToolbox；其他平台依 NVENC、AMF、QSV、VideoToolbox 順序探測，最後使用 `libx264`／`libx265`。實際可用性取決於 FFmpeg、驅動程式與硬體。

轉碼失敗時，若原先啟用硬體解碼，先停用硬體解碼並保留硬體編碼；仍失敗且使用硬體編碼時，再改用軟體解碼與軟體編碼。VideoToolbox 用於編碼，不作為解碼側的 `-hwaccel` 值。

同一裁切比例的多目標共用解碼，相同尺寸的不同 codec 共用縮放、文字與浮水印。多支影片可平行處理；工作數 `0` 表示自動，macOS 最多 4 個，其他平台最多 8 個且不超過 CPU 數量。終端機的 `Ctrl+C` 會傳播取消狀態並終止 FFmpeg 子程序，清理未完成的暫存輸出。

## 軟體更新

「關於／更新」只查詢 `minhung1126/auto-reframe-videos` 的最新正式 GitHub Release。

| 執行方式 | 更新流程 |
| --- | --- |
| 桌面安裝版 | 儲存並驗證設定後，開啟瀏覽器下載對應 OS／CPU 的安裝檔；關閉程式，再手動安裝新版。桌面版自動安裝尚未實作 |
| 解壓縮的原始碼 Release ZIP | 可下載原始碼更新，驗證 GitHub SHA-256 digest 與逐檔 manifest，再備份、交易式安裝並重新啟動；失敗時回滾 |
| Git checkout | 停用自動安裝，請以 Git 更新 |

原始碼更新保留使用者資料，舊程式備份留在 `.update-backups/`。`.release-manifest.json` 用於核對 managed files；缺少可信 manifest 或偵測到本機修改時，會拒絕自動安裝。處理影片期間無法安裝更新。桌面版的瀏覽器下載由使用者自行完成安裝檔驗證與安裝。

## 開發、驗證與發布

本節使用完整的 Git checkout；原始碼 Release ZIP 僅包含執行所需檔案，不含測試、建置腳本與開發文件。Windows／macOS 是主要使用平台，Linux 由 CI 驗證核心邏輯與封裝。

```text
auto_reframe_core/       應用程式、GUI、處理核心、更新器與統一入口
packaging/               桌面打包規格、安裝器、固定依賴與授權來源
scripts/                 FFmpeg／桌面／原始碼 Release 建置與成品驗證
tests/                   核心、GUI、取消、影片資訊及更新安全測試
docs/                    桌面分發規範與發布檢查表
fonts/                   內附字型與授權
config.json.example      完整 GUI 預設設定
run.bat / run.command    Windows／macOS 原始碼 GUI 啟動器
```

修改前請閱讀 [`AGENTS.md`](https://github.com/minhung1126/auto-reframe-videos/blob/main/AGENTS.md)。基本驗證：

```bash
python -m compileall -q auto_reframe_core tests scripts
python -m auto_reframe_core --help
python -m unittest discover -s tests
git diff --check
```

GUI 測試需要 Tk；Linux 無顯示環境時可使用 Xvfb。FFmpeg／FFprobe 可用時，影片資訊測試會執行實際整合驗證。Linux CI 亦執行 frozen 資源與轉碼測試，原生安裝體驗依[桌面驗收矩陣](https://github.com/minhung1126/auto-reframe-videos/blob/main/docs/DESKTOP_DISTRIBUTION.md)另行確認。

原始碼 Release 封裝與 staging 驗證：

```bash
python scripts/build_release.py
python -m scripts.verify_release
```

取得允許且有非敏感測試影片時，可執行短片 smoke test；測試使用系統暫存目錄並於結束後清理：

```bash
python tests/input_smoke.py path/to/video.mp4 --seconds 2
```

版本唯一來源為 [`auto_reframe_core/version.py`](auto_reframe_core/version.py)。發布時依[公開發布檢查表](https://github.com/minhung1126/auto-reframe-videos/blob/main/docs/PUBLIC_RELEASE_CHECKLIST.md)完成隱私、授權、Release immutability 與 repository hardening 檢查，將通過驗證的變更推送到 default branch，再於相同 commit 建立並推送完全一致的 `vX.Y.Z` tag。

Release workflow 由 tag push 觸發，驗證版本與 default branch commit、執行 Windows／macOS／Linux 測試及三種原生桌面建置，通過版本／commit／checksum gate 後才發布原始碼 ZIP、三份未簽署安裝檔、checksum 與 build records。原始碼 ZIP 依明確 allowlist 建置並含逐檔 manifest；桌面 FFmpeg 的固定來源、對應原始碼與授權要求見[桌面分發文件](https://github.com/minhung1126/auto-reframe-videos/blob/main/docs/DESKTOP_DISTRIBUTION.md)。

## 授權

專案程式碼採 **All Rights Reserved（保留所有權利）**，完整條款見 [`LICENSE`](LICENSE)。公開可見不代表取得使用、修改、建立衍生作品或再散布的權利。

內附 Noto Serif TC 字型依 SIL Open Font License 1.1 授權，授權文字見 [`fonts/LICENSE`](fonts/LICENSE)，固定來源與雜湊見 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。桌面版另隨附 Python、Tcl/Tk、TkDND 與 FFmpeg 等元件的授權及來源紀錄。
