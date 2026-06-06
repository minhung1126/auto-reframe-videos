# Agent Notes / AI 開發注意事項

`auto-reframe-videos` 是跨平台影片重製與壓縮工具，主要支援 Windows 與 macOS。專案使用 FFmpeg/FFprobe，會優先偵測硬體編碼器，無可用硬體加速時回退至軟體編碼。

## 不可更改的版面核心

`auto_reframe.py` 產出的最終影片必須永遠維持以下由上而下的三層結構：

1. `top_text.txt`：渲染在上方黑色補邊。
2. 輸入影片：裁切/縮放後置中放在中間。
3. `bottom_text.txt`：渲染在下方黑色補邊。

嚴禁顛倒、合併、移除或重新解讀這三個區域。

上方文字規則：
- 水平必須置中。
- 垂直方向必須從影片上緣往上增長。
- 最下面一行上方文字要維持在影片上緣附近，距離由 `text_margin` 控制。

下方文字規則：
- 水平必須置中。
- 垂直方向必須從影片下緣往下增長。
- 第一行下方文字要維持在影片下緣附近，距離由 `text_margin` 控制。

受保護的實作位置：

```text
auto_reframe_core/text_layout.py
```

任何修改都必須讓以下測試通過：

```text
tests/test_behavior_guards.py
```

## FFmpeg Drawtext 規則

處理多行文字時，絕對不要把包含換行的完整字串傳給單一 `drawtext` filter。

正確做法：
- 文字檔以 UTF-8 / UTF-8-SIG 讀取。
- 移除 `\r`，只移除結尾多餘換行。
- 在 Python 內使用 `.splitlines()` 切成單行。
- 每一行各自產生一個獨立的 `drawtext` filter。
- 傳給 FFmpeg 的 `text=` 內容不可包含任何 newline。

排版與 baseline 規則：
- 每個文字 `drawtext` filter 都必須包含 `fix_bounds=true`。
- 上方文字的 Y 座標必須使用 FFmpeg 完整變數名稱 `ascent`。
- 嚴禁把 `ascent` 簡寫成 `a`，FFmpeg 會解析失敗。

## Config Contract

輸出目標統一使用 `targets` 設定。

Reframe target 範例：

```python
{"ratio": (4, 5), "resolution": "source", "vcodec": h265}
```

Compress target 範例：

```python
{"resolution": "1080p", "vcodec": h264}
```

`resolution` 的意思是「不大於此解析度的最大標準解析度」。不可為了達到指定解析度而插值放大。

支援解析度：
- `4k`
- `2k`
- `1080p` / `fhd`
- `720p` / `hd`
- `480p`
- `360p`
- `source`

支援 codec：
- `h264`
- `h265`

## 專案結構

```text
auto_reframe.py          # 使用者入口：直式重製與 ReframeConfig
auto_compress.py         # 使用者入口：壓縮與 CompressConfig
video_utils.py           # 相容層與共用低階工具
auto_reframe_core/       # 內部核心模組
tests/                   # 行為守衛測試
```

`auto_reframe_core/` 職責分工：

- `platform_profile.py`：Windows/macOS 平台判斷與 worker 上限。
- `encoder_profiles.py`：H.264/H.265 硬體編碼器偵測與 encoder args。
- `target_specs.py`：`targets` 驗證與正規化。
- `reframe_geometry.py`：裁切、補邊與 final canvas 尺寸計算。
- `output_plans.py`：輸出尺寸、命名、tmp/final 檔案規劃。
- `ffmpeg_graphs.py`：FFmpeg command / filter graph 組裝。
- `text_layout.py`：固定 top/video/bottom 文字版面。
- `batch_runner.py`：掃描輸入、清理 tmp、平行執行與任務總結。

## 開發規則

- 平台判斷集中在 `auto_reframe_core/platform_profile.py`。
- 編碼器偵測與 encoder-specific FFmpeg args 集中在 `auto_reframe_core/encoder_profiles.py`。
- 輸出命名與規劃集中在 `auto_reframe_core/output_plans.py`。
- FFmpeg graph 組裝集中在 `auto_reframe_core/ffmpeg_graphs.py`。
- 固定文字版面集中在 `auto_reframe_core/text_layout.py`，修改前請特別小心。
- 根目錄入口腳本必須保留，避免破壞既有使用方式：
  - `python auto_reframe.py`
  - `python auto_compress.py`

完成修改前請執行：

```bash
python3 -m py_compile auto_reframe.py auto_compress.py video_utils.py auto_reframe_core/*.py tests/test_behavior_guards.py
python3 -m unittest discover -s tests
git diff --check
```
