# AGENTS.md — AI 開發與維護規範

## 專案定位

`auto-reframe-videos` 是以 Python、FFmpeg／FFprobe 與 Tkinter 建立的跨平台影片工具，
提供：

- 將來源影片裁切後置入固定畫布的 Reframe 模式；
- 保留來源比例、只縮小與轉碼的 Compress 模式；
- Windows／macOS 共用 GUI；
- H.264／H.265 硬體編碼偵測與軟體回退；
- 多目標單次解碼、平行批次、PNG 浮水印；
- 經驗證的 GitHub Release 建置與安全自動更新。

Windows 與 macOS 是主要使用平台；Linux 由 CI 驗證匯入、核心邏輯、Release 建置
與測試。FFmpeg／FFprobe 是外部必要執行檔，Tkinter 是 GUI 與部分測試的必要元件。

## 修改原則

1. 先查看 `git status` 與相關 diff。工作樹可能包含使用者尚未提交的變更；不得覆寫、
   還原、暫存或順手提交不屬於目前任務的檔案。
2. 以現行程式碼與測試中的行為契約為準。測試失敗時修正實作，不得為了變綠而削弱
   守衛測試。
3. 使用者入口統一為 `python -m auto_reframe_core`；不得重新加入根目錄 Python
   相容入口或分模式啟動器。
4. 所有實作放入 `auto_reframe_core/`；根目錄只保留跨平台啟動器、必要設定、
   文件、授權與專案目錄。
5. 不得將個人媒體、浮水印、帳號資料、Email、Notebook metadata、憑證或本機路徑
   加入版本控制或 Release。

## 不可破壞的 Reframe 版面

Reframe 輸出由上而下永遠是三層：

1. `top_text.txt`／GUI 上方文字：位於上方黑色補邊。
2. 裁切、縮放後的來源影片：置中位於畫面中央。
3. `bottom_text.txt`／GUI 下方文字：位於下方黑色補邊。

不得顛倒、合併、移除或重新解讀三個區域。

上方文字：

- 水平置中；
- 由影片上緣向上增長；
- 最下面一行靠近影片上緣，距離由 `text_margin` 控制。

下方文字：

- 水平置中；
- 由影片下緣向下增長；
- 第一行靠近影片下緣，距離由 `text_margin` 控制。

受保護實作與測試：

```text
auto_reframe_core/reframe_geometry.py
auto_reframe_core/text_layout.py
auto_reframe_core/ffmpeg_graphs.py
tests/test_behavior_guards.py
```

## Drawtext 與文字處理契約

- 文字以 UTF-8／UTF-8-SIG 為主要格式；讀取後移除 `\r`，只移除尾端多餘的
  newline，不能破壞中間空白行。
- GUI 直接提供的文字與文字檔必須經過相同正規化。
- 多行文字必須先在 Python 使用 `.splitlines()`，每行產生獨立的 `drawtext`
  filter；傳給單一 `text=` 的內容絕不可含 newline。
- 反斜線、單引號與冒號必須由 `escape_drawtext_text()` 處理。
- 每個文字 filter 必須含 `expansion=none`，讓 `%` 保持字面值；不得重新以 `%%`
  取代這項契約。
- 每個文字 filter 必須含 `fix_bounds=true`。
- 上方文字 Y 座標必須使用 FFmpeg 完整變數名稱 `ascent`，不得簡寫成 `a`。
- 文字尺寸、間距與邊距以 1920px 高為基準依輸出高度縮放。

## 目標、尺寸與輸出識別

Python API 統一使用 `targets`：

```python
# ReframeConfig
{"ratio": (4, 5), "resolution": "source", "vcodec": h265}

# CompressConfig
{"resolution": "1080p", "vcodec": h264}
```

GUI 設定檔則使用版本化文件，且把兩種模式分開：

```json
{
  "version": 1,
  "settings": {
    "targets": {
      "reframe": [],
      "compress": []
    }
  }
}
```

支援解析度：

- `4k`
- `2k`
- `1080p`／`fhd`
- `720p`／`hd`
- `480p`
- `360p`
- `source`

支援 codec：`h264`、`h265`。

尺寸契約：

- `resolution` 是「不大於指定值的最大標準解析度」，永遠不可插值放大來源。
- 編碼輸出寬高必須是有效偶數；需要對齊時向下取偶數，不能向上放大。
- Reframe target 的中央影片比例不得比 `final_ratio` 更窄高，避免失真與負補邊。
- `get_video_info()` 的 `width`／`height` 是 FFmpeg autorotate 後的顯示尺寸；
  `source_width`／`source_height` 與 `coded_width`／`coded_height` 保留原始資訊。
- 旋轉資訊優先採用 display matrix，再回退至 legacy `rotate` tag；接近 90° 倍數
  的角度依 FFmpeg 的 ±0.5° 行為處理。修改時必須通過 `tests/test_video_info.py`。
- 多個 target 若解析成完全相同的路徑與有效輸出，可去重；若同一路徑代表不同尺寸
  或輸出身分，必須明確報錯，不得靜默覆寫。
- 輸出命名與 `.tmp`／final 規劃集中在 `auto_reframe_core/output_plans.py`。

## PNG 浮水印契約

- GUI 只掃描專案小寫 `watermark/` 內的 `.png`／`.PNG`，排序後顯示。
- 使用者浮水印屬本機資料，不得加入 Git 或 Release。
- 啟用浮水印時必須有可讀檔案；位置、寬度比例、透明度與邊距都要驗證。
- 浮水印在每個輸出解析度完成縮放與文字處理後套用，再依 codec 分支，避免重複工作。
- 單張 PNG 必須以 `eof_action=repeat:shortest=0:repeatlast=1` 覆蓋完整影片。
- 預設浮水印必須重現 Lightroom「等比例 7、垂直插入 3」，不可改回固定輸出寬度
  百分比，也不可針對特定成品用誤差補償常數微調。
- `watermark_width_ratio=0.07` 表示 Lightroom 等比例值 7。若輸出畫布為
  `Wo × Ho`、PNG 完整畫布為 `Wm × Hm`，縮放倍率固定使用
  `s = 0.07 × sqrt((Wo × Ho) / (Wm × Hm))`；輸出 PNG 寬高分別為
  `Wm × s`、`Hm × s`，FFmpeg 以 `iw`／`ih` 與 `-2` 保持原始比例。
- `watermark_margin=3` 表示 Lightroom 垂直插入值 3。下方 PNG 畫布邊距固定使用
  `round(0.03 × sqrt(Wo × Ho))`；PNG 自帶透明留白屬畫布的一部分，必須自然縮放，
  不得裁掉或另加補償。
- 目前幾何驗證用的本機 PNG 畫布為 1008×96；alpha 內容範圍為
  `(16, 16)–(992, 80)`。該尺寸只用來驗證公式，PNG 本身與檔名不得加入 Git、
  測試 fixture 或 Release。
- 有浮水印的輸出路徑與檔名必須含 `_wm`，不可與無浮水印輸出共用身分。

相關位置：

```text
auto_reframe_core/watermark.py
auto_reframe_core/ffmpeg_graphs.py
auto_reframe_core/output_plans.py
auto_reframe_core/gui_options.py
```

## FFmpeg、硬體回退與取消

- 平台判斷與 worker 上限只放在 `auto_reframe_core/platform_profile.py`。
- 編碼器探測與 encoder-specific args 只放在
  `auto_reframe_core/encoder_profiles.py`。
- Filter graph 與 FFmpeg command 組裝集中在
  `auto_reframe_core/ffmpeg_graphs.py`。
- 同解析度的多 codec 輸出應共用解碼、縮放、文字與浮水印分支。
- 硬體失敗的順序必須保持：
  1. 硬體優先；
  2. 若使用硬體解碼，停用硬體解碼但保留硬體編碼；
  3. 若使用硬體編碼，最後回退至軟體解碼與 `libx264`／`libx265`。
- VideoToolbox 不可被當成 decode-side `-hwaccel` 值。
- 缺少 FFmpeg、探測逾時與子程序錯誤必須是可捕捉、可回退或可向使用者說明的失敗。
- `KeyboardInterrupt`／批次取消狀態必須傳播到所有 worker，終止已註冊的 FFmpeg
  子程序，逾時後才 kill，並阻止新的重試或輸出 promotion。
- 未完成輸出只寫入 `.tmp`；整組成功後才 promotion。取消或失敗要清理本次 `.tmp`，
  已完成的 final 輸出則保留。

取消與子程序生命週期由 `auto_reframe_core.video_utils.FFmpegCancellation`、
`run_ffmpeg_with_progress()`、`run_parallel()` 與 `tests/test_cancellation.py`
共同守衛。

## GUI 與設定檔契約

- 統一入口為 `python -m auto_reframe_core`；省略 mode 時啟動 GUI，也支援
  `reframe` 與 `compress` mode。
- Windows／macOS 雙擊啟動器分別為 `run.bat` 與 `run.command`，兩者都只啟動
  統一入口的 GUI mode。
- 更新器可辨識舊版 `auto_reframe_gui.py`，僅用於讓既有 Release 升級並交易式
  移除舊檔；新 Release 不得再包含根目錄相容入口。
- GUI 的 `input/`、`output/`、`watermark/` 是固定專案內路徑，不可改成任意外部路徑
  而破壞既有資料與更新保護模型。
- `config.json.example` 是已提交的完整預設值來源；`config.json` 是忽略且可刪除的
  本機覆寫。不得把個人路徑或個人設定寫回 example。
- 設定檔寫入必須維持版本驗證、UTF-8 與 temporary-file replace 的原子流程。
- Tk widget 只能在主執行緒更新；背景處理、更新檢查與下載結果透過 queue／`after()`
  回到主執行緒。
- 影片處理中不得同時安裝更新；更新前必須儲存並驗證目前設定。

## 安全更新與 Release 契約

版本唯一來源：

```text
auto_reframe_core/version.py
```

更新器必須維持：

- 只接受 `minhung1126/auto-reframe-videos` 的 latest stable Release；
- 嚴格驗證 canonical Release／asset URL、唯一資產名稱、大小與 GitHub SHA-256 digest；
- 限制 API、ZIP、解壓縮總量、檔案數與壓縮比；
- 拒絕 traversal、絕對路徑、Windows reserved names、symlink、重複或 Unicode／
  大小寫碰撞路徑；
- 驗證 `.release-manifest.json` 的版本、逐檔尺寸、mode 與 SHA-256；
- 保護 `.git`、`.update-backups`、`config.json`、`input/`、`output/`、
  `top_text.txt`、`bottom_text.txt`、`watermark/`；
- 拒絕在 Git working tree 中自動安裝，也不得覆寫使用者修改過的 managed files；
- 先備份、交易式取代，失敗時回滾，最後才重新啟動；
- 只清理由 updater 建立且位於系統暫存目錄的
  `auto-reframe-update-*` 工作目錄。

Release 建置必須維持：

- `scripts/build_release.py` 的明確 allowlist；不得改為「打包整個 repository」；
- Release 不含測試、workflows、Agent 指示、個人浮水印、輸入輸出、設定或秘密；
- 版本參數必須與 `VERSION` 完全一致；
- ZIP 內含逐檔 manifest、固定檔案模式與可重現時間戳；
- 根目錄 `LICENSE`、`THIRD_PARTY_NOTICES.md`、`fonts/LICENSE` 與官方字型一併保留；
- `scripts/verify_release.py` 必須透過與 GUI 相同的 staging 路徑驗證成品；
- `.github/workflows/release.yml` 只能由 default branch 手動觸發，先通過 Windows、
  macOS、Linux 測試，再以最小 `contents: write` 權限發布。

發布前更新 `VERSION`，並依 `docs/PUBLIC_RELEASE_CHECKLIST.md` 完成隱私、Release
immutability 與 repository hardening 檢查。

## 隱私、授權與第三方檔案

- `.gitignore` 中的 runtime、Notebook、秘密檔案與簽署材料規則不得弱化。
- 不得提交 Gmail、Colab metadata、Drive 路徑、個人 watermark／社群識別、
  `.env`、credentials、private keys、輸入影片或輸出影片。
- 修改歷史、刪除 refs、reflog 或執行 aggressive GC 都是不可逆操作，只有使用者
  明確授權時才能進行。
- 專案程式碼是 **All Rights Reserved**；不得自行改成開放原始碼授權或移除
  `LICENSE`。
- `fonts/NotoSerifTC.ttf` 是 Google Fonts 官方檔，固定來源 commit、SHA-256、
  Adobe copyright 與 SIL OFL 1.1 記錄在 `THIRD_PARTY_NOTICES.md` 與
  `fonts/LICENSE`。更換字型時必須同時更新來源、雜湊、notice、license 與測試。

## 專案結構與職責

```text
run.bat                         Windows GUI 啟動器
run.command                     macOS GUI 啟動器
config.json.example             GUI 已提交預設值
auto_reframe_core/
  __main__.py                   `python -m auto_reframe_core` 統一入口
  cli.py                        GUI／Reframe／Compress mode 路由
  reframe.py                    ReframeConfig、處理器與工作協調
  compress.py                   CompressConfig、處理器與工作協調
  gui.py                        Tk GUI、設定、背景工作與更新 UI
  video_utils.py                FFprobe、bitrate、進度、平行化與取消
  batch_runner.py               掃描 input、清理 tmp、批次總結
  config_store.py               版本化 GUI 設定讀寫
  encoder_profiles.py           硬體編碼探測與 encoder args
  ffmpeg_graphs.py              Reframe／Compress filter graph
  gui_options.py                GUI label/key、比例解析、PNG 掃描
  output_plans.py               尺寸、命名、去重、tmp/final 計畫
  platform_profile.py           平台差異、worker 上限、Windows pause
  reframe_geometry.py           裁切、置中、補邊與 final canvas
  target_specs.py               targets 驗證與正規化
  text_layout.py                受保護的 top/video/bottom 排版
  update_installer.py           交易式安裝、備份、回滾、重啟
  updater.py                    Release 查詢、下載、驗證與 staging
  version.py                    VERSION／__version__
  watermark.py                  浮水印驗證與 overlay helpers
scripts/
  build_release.py              Release allowlist、manifest、ZIP、checksum
  verify_release.py             用 updater staging 驗證 Release
tests/
  test_entrypoint.py             統一入口、模式路由與 Windows UTF-8
  test_behavior_guards.py       版面、graph、targets、GUI、浮水印守衛
  test_cancellation.py          中斷與 FFmpeg 子程序終止
  test_release_builder.py       Release allowlist、manifest、字型來源
  test_updater.py               下載、解壓、安裝、回滾安全性
  test_video_info.py            FFprobe rotation／display dimensions
  input_smoke.py                真實影片短片 smoke test
```

新增共用功能時先放到對應模組，不要重新把平台判斷、輸出命名、graph、文字排版、
浮水印或更新安全邏輯散回入口檔案。

## 完成前驗證

所有程式修改至少執行：

```bash
python -m compileall -q auto_reframe_core tests scripts
python -m unittest discover -s tests
git diff --check
```

注意：

- Linux CI 會先安裝 `python3-tk`；本機 Python 若缺 Tkinter，GUI 守衛測試會失敗。
- 有 FFmpeg／FFprobe 時，`tests/test_video_info.py` 的整合測試會實際執行；
  缺少時只跳過該整合部分。
- 修改 Release／updater／version 時另執行：

```bash
python scripts/build_release.py --version X.Y.Z
python -m scripts.verify_release --version X.Y.Z
```

- 修改真實 FFmpeg 版面或 graph 時，在取得使用者允許且有非敏感測試影片後執行：

```bash
python tests/input_smoke.py path/to/video.mp4 --seconds 2
```

完成後再次確認：

- `git status` 只包含本次任務預期檔案；
- 沒有 staged 使用者變更；
- 沒有新增 runtime data、個人識別或秘密；
- 若無法執行某項測試，清楚說明原因與尚未驗證的風險。
