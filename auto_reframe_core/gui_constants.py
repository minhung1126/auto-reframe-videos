"""Shared GUI labels and resource locations."""

from auto_reframe_core.runtime_paths import (
    resource_root,
    user_data_root,
    watermark_root,
    logs_root,
)

SCRIPT_DIR = resource_root()


CONFIG_PATH = user_data_root() / "config.json"


CONFIG_EXAMPLE_PATH = SCRIPT_DIR / "config.json.example"


WATERMARK_DIR = watermark_root()


UPDATE_ERROR_PATH = logs_root(create=False) / "update-error.log"


CREDIT_SYMBOL = "©"


VIDEO_EXTENSIONS = {
    ".mp4",
    ".mkv",
    ".avi",
    ".mov",
    ".wmv",
    ".flv",
    ".webm",
    ".ts",
    ".m4v",
}


CJK_EDITOR_FONT_CANDIDATES = (
    "Microsoft JhengHei UI",
    "Microsoft JhengHei",
    "PingFang TC",
    "Noto Sans CJK TC",
    "Noto Sans TC",
    "Source Han Sans TC",
    "Arial Unicode MS",
)


MODE_LABELS = {
    "reframe": "比例裁切／直式重製",
    "compress": "影片壓縮／縮小解析度",
}


MODE_KEYS_BY_LABEL = {label: key for key, label in MODE_LABELS.items()}


OUTPUT_CONFLICT_SKIP = "skip"


OUTPUT_CONFLICT_OVERWRITE = "overwrite"


OUTPUT_CONFLICT_DELETE = "delete"


OUTPUT_CONFLICT_CANCEL = "cancel"


OUTPUT_CONFLICT_LABELS = {
    OUTPUT_CONFLICT_SKIP: "略過既有同名檔，只產生缺少的輸出",
    OUTPUT_CONFLICT_OVERWRITE: "覆寫本次同名檔，保留資料夾內其他內容",
    OUTPUT_CONFLICT_DELETE: "刪除列出的本次輸出檔後重做",
}
