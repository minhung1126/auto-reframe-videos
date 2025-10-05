"""
Configuration for the auto-reframe-videos project.
"""
import os
import multiprocessing

# --- 並行處理設定 ---
# 設定最大同時處理的影片數量。預設為 CPU 核心數的1/4。
# 您可以根據您的硬體效能和記憶體大小來調整此數值。
# 例如，如果您有 8 核心 CPU，這裡會設定為 2。
MAX_CONCURRENT_PROCESSES = multiprocessing.cpu_count() // 4

# --- 設定 ---
INPUT_FOLDER = 'input_videos'
OUTPUT_FOLDER = 'output_videos'
MODELS_DIR = 'models'
REDETECT_INTERVAL = 30 # 每 N 幀重新進行一次完整偵測來校準追蹤器
TARGET_ASPECT_RATIO = 9 / 16

# Final output video settings
FINAL_OUTPUT_WIDTH = 1080
FINAL_OUTPUT_HEIGHT = 1920
FINAL_OUTPUT_BITRATE = "10M"

# --- 模型選擇 ---
# 請選擇 'YOLO' 或 'SSD'
# YOLO: YOLOv4-Tiny - 速度非常快，準確度稍低
# SSD: MobileNet-SSD - 速度較慢，但準確度稍高
MODEL_CHOICE = 'SSD'

# --- 模型通用設定 ---
CONFIDENCE_THRESHOLD = 0.5 # 物件偵測的信心度閾值
NMS_THRESHOLD = 0.3      # 非極大值抑制的閾值 (YOLO專用)
SMOOTHING_FACTOR = 0.1   # 攝影機移動平滑度，數值越小越平滑 (0.0 - 1.0)

# --- YOLOv4-Tiny 模型設定 ---
COCO_NAMES_URL = 'https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names'
YOLO_CFG_URL = 'https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg'
YOLO_WEIGHTS_URL = 'https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights'
COCO_NAMES_PATH = os.path.join(MODELS_DIR, 'coco.names')
YOLO_CFG_PATH = os.path.join(MODELS_DIR, 'yolov4-tiny.cfg')
YOLO_WEIGHTS_PATH = os.path.join(MODELS_DIR, 'yolov4-tiny.weights')

# --- MobileNet-SSD 模型設定 ---
SSD_PROTOTXT_URL = 'https://github.com/chuanqi305/MobileNet-SSD/raw/master/deploy.prototxt'
SSD_MODEL_URL = 'https://github.com/chuanqi305/MobileNet-SSD/raw/master/mobilenet_iter_73000.caffemodel'
SSD_PROTOTXT_PATH = os.path.join(MODELS_DIR, 'MobileNetSSD_deploy.prototxt')
SSD_MODEL_PATH = os.path.join(MODELS_DIR, 'MobileNetSSD_deploy.caffemodel')
SSD_CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]