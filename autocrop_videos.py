import cv2
import os
import numpy as np
import requests
import sys
import subprocess
import re
import shutil
import multiprocessing
from tqdm import tqdm

# --- 並行處理設定 ---
# 設定最大同時處理的影片數量。預設為 CPU 核心數的1/4。
# 您可以根據您的硬體效能和記憶體大小來調整此數值。
# 例如，如果您有 8 核心 CPU，這裡會設定為 2。
MAX_CONCURRENT_PROCESSES = multiprocessing.cpu_count() // 4

# --- 設定 ---
INPUT_FOLDER = 'input_videos'
OUTPUT_FOLDER = 'output_videos'
MODELS_DIR = 'models'
DETECT_EVERY_N_FRAMES = 3 # 每 N 幀偵測一次
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
# --- 設定結束 ---

def download_file(url, path):
    try:
        print(f"正在下載 {os.path.basename(path)}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(path, 'wb') as f:
            for data in response.iter_content(1024):
                f.write(data)
        print(f"下載完成: {path}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"下載失敗: {e}")
        return False

def ensure_models_exist(model_choice):
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)

    if model_choice == 'YOLO':
        if not os.path.exists(COCO_NAMES_PATH) and not download_file(COCO_NAMES_URL, COCO_NAMES_PATH): return False
        if not os.path.exists(YOLO_CFG_PATH) and not download_file(YOLO_CFG_URL, YOLO_CFG_PATH): return False
        if not os.path.exists(YOLO_WEIGHTS_PATH) and not download_file(YOLO_WEIGHTS_URL, YOLO_WEIGHTS_PATH): return False
    elif model_choice == 'SSD':
        if not os.path.exists(SSD_PROTOTXT_PATH) and not download_file(SSD_PROTOTXT_URL, SSD_PROTOTXT_PATH): return False
        if not os.path.exists(SSD_MODEL_PATH) and not download_file(SSD_MODEL_URL, SSD_MODEL_PATH): return False
    else:
        return False
    return True

def get_best_encoder():
    print("正在自動偵測最佳影片編碼器...")
    ENCODER_PRIORITY = {"h264_nvenc": "NVIDIA GPU (NVENC)", "h264_amf": "AMD GPU (AMF)", "h264_qsv": "Intel iGPU (QSV)", "libx264": "CPU (libx264)"}
    try:
        result = subprocess.run(['ffmpeg', '-encoders'], capture_output=True, text=True, encoding='utf-8')
        available_encoders = result.stdout
    except (FileNotFoundError, UnicodeDecodeError):
        available_encoders = ""
    for encoder, name in ENCODER_PRIORITY.items():
        if re.search(r'\b' + re.escape(encoder) + r'\b', available_encoders):
            print(f"[偵測成功] 找到最佳可用編碼器: {name}")
            return encoder
    print("[警告] 未找到任何硬體編碼器，將使用 CPU 進行編碼。" )
    return 'libx264'

def process_video(video_path, output_path, net, encoder, model_choice, classes):
    """處理單一影片檔案，偵測「人物」並裁剪為 9:16 的直向影片。"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"錯誤：無法開啟影片檔案 {video_path}")
        return

    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_height = original_height
    output_width = int(output_height * TARGET_ASPECT_RATIO)
    if output_width % 2 != 0: output_width -= 1
    if output_width > original_width:
        output_width = original_width

    temp_output_path = os.path.join(os.path.dirname(output_path), "temp_" + os.path.basename(output_path))

    # --- 全新架構: 使用 image2pipe ---
    command = [
        'ffmpeg',
        '-y',
        '-f', 'image2pipe',      # 設定輸入格式為圖片流
        '-framerate', str(fps),  # 設定圖片流的幀率
        '-vcodec', 'bmp',         # 指定傳入的圖片為無損的 bmp 格式
        '-i', '-',                # 從標準輸入讀取
        '-an',                   # 暫不處理音訊
        '-c:v', encoder,         # 使用指定的編碼器
        '-pix_fmt', 'yuv420p',
        '-s', f'{output_width}x{output_height}', # Set intermediate resolution
        '-preset', 'fast',      # 使用較快的預設，因為瓶頸已不在偵測
    ]
    if encoder == 'libx264': command.extend(['-crf', '21'])
    else: command.extend(['-cq', '21'])
    command.append(temp_output_path)

    proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    last_person_center_x = original_width // 2
    current_center_x = original_width // 2
    frame_count = 0

    # Note: tqdm might not display perfectly for parallel processes, but it will show for each file.
    with tqdm(total=total_frames, desc=f"裁切影片: {os.path.basename(video_path)}", unit="frame", position=multiprocessing.current_process()._identity[0]-1) as pbar:
        while True:
            ret, frame = cap.read()
            if not ret: break

            if frame_count % DETECT_EVERY_N_FRAMES == 0:
                (H, W) = frame.shape[:2]
                person_detections = []

                if model_choice == 'YOLO':
                    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
                    net.setInput(blob)
                    layerOutputs = net.forward(net.getUnconnectedOutLayersNames())
                    boxes, confidences, classIDs = [], [], []
                    for output in layerOutputs:
                        for detection in output:
                            scores = detection[5:]
                            classID = np.argmax(scores)
                            confidence = scores[classID]
                            if classes[classID] == "person" and confidence > CONFIDENCE_THRESHOLD:
                                box = detection[0:4] * np.array([W, H, W, H])
                                (centerX, centerY, width, height) = box.astype("int")
                                x = int(centerX - (width / 2))
                                y = int(centerY - (height / 2))
                                boxes.append([x, y, int(width), int(height)])
                                confidences.append(float(confidence))
                                classIDs.append(classID)
                    idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
                    if len(idxs) > 0:
                        for i in idxs.flatten():
                            (x, y, w, h) = boxes[i]
                            person_detections.append((x, y, x + w, y + h))

                elif model_choice == 'SSD':
                    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
                    net.setInput(blob)
                    detections = net.forward()
                    for i in np.arange(0, detections.shape[2]):
                        confidence = detections[0, 0, i, 2]
                        if confidence > CONFIDENCE_THRESHOLD:
                            idx = int(detections[0, 0, i, 1])
                            if classes[idx] == "person":
                                box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                                (startX, startY, endX, endY) = box.astype("int")
                                person_detections.append((startX, startY, endX, endY))

                if len(person_detections) > 0:
                    largest_person = max(person_detections, key=lambda rect: (rect[2] - rect[0]) * (rect[3] - rect[1]))
                    (startX, _, endX, _) = largest_person
                    current_center_x = (startX + endX) // 2

            last_person_center_x = int((1 - SMOOTHING_FACTOR) * last_person_center_x + SMOOTHING_FACTOR * current_center_x)
            crop_x = last_person_center_x - (output_width // 2)
            crop_x = max(0, int(crop_x))
            crop_x = min(int(crop_x), original_width - output_width)
            cropped_frame = frame[:, crop_x : crop_x + output_width]
            if cropped_frame.shape[1] != output_width or cropped_frame.shape[0] != output_height:
                cropped_frame = cv2.resize(cropped_frame, (output_width, output_height))
            
            ret, buf = cv2.imencode('.bmp', cropped_frame)
            if ret:
                try:
                    proc.stdin.write(buf.tobytes())
                except (OSError, BrokenPipeError) as e:
                    # This error can happen if ffmpeg closes early.
                    break
            
            frame_count += 1
            pbar.update(1)

    stdout, stderr = proc.communicate()
    cap.release()

    if proc.returncode != 0:
        print(f"\n[錯誤] 影片裁切失敗: {os.path.basename(video_path)}")
        print(stderr.decode('utf-8', errors='ignore'))
        if os.path.exists(temp_output_path): os.remove(temp_output_path)
        return

    has_audio_command = ['ffprobe', '-v', 'error', '-select_streams', 'a:0', '-show_entries', 'stream=codec_name', '-of', 'default=noprint_wrappers=1:nokey=1', video_path]
    try:
        result = subprocess.run(has_audio_command, capture_output=True, text=True, encoding='utf-8', check=True)
        has_audio = result.stdout.strip() != ""
    except (subprocess.CalledProcessError, FileNotFoundError):
        has_audio = False

    if has_audio:
        final_command = ['ffmpeg', '-y', '-i', temp_output_path, '-i', video_path, '-c:v', 'copy', '-c:a', 'copy', '-map', '0:v:0', '-map', '1:a:0', '-shortest', output_path]
        try:
            subprocess.run(final_command, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            print(f"\n[錯誤] 合併音訊失敗: {os.path.basename(video_path)}")
            print(e.stderr.decode('utf-8', errors='ignore'))
            os.rename(temp_output_path, output_path) # Keep the video-only part
            return
    else:
        os.rename(temp_output_path, output_path)

    if os.path.exists(temp_output_path): os.remove(temp_output_path)

# --- Multiprocessing Worker Functions ---

g_net = None
g_classes = None
g_best_encoder = None
g_model_choice = None

def worker_init(model_choice, encoder):
    """Initializer for each worker process."""
    global g_net, g_classes, g_best_encoder, g_model_choice
    
    g_model_choice = model_choice
    g_best_encoder = encoder

    if g_model_choice == 'YOLO':
        with open(COCO_NAMES_PATH, 'r') as f:
            g_classes = [line.strip() for line in f.readlines()]
        g_net = cv2.dnn.readNetFromDarknet(YOLO_CFG_PATH, YOLO_WEIGHTS_PATH)
    elif g_model_choice == 'SSD':
        g_classes = SSD_CLASSES
        g_net = cv2.dnn.readNetFromCaffe(SSD_PROTOTXT_PATH, SSD_MODEL_PATH)

    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        g_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        g_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

def process_single_video_task(video_file):
    """A task function that processes a single video file. Runs in a worker process."""
    global g_net, g_classes, g_best_encoder, g_model_choice

    base_filename = os.path.splitext(video_file)[0]
    final_output_filename = f"{base_filename}_reframe.mp4"
    final_output_path = os.path.join(OUTPUT_FOLDER, final_output_filename)

    if os.path.exists(final_output_path):
        return f"已跳過 (檔案已存在): {final_output_filename}"

    intermediate_output_path = None
    try:
        video_path = os.path.join(INPUT_FOLDER, video_file)
        intermediate_output_filename = f"{base_filename}_cropped.mp4"
        intermediate_output_path = os.path.join(OUTPUT_FOLDER, intermediate_output_filename)

        process_video(video_path, intermediate_output_path, g_net, g_best_encoder, g_model_choice, g_classes)

        compress_resize_command = [
            'ffmpeg', '-y', '-i', intermediate_output_path,
            '-c:v', g_best_encoder,
            '-b:v', FINAL_OUTPUT_BITRATE,
            '-s', f'{FINAL_OUTPUT_WIDTH}x{FINAL_OUTPUT_HEIGHT}',
            '-c:a', 'copy',
            '-loglevel', 'error',
            final_output_path
        ]
        subprocess.run(compress_resize_command, check=True, capture_output=True)
        
        return f"處理完成: {final_output_filename}"
    except subprocess.CalledProcessError as e:
        return f"處理失敗: {video_file} - FFmpeg Error: {e.stderr.decode('utf-8', errors='ignore')}"
    except Exception as e:
        return f"處理失敗: {video_file} - {str(e)}"
    finally:
        if intermediate_output_path and os.path.exists(intermediate_output_path):
            os.remove(intermediate_output_path)

def main():
    if shutil.which('ffmpeg') is None or shutil.which('ffprobe') is None:
        print("[錯誤] 系統中找不到 FFmpeg 或 ffprobe。請安裝 FFmpeg 並將其加入至系統 PATH 環境變數中。" )
        return

    if not ensure_models_exist(MODEL_CHOICE):
        print("無法下載必要的模型檔案，程式即將結束。" )
        return
    print("模型檔案已就緒。" )

    if not os.path.exists(INPUT_FOLDER): os.makedirs(INPUT_FOLDER)
    if not os.path.exists(OUTPUT_FOLDER): os.makedirs(OUTPUT_FOLDER)

    supported_formats = ('.mp4', '.mov', '.avi', '.mkv', '.MP4')
    video_files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(supported_formats)]

    if not video_files:
        print(f"在 '{INPUT_FOLDER}' 中沒有找到任何影片檔案。" )
        return

    print(f"在 '{INPUT_FOLDER}' 中找到 {len(video_files)} 個影片檔案，準備開始並行處理。" )
    
    best_encoder = get_best_encoder()
    
    # On Windows, 'spawn' is used, so we need to protect the main entry point.
    # The Pool should be created within this block.
    if __name__ == '__main__':
        num_processes = min(MAX_CONCURRENT_PROCESSES, len(video_files))
        if num_processes == 0:
            print("沒有影片檔案可處理。" )
            return

        print(f"將使用 {num_processes} 個背景工作程序進行處理..." )

        with multiprocessing.Pool(processes=num_processes, initializer=worker_init, initargs=(MODEL_CHOICE, best_encoder)) as pool:
            # Using imap_unordered to get results as they complete
            results = list(tqdm(pool.imap_unordered(process_single_video_task, video_files), total=len(video_files), desc="整體進度"))

        print("\n" + "="*20 + " 處理報告 " + "="*20)
        success_count = 0
        fail_count = 0
        for res in results:
            if res.startswith("處理完成"):
                success_count += 1
            else:
                fail_count += 1
                print(res) # Print errors
        
        print("\n" + f"報告總結：成功 {success_count} 個，失敗 {fail_count} 個。" )
        print("="*54)
        input("\n處理完成，請按 Enter 鍵結束...")


if __name__ == '__main__':
    # This check is crucial for multiprocessing on Windows
    multiprocessing.freeze_support() 
    main()
