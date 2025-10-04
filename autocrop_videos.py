import cv2
import os
import numpy as np
import requests
import sys
import subprocess
import re
import shutil
import time
from tqdm.notebook import tqdm # Use tqdm.notebook for better Colab integration

# --- Path Setup for Colab Optimization ---
# We define paths for Google Drive (for final storage) and local Colab runtime (for processing).
GDRIVE_BASE_DIR = '/content/drive/MyDrive'
LOCAL_PROCESSING_DIR = '/content/processing_temp' # Fast, temporary local storage

# Google Drive paths
GDRIVE_INPUT_FOLDER = os.path.join(GDRIVE_BASE_DIR, 'input_videos')
GDRIVE_OUTPUT_FOLDER = os.path.join(GDRIVE_BASE_DIR, 'output_videos')
GDRIVE_MODELS_DIR = os.path.join(GDRIVE_BASE_DIR, 'models')

# --- Global Settings ---
DETECT_EVERY_N_FRAMES = 9 # Process every Nth frame for detection
TARGET_ASPECT_RATIO = 9 / 16

# Final output video settings
FINAL_OUTPUT_WIDTH = 1080
FINAL_OUTPUT_HEIGHT = 1920
FINAL_OUTPUT_BITRATE = "10M"

# --- Model Configuration ---
TARGET_CLASS = 'person'    # Class to detect (from coco.names or SSD_CLASSES)
CONFIDENCE_THRESHOLD = 0.5 # Confidence threshold for detection
SMOOTHING_FACTOR = 0.1   # Camera movement smoothing (lower is smoother, 0.0-1.0)

# --- MobileNet-SSD Model ---
SSD_PROTOTXT_URL = 'https://github.com/chuanqi305/MobileNet-SSD/raw/master/deploy.prototxt'
SSD_MODEL_URL = 'https://github.com/chuanqi305/MobileNet-SSD/raw/master/mobilenet_iter_73000.caffemodel'
SSD_PROTOTXT_PATH = os.path.join(GDRIVE_MODELS_DIR, 'MobileNetSSD_deploy.prototxt')
SSD_MODEL_PATH = os.path.join(GDRIVE_MODELS_DIR, 'MobileNetSSD_deploy.caffemodel')
SSD_CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]
# --- End of Settings ---

def download_file(url, path):
    """Downloads a file from a URL to a given path with a progress bar."""
    try:
        print(f"正在下載 {os.path.basename(path)}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        with open(path, 'wb') as f, tqdm(
            desc=os.path.basename(path),
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(1024):
                size = f.write(data)
                bar.update(size)
        print(f"下載完成: {path}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"下載失敗: {e}")
        return False

def ensure_models_exist():
    """Ensures that the required AI models are downloaded."""
    if not os.path.exists(GDRIVE_MODELS_DIR):
        os.makedirs(GDRIVE_MODELS_DIR)

    all_models_exist = True
    if not os.path.exists(SSD_PROTOTXT_PATH):
        if not download_file(SSD_PROTOTXT_URL, SSD_PROTOTXT_PATH):
            all_models_exist = False
    if not os.path.exists(SSD_MODEL_PATH):
        if not download_file(SSD_MODEL_URL, SSD_MODEL_PATH):
            all_models_exist = False
    return all_models_exist

def get_best_encoder():
    """Detects the best available hardware-accelerated FFmpeg encoder."""
    print("正在自動偵測最佳影片編碼器...")
    # In Colab with GPU, 'h264_nvenc' is the priority for NVIDIA hardware acceleration.
    ENCODER_PRIORITY = {
        "h264_nvenc": "NVIDIA GPU (NVENC)",
        "h264_amf": "AMD GPU (AMF)",
        "h264_qsv": "Intel iGPU (QSV)",
        "libx264": "CPU (libx264)"
    }
    try:
        result = subprocess.run(['ffmpeg', '-encoders'], capture_output=True, text=True, encoding='utf-8')
        available_encoders = result.stdout
    except (FileNotFoundError, UnicodeDecodeError):
        print("[警告] FFmpeg 未安裝或無法讀取其輸出，將預設使用 CPU 編碼。" )
        return 'libx264'

    for encoder, name in ENCODER_PRIORITY.items():
        if re.search(r'\b' + re.escape(encoder) + r'\b', available_encoders):
            print(f"[偵測成功] 找到最佳可用編碼器: {name}")
            return encoder

    print("[警告] 未找到任何硬體編碼器，將使用 CPU 進行編碼 (速度較慢)。")
    return 'libx264'

def process_video(video_path, output_path, net, encoder, classes, use_tracker):
    """
    Handles Stage 1: Detects persons, calculates a smooth crop window,
    and pipes the cropped frames to FFmpeg to create an intermediate video.
    Returns True on success, False on failure.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"錯誤：無法開啟影片檔案 {video_path}")
        return False

    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate crop dimensions for the intermediate file
    output_height = original_height
    output_width = int(output_height * TARGET_ASPECT_RATIO)
    if output_width % 2 != 0: output_width -= 1
    if output_width > original_width:
        output_width = original_width

    # FFmpeg command to encode the piped, cropped frames
    command = [
        'ffmpeg', '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-pix_fmt', 'bgr24',
        '-s', f'{output_width}x{output_height}', # The size of the frames we are piping
        '-r', str(fps),
        '-i', '-',
        '-an', # No audio for this intermediate file
        '-c:v', encoder,
        '-pix_fmt', 'yuv420p',
        '-preset', 'fast',
        '-loglevel', 'error',
    ]
    if encoder == 'libx264': command.extend(['-crf', '21'])
    else: command.extend(['-cq', '21'])
    command.append(output_path)

    proc = subprocess.Popen(command, stdin=subprocess.PIPE)

    last_person_center_x = original_width // 2
    current_center_x = original_width // 2
    frame_count = 0
    tracker = None

    with tqdm(total=total_frames, desc=f"裁切分析", unit="frame", leave=False) as pbar:
        while True:
            ret, frame = cap.read()
            if not ret: break

            (H, W) = frame.shape[:2]

            # --- Detection/Tracking (same as original) ---
            person_found_in_frame = False
            if use_tracker and tracker is not None:
                success, tracked_bbox = tracker.update(frame)
                if success:
                    (x, y, w, h) = [int(v) for v in tracked_bbox]
                    current_center_x = x + w // 2
                    person_found_in_frame = True
                else:
                    tracker = None

            if not person_found_in_frame and (frame_count % DETECT_EVERY_N_FRAMES == 0):
                blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
                net.setInput(blob)
                detections = net.forward()
                person_detections = []
                for i in np.arange(0, detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    if confidence > CONFIDENCE_THRESHOLD:
                        idx = int(detections[0, 0, i, 1])
                        if classes[idx] == TARGET_CLASS:
                            box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                            person_detections.append(box.astype("int"))
                
                if len(person_detections) > 0:
                    largest_person_box = max(person_detections, key=lambda rect: (rect[2] - rect[0]) * (rect[3] - rect[1]))
                    (startX, startY, endX, endY) = largest_person_box
                    current_center_x = (startX + endX) // 2
                    if use_tracker:
                        tracker = cv2.legacy.TrackerKCF_create()
                        tracker.init(frame, (startX, startY, endX - startX, endY - startY))

            # --- Smoothing and Cropping ---
            last_person_center_x = int((1 - SMOOTHING_FACTOR) * last_person_center_x + SMOOTHING_FACTOR * current_center_x)
            crop_x = max(0, int(last_person_center_x - (output_width // 2)))
            crop_x = min(int(crop_x), original_width - output_width)
            cropped_frame = frame[:, crop_x : crop_x + output_width]
            
            try:
                proc.stdin.write(cropped_frame.tobytes())
            except (OSError, BrokenPipeError):
                break
            
            frame_count += 1
            pbar.update(1)

    proc.stdin.close()
    proc.wait()
    cap.release()

    if proc.returncode != 0:
        print(f"\n[錯誤] 影片裁切階段失敗: {os.path.basename(video_path)}")
        # stderr was not captured, so we can't print it. The command has -loglevel error.
        return False
    
    return True

def run_processing_pipeline():
    """
    Main function to run the entire video processing pipeline.
    This is the entry point for the Colab notebook.
    It sets up the environment, finds videos, and processes them sequentially.
    """
    # --- Initial Setup ---
    if 'google.colab' in sys.modules and not os.path.exists('/content/drive'):
        print("[錯誤] Google Drive 尚未掛載。請先在 Colab 中執行掛載 Drive 的儲存格。" )
        return

    if shutil.which('ffmpeg') is None or shutil.which('ffprobe') is None:
        print("[錯誤] 系統中找不到 FFmpeg 或 ffprobe。" )
        print("[提示] 在 Colab 中，請先執行 !apt-get install ffmpeg 來安裝。" )
        return

    if not ensure_models_exist():
        print("無法下載必要的模型檔案，程式即將結束。" )
        return
    print("模型檔案已就緒。" )

    # --- Load AI Model ---
    net = cv2.dnn.readNetFromCaffe(SSD_PROTOTXT_PATH, SSD_MODEL_PATH)
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        print("[資訊] 偵測到 CUDA，已啟用 GPU 加速 AI 偵測。" )
    else:
        print("[警告] 未偵測到 CUDA。AI 偵測將使用 CPU 執行。" )

    # --- Check for Contrib Tracker ---
    use_tracker = hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerCSRT_create')
    if use_tracker:
        print("[資訊] 偵測到進階追蹤模組 (opencv-contrib-python)，已啟用物件追蹤功能。" )
    else:
        print("[警告] 未偵測到進階追蹤模組 (opencv-contrib-python)。將使用基本偵測模式。" )
        print("[提示] 若要啟用更流暢的物件追蹤，建議執行: !pip install opencv-contrib-python-headless")

    # --- Find Videos to Process ---
    if not os.path.exists(GDRIVE_INPUT_FOLDER): os.makedirs(GDRIVE_INPUT_FOLDER)
    if not os.path.exists(GDRIVE_OUTPUT_FOLDER): os.makedirs(GDRIVE_OUTPUT_FOLDER)

    supported_formats = ('.mp4', '.mov', '.avi', '.mkv', '.MP4', '.webm')
    all_video_files = [f for f in os.listdir(GDRIVE_INPUT_FOLDER) if f.lower().endswith(supported_formats)]

    if not all_video_files:
        print(f"\n在 '{GDRIVE_INPUT_FOLDER}' 中沒有找到任何影片檔案。" )
        print("請將影片上傳至您的 Google Drive 中的 'input_videos' 資料夾。" )
        return
    
    print(f"\n在 '{GDRIVE_INPUT_FOLDER}' 中共找到 {len(all_video_files)} 個影片檔案。" )

    # --- Filter out already processed files ---
    files_to_process = []
    print("正在檢查進度以支援續傳..." )
    for video_file in all_video_files:
        base_filename = os.path.splitext(video_file)[0]
        final_output_filename = f"{base_filename}_reframe_{FINAL_OUTPUT_WIDTH}x{FINAL_OUTPUT_HEIGHT}.mp4"
        final_output_path = os.path.join(GDRIVE_OUTPUT_FOLDER, final_output_filename)

        if os.path.exists(final_output_path):
            print(f"  -> 已跳過: {video_file} (已處理)")
        else:
            files_to_process.append(video_file)

    if not files_to_process:
        print("\n所有影片都已經處理完成，沒有需要處理的新檔案。" )
        return
    
    print(f"\n找到 {len(files_to_process)} 個新影片需要處理，準備開始..." )
    
    best_encoder = get_best_encoder()
    results = []

    # --- Main Processing Loop (Colab Optimized) ---
    # Clean up and create local temp directory for fast processing
    if os.path.exists(LOCAL_PROCESSING_DIR):
        shutil.rmtree(LOCAL_PROCESSING_DIR)
    os.makedirs(LOCAL_PROCESSING_DIR)

    for video_file in tqdm(files_to_process, desc="整體進度", unit="video"):
        gdrive_video_path = os.path.join(GDRIVE_INPUT_FOLDER, video_file)
        local_video_path = os.path.join(LOCAL_PROCESSING_DIR, video_file)
        
        base_filename = os.path.splitext(video_file)[0]
        local_intermediate_path = os.path.join(LOCAL_PROCESSING_DIR, f"{base_filename}_cropped.mp4")
        local_final_path = os.path.join(LOCAL_PROCESSING_DIR, f"{base_filename}_reframe_{FINAL_OUTPUT_WIDTH}x{FINAL_OUTPUT_HEIGHT}.mp4")
        gdrive_final_path = os.path.join(GDRIVE_OUTPUT_FOLDER, os.path.basename(local_final_path))

        try:
            # STAGE 0: Copy video from Drive to local runtime for fast access
            print(f"\n[{video_file}] 正在從 Google Drive 複製到本機暫存區...")
            shutil.copy(gdrive_video_path, local_video_path)

            # STAGE 1: Crop the video based on person detection (all local)
            print(f"[{video_file}] 第 1/2 階段: 進行 AI 偵測與智慧裁切 (本地端)..." )
            success = process_video(local_video_path, local_intermediate_path, net, best_encoder, SSD_CLASSES, use_tracker)
            if not success:
                raise Exception("智慧裁切階段失敗。" )

            # STAGE 2: Scale to final resolution and merge audio (all local)
            print(f"[{video_file}] 第 2/2 階段: 縮放至 {FINAL_OUTPUT_WIDTH}x{FINAL_OUTPUT_HEIGHT} 並合併音訊 (本地端)..." )

            has_audio_command = ['ffprobe', '-v', 'error', '-select_streams', 'a:0', '-show_entries', 'stream=codec_name', '-of', 'default=noprint_wrappers=1:nokey=1', local_video_path]
            try:
                result = subprocess.run(has_audio_command, capture_output=True, text=True, encoding='utf-8', check=True)
                has_audio = result.stdout.strip() != ""
            except (subprocess.CalledProcessError, FileNotFoundError):
                has_audio = False

            final_command = [
                'ffmpeg', '-y',
                '-i', local_intermediate_path, # Input 0: The locally cropped video
                '-i', local_video_path,      # Input 1: The local original video (for audio)
                '-c:v', best_encoder,
                '-vf', f'scale={FINAL_OUTPUT_WIDTH}:{FINAL_OUTPUT_HEIGHT}',
                '-b:v', FINAL_OUTPUT_BITRATE,
                '-preset', 'fast',
                '-map', '0:v:0',
                '-loglevel', 'error',
            ]

            if has_audio:
                final_command.extend(['-c:a', 'copy', '-map', '1:a:0', '-shortest'])
            else:
                final_command.extend(['-an'])

            final_command.append(local_final_path)
            subprocess.run(final_command, check=True, capture_output=False)

            # STAGE 3: Move final video from local runtime to Google Drive
            print(f"[{video_file}] 處理完成，正在將結果移回 Google Drive...")
            shutil.move(local_final_path, gdrive_final_path)
            
            results.append(f"處理完成: {os.path.basename(gdrive_final_path)}")

        except subprocess.CalledProcessError as e:
            error_message = e.stderr.decode('utf-8', errors='ignore') if e.stderr else "FFmpeg 未提供錯誤訊息。"
            results.append(f"處理失敗: {video_file} - FFmpeg 錯誤: {error_message}")
        except Exception as e:
            results.append(f"處理失敗: {video_file} - {str(e)}")

    # --- Final Cleanup & Report ---
    if os.path.exists(LOCAL_PROCESSING_DIR):
        shutil.rmtree(LOCAL_PROCESSING_DIR)
        print("\n已清理本機暫存檔案。")

    print("\n" + "="*25 + " 處理報告 " + "="*25)
    success_count = sum(1 for res in results if res.startswith("處理完成"))
    fail_count = len(results) - success_count
    
    for res in results:
        if res.startswith("處理失敗"):
            print(f"- {res}")
    
    print("\n" + f"報告總結：成功 {success_count} 個，失敗 {fail_count} 個。" )
    print(f"所有完成的影片都已儲存至 '{GDRIVE_OUTPUT_FOLDER}'")
    print("="*60)