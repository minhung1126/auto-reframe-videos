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

import config
import ffmpeg_utils
import detection





def process_video(video_path, output_path, net, encoder, classes):
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
    output_width = int(output_height * config.TARGET_ASPECT_RATIO)
    if output_width % 2 != 0: output_width -= 1
    if output_width > original_width:
        output_width = original_width

    temp_output_path = os.path.join(os.path.dirname(output_path), "temp_" + os.path.basename(output_path))

    proc = ffmpeg_utils.open_ffmpeg_pipe_for_cropping(fps, encoder, output_width, output_height, temp_output_path)

    # --- State variables for tracking and smoothing ---
    last_person_center_x = original_width // 2
    current_center_x = original_width // 2
    frame_count = 0
    tracker = None
    tracker_initialized = False

    with tqdm(total=total_frames, desc=f"裁切影片: {os.path.basename(video_path)}", unit="frame", position=multiprocessing.current_process()._identity[0]-1) as pbar:
        while True:
            ret, frame = cap.read()
            if not ret: break

            # --- Tracker-based Detection Logic ---
            # Re-run full detection every REDETECT_INTERVAL frames or if tracker is not initialized
            if not tracker_initialized or frame_count % config.REDETECT_INTERVAL == 0:
                bbox = detection.detect_people(frame, net, classes)
                if bbox is not None:
                    # Initialize a new tracker
                    tracker = cv2.TrackerCSRT_create()
                    tracker.init(frame, bbox)
                    tracker_initialized = True
                    # Update center based on the new detection
                    current_center_x = bbox[0] + bbox[2] / 2
                else:
                    # No person detected, de-initialize tracker
                    tracker_initialized = False
            else:
                # Update tracker
                success, bbox = tracker.update(frame)
                if success:
                    # Update center based on tracker's result
                    current_center_x = bbox[0] + bbox[2] / 2
                else:
                    # Tracker lost the object
                    tracker_initialized = False

            # --- Smoothing and Cropping ---
            last_person_center_x = int((1 - config.SMOOTHING_FACTOR) * last_person_center_x + config.SMOOTHING_FACTOR * current_center_x)
            crop_x = last_person_center_x - (output_width // 2)
            crop_x = max(0, int(crop_x))
            crop_x = min(int(crop_x), original_width - output_width)
            cropped_frame = frame[:, crop_x : crop_x + output_width]
            if cropped_frame.shape[1] != output_width or cropped_frame.shape[0] != output_height:
                cropped_frame = cv2.resize(cropped_frame, (output_width, output_height))
            
            # --- Pipe to FFmpeg ---
            ret, buf = cv2.imencode('.bmp', cropped_frame)
            if ret:
                try:
                    proc.stdin.write(buf.tobytes())
                except (OSError, BrokenPipeError) as e:
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

    has_audio = ffmpeg_utils.check_has_audio(video_path)

    if has_audio:
        if not ffmpeg_utils.merge_audio(temp_output_path, video_path, output_path):
            os.rename(temp_output_path, output_path) # Keep the video-only part
            return
    else:
        os.rename(temp_output_path, output_path)

    if os.path.exists(temp_output_path): os.remove(temp_output_path)

# --- Multiprocessing Worker Functions ---

g_net = None
g_classes = None
g_best_encoder = None

def worker_init(encoder):
    """Initializer for each worker process."""
    global g_net, g_classes, g_best_encoder
    
    g_best_encoder = encoder
    g_net, g_classes = detection.load_model()

def process_single_video_task(video_file):
    """A task function that processes a single video file. Runs in a worker process."""
    global g_net, g_classes, g_best_encoder

    base_filename = os.path.splitext(video_file)[0]
    final_output_filename = f"{base_filename}_reframe.mp4"
    final_output_path = os.path.join(config.OUTPUT_FOLDER, final_output_filename)

    if os.path.exists(final_output_path):
        return f"已跳過 (檔案已存在): {final_output_filename}"

    intermediate_output_path = None
    try:
        video_path = os.path.join(config.INPUT_FOLDER, video_file)
        intermediate_output_filename = f"{base_filename}_cropped.mp4"
        intermediate_output_path = os.path.join(config.OUTPUT_FOLDER, intermediate_output_filename)

        process_video(video_path, intermediate_output_path, g_net, g_best_encoder, g_classes)

        if not ffmpeg_utils.compress_video(
            intermediate_output_path,
            final_output_path,
            g_best_encoder,
            config.FINAL_OUTPUT_BITRATE,
            config.FINAL_OUTPUT_WIDTH,
            config.FINAL_OUTPUT_HEIGHT
        ):
            raise subprocess.CalledProcessError(1, "ffmpeg compress_video")
        
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
        print("[錯誤] 系統中找不到 FFmpeg 或 ffprobe。請安裝 FFmpeg 並將其加入至系統 PATH 環境變數中。")
        return

    if not detection.ensure_models_exist():
        print("無法下載必要的模型檔案，程式即將結束。")
        return
    print("模型檔案已就緒。")

    if not os.path.exists(config.INPUT_FOLDER): os.makedirs(config.INPUT_FOLDER)
    if not os.path.exists(config.OUTPUT_FOLDER): os.makedirs(config.OUTPUT_FOLDER)

    supported_formats = ('.mp4', '.mov', '.avi', '.mkv', '.MP4')
    video_files = [f for f in os.listdir(config.INPUT_FOLDER) if f.lower().endswith(supported_formats)]

    if not video_files:
        print(f"在 '{config.INPUT_FOLDER}' 中沒有找到任何影片檔案。")
        return

    print(f"在 '{config.INPUT_FOLDER}' 中找到 {len(video_files)} 個影片檔案，準備開始並行處理。")
    
    best_encoder = ffmpeg_utils.get_best_encoder()
    
    # On Windows, 'spawn' is used, so we need to protect the main entry point.
    # The Pool should be created within this block.
    if __name__ == '__main__':
        num_processes = min(config.MAX_CONCURRENT_PROCESSES, len(video_files))
        if num_processes == 0:
            print("沒有影片檔案可處理。")
            return

        print(f"將使用 {num_processes} 個背景工作程序進行處理...")

        with multiprocessing.Pool(processes=num_processes, initializer=worker_init, initargs=(best_encoder,)) as pool:
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
        
        print("\n" + f"報告總結：成功 {success_count} 個，失敗 {fail_count} 個。")
        print("="*54)
        input("\n處理完成，請按 Enter 鍵結束...")


if __name__ == '__main__':
    # This check is crucial for multiprocessing on Windows
    multiprocessing.freeze_support() 
    main()
