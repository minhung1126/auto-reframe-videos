
import cv2
import mediapipe as mp
import numpy as np
import os
import argparse
from tqdm import tqdm
import concurrent.futures
import subprocess

# --- Constants ---
MAX_WORKERS = max(1, os.cpu_count() // 4)
LANDMARK_DETECTION_INTERVAL = 3  # Run pose detection every 3 frames


def find_best_encoder(verbose=True):
    """
    Tests for available hardware encoders and returns the best available one.
    Falls back to a high-quality software encoder if none are found.
    """
    encoder_profiles = [
        {'name': 'NVIDIA NVENC', 'vcodec': 'h264_nvenc', 'params': ['-preset', 'fast', '-cq:v', '20']},
        {'name': 'Intel QSV',    'vcodec': 'h264_qsv',   'params': ['-preset', 'fast', '-cq:v', '20']},
        {'name': 'AMD AMF',      'vcodec': 'h264_amf',   'params': ['-quality', 'quality', '-rc', 'cqp', '-qp_i', '20', '-qp_p', '20', '-qp_b', '20']},
        {'name': 'Software (libx264)', 'vcodec': 'libx264', 'params': ['-preset', 'medium', '-crf', '20']}
    ]

    if verbose: print("正在偵測可用的最佳品質編碼器...")

    for profile in encoder_profiles:
        try:
            cmd = [
                'ffmpeg', '-f', 'lavfi', '-i', 'nullsrc=s=640x480', '-t', '1',
                '-c:v', profile['vcodec'], *profile['params'], '-f', 'null', '-'
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            if verbose: print(f"成功找到編碼器： {profile['name']}")
            return profile['vcodec'], profile['params']
        except FileNotFoundError:
            print("錯誤：找不到 FFmpeg。請確保 FFmpeg 已安裝並在其系統 PATH 中。")
            raise
        except subprocess.CalledProcessError:
            if verbose: print(f"資訊：未找到 {profile['name']}.")
            continue
    
    return 'libx264', ['-preset', 'medium', '-crf', '20']

def has_embedded_thumbnail(video_path):
    """Checks if a video file has an embedded thumbnail (attached_pic)."""
    if not os.path.exists(video_path):
        return False
    try:
        cmd = [
            'ffprobe', '-v', 'error', '-select_streams', 'v',
            '-show_entries', 'stream_disposition=attached_pic',
            '-of', 'csv=p=0', # Use CSV output for simpler parsing
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        # The command outputs 'attached_pic' for each stream that is an attached picture.
        # We just need to see if 'attached_pic' is in the output.
        return 'attached_pic' in result.stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        # If ffprobe fails or is not found, assume no thumbnail
        return False

def generate_thumbnail_ffmpeg(video_path, output_dir, timestamp_sec=2):
    """
    Generates a thumbnail for the video using FFmpeg.
    Saves the thumbnail in the specified output directory.
    Returns the path to the thumbnail on success, otherwise None.
    """
    base_name = os.path.basename(video_path)
    file_name, _ = os.path.splitext(base_name)
    thumbnail_path = os.path.join(output_dir, f"{file_name}.jpg")
    
    # Use ffprobe to get video duration to avoid errors with short clips
    try:
        ffprobe_cmd = [
            'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1', video_path
        ]
        result = subprocess.run(ffprobe_cmd, capture_output=True, text=True, check=True)
        duration = float(result.stdout)
        
        # If the requested timestamp is beyond the video length, use the midpoint
        if timestamp_sec > duration:
            print(f"[{base_name}] 警告：請求的縮圖時間戳 ({timestamp_sec}s) 超過影片長度 ({duration:.2f}s)。將使用影片中點作為替代。")
            timestamp_sec = duration / 2
            
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError) as e:
        print(f"[{base_name}] 警告：無法取得影片長度，將使用預設時間戳。錯誤： {e}")
        # We can still proceed; ffmpeg will likely pick the last frame if the timestamp is out of bounds.

    # FFmpeg command to extract one frame
    ffmpeg_cmd = [
        'ffmpeg', '-y', '-ss', str(timestamp_sec), '-i', video_path,
        '-vframes', '1', 
        '-q:v', '2',  # High quality for the output JPEG
        thumbnail_path
    ]

    try:
        # Using capture_output=True to hide ffmpeg's verbose output unless an error occurs
        subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
        print(f"[{base_name}] 成功使用 FFmpeg 產生縮圖並儲存至： {thumbnail_path}")
        return thumbnail_path
    except subprocess.CalledProcessError as e:
        print(f"\n[{base_name}] 錯誤：使用 FFmpeg 產生縮圖失敗。")
        print(f"FFmpeg 錯誤訊息：\n{e.stderr.decode(errors='ignore')}")
        return None
    except FileNotFoundError:
        # This error was likely already caught by the main script, but good to have it here too.
        print("\n錯誤：找不到 FFmpeg/ffprobe。請確保它們已安裝並在其系統 PATH 中。")
        return None

def embed_thumbnail_and_cleanup(video_path, thumbnail_path):
    """
    Embeds the thumbnail into the video file and deletes the external thumbnail.
    """
    base_name = os.path.basename(video_path)
    temp_video_path = video_path + ".thumb.mp4"

    # Command to embed the thumbnail as a cover art
    ffmpeg_cmd = [
        'ffmpeg', '-y',
        '-i', video_path,
        '-i', thumbnail_path,
        '-map', '0',       # Map all streams from the video
        '-map', '1',       # Map all streams from the image
        '-c', 'copy',    # Copy all streams without re-encoding
        '-disposition:1', 'attached_pic', # Set the image stream as attached picture
        temp_video_path
    ]

    try:
        print(f"[{base_name}] 正在將縮圖嵌入影片中...")
        subprocess.run(ffmpeg_cmd, check=True, capture_output=True)

        # On success, replace the original video and then delete the temp thumbnail
        os.remove(video_path)
        os.rename(temp_video_path, video_path)
        os.remove(thumbnail_path)
        print(f"[{base_name}] 成功嵌入縮圖並清除暫存檔案。")

    except subprocess.CalledProcessError as e:
        print(f"\n[{base_name}] 錯誤：嵌入縮圖失敗。")
        print(f"FFmpeg 錯誤訊息：\n{e.stderr.decode(errors='ignore')}")
        # Clean up the temporary video file if it exists
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
    except FileNotFoundError:
        print("\n錯誤：找不到 FFmpeg。請確保 FFmpeg 已安裝並在其系統 PATH 中。")
    except Exception as e:
        print(f"\n[{base_name}] 清理檔案時發生未預期的錯誤: {e}")

def process_video(input_path, output_dir, worker_id=0):
    """
    Orchestrates the video processing pipeline with resume capability.
    Checks for existing files before running each stage.
    """
    # --- 1. Path and Video Info Setup ---
    base_name = os.path.basename(input_path)
    file_name, _ = os.path.splitext(base_name)
    
    raw_dir = os.path.join(output_dir, "raw")
    compressed_dir = os.path.join(output_dir, "compressed_fhd_12mbps")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(compressed_dir, exist_ok=True)

    high_quality_output_path = os.path.join(raw_dir, f"{file_name}_portrait_raw.mp4")
    compressed_output_path = os.path.join(compressed_dir, f"{file_name}_portrait_1080p_12mbps.mp4")

    # --- Final Product Check ---
    if has_embedded_thumbnail(compressed_output_path):
        print(f"[{base_name}] ✅ 偵測到已完成且帶有縮圖的檔案，跳過所有處理。")
        return

    # --- Get Video Info ---
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"錯誤：無法開啟影片檔案： {input_path}")
        return
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # --- Stage 1: High-Quality Re-framing ---
    if os.path.exists(high_quality_output_path):
        print(f"[{base_name}] ⏩ 偵測到已存在的 RAW 檔案，跳過第 1 階段。")
    else:
        print(f"[{base_name}] ▶️ 第 1 階段：開始高品質畫面重構...")
        
        # --- Configuration for Stage 1 ---
        hq_target_h = orig_h
        hq_target_w = int(hq_target_h * 9 / 16)
        SMOOTHING_FACTOR = 0.01

        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False, min_detection_confidence=0.5)
        cap = cv2.VideoCapture(input_path)
        
        crop_h = orig_h
        crop_w = int(crop_h * hq_target_w / hq_target_h)
        if crop_w > orig_w:
            print(f"錯誤：計算出的裁切寬度 ({crop_w}) 大於原始寬度 ({orig_w})。")
            cap.release()
            return

        chosen_vcodec, chosen_params = find_best_encoder(verbose=False) # Don't need verbose output here
        
        ffmpeg_cmd_hq = [
            'ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
            '-s', f'{hq_target_w}x{hq_target_h}', '-r', str(fps), '-i', '-',
            '-i', input_path, '-c:v', chosen_vcodec, *chosen_params,
            '-c:a', 'aac', '-b:a', '192k', '-map', '0:v:0', '-map', '1:a:0?',
            '-movflags', '+faststart', '-shortest', high_quality_output_path
        ]

        try:
            # Set stderr=None to prevent pipe buffer from filling up and causing a deadlock.
            # FFmpeg errors will be printed directly to the console.
            ffmpeg_process_hq = subprocess.Popen(ffmpeg_cmd_hq, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=None)
        except FileNotFoundError:
            print("\n錯誤：找不到 FFmpeg。請確保 FFmpeg 已安裝並在其系統 PATH 中。")
            return

        smoothed_x1 = float((orig_w - crop_w) // 2)
        progress_bar = tqdm(total=total_frames, desc=f"重構 {file_name}", position=worker_id)

        while True:
            ret, frame = cap.read()
            if not ret: break

            if progress_bar.n % LANDMARK_DETECTION_INTERVAL == 0:
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(image_rgb)
                if results.pose_landmarks:
                    landmarks_x = [lm.x for lm in results.pose_landmarks.landmark]
                    person_center_x_rel = (min(landmarks_x) + max(landmarks_x)) / 2
                    person_center_x_px = int(person_center_x_rel * orig_w)
                    target_x1 = person_center_x_px - crop_w // 2
                    smoothed_x1 = (SMOOTHING_FACTOR * target_x1) + ((1 - SMOOTHING_FACTOR) * smoothed_x1)
            
            crop_x1 = int(smoothed_x1)
            if crop_x1 < 0: crop_x1 = 0
            if crop_x1 + crop_w > orig_w: crop_x1 = orig_w - crop_w
            smoothed_x1 = float(crop_x1)
            
            cropped_frame = frame[:, crop_x1:crop_x1 + crop_w]
            resized_frame = cv2.resize(cropped_frame, (hq_target_w, hq_target_h), interpolation=cv2.INTER_LANCZOS4)

            try:
                ffmpeg_process_hq.stdin.write(resized_frame.tobytes())
            except (IOError, BrokenPipeError):
                print(f"\n[{base_name}] 錯誤：與 FFmpeg 的連線中斷。")
                break
            progress_bar.update(1)

        progress_bar.close()
        if ffmpeg_process_hq.stdin: ffmpeg_process_hq.stdin.close()
        
        # Wait for the process to finish.
        ffmpeg_process_hq.communicate()
        cap.release()
        pose.close()

        if ffmpeg_process_hq.returncode != 0:
            print(f"\n[{base_name}] ❌ 錯誤：第 1 階段高品質重構失敗。")
            print(f"FFmpeg 的錯誤訊息應該已顯示在主控台中。")
            return
        print(f"[{base_name}] ✅ 第 1 階段完成。")

    # --- Stage 2: Compression to FHD 12Mbps ---
    if os.path.exists(compressed_output_path):
        print(f"[{base_name}] ⏩ 偵測到已存在的壓縮檔案，跳過第 2 階段。")
    else:
        print(f"[{base_name}] ▶️ 第 2 階段：開始壓縮至 FHD 12Mbps...")
        chosen_vcodec, chosen_params = find_best_encoder(verbose=False) # Don't need verbose output here
        compress_params = []
        if 'nvenc' in chosen_vcodec or 'qsv' in chosen_vcodec:
            compress_params.extend(['-preset', 'fast'])
        elif 'libx264' in chosen_vcodec:
            compress_params.extend(['-preset', 'fast', '-maxrate', '12M', '-bufsize', '24M'])

        ffmpeg_cmd_compress = [
            'ffmpeg', '-y', '-i', high_quality_output_path, '-c:v', chosen_vcodec,
            '-vf', 'scale=1080:1920', '-b:v', '12M', *compress_params,
            '-c:a', 'copy', '-movflags', '+faststart', compressed_output_path
        ]

        try:
            subprocess.run(ffmpeg_cmd_compress, check=True, capture_output=True)
            print(f"[{base_name}] ✅ 第 2 階段完成。")
        except subprocess.CalledProcessError as e:
            print(f"\n[{base_name}] ❌ 錯誤：第 2 階段壓縮失敗。")
            print(f"FFmpeg 錯誤訊息：\n{e.stderr.decode(errors='ignore')}")
            return
        except FileNotFoundError:
            print("\n錯誤：找不到 FFmpeg。請確保 FFmpeg 已安裝並在其系統 PATH 中。")
            return

    # --- Stage 3 & 4: Thumbnail Generation and Embedding ---
    print(f"[{base_name}] ▶️ 第 3/4 階段：產生並嵌入縮圖...")
    temp_thumb_path = generate_thumbnail_ffmpeg(compressed_output_path, compressed_dir)
    if temp_thumb_path:
        embed_thumbnail_and_cleanup(compressed_output_path, temp_thumb_path)
    else:
        print(f"[{base_name}] ❌ 錯誤：無法產生或嵌入縮圖。")
        return
    
    print(f"[{base_name}] 🎉 處理完成。")

def main():
    """
    Main function to find and process all videos in the input directory.
    """
    INPUT_DIR = "input_videos"
    OUTPUT_DIR = "output_videos"
    SUPPORTED_EXTENSIONS = ['.mp4', '.mov', '.avi', '-mkv', '.wmv', '.flv']

    # Create input and output directories if they don't exist
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"資訊：確保 '{INPUT_DIR}' 和 '{OUTPUT_DIR}' 資料夾存在。")

    video_files = [f for f in os.listdir(INPUT_DIR) if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS]

    if not video_files:
        print(f"在 '{INPUT_DIR}' 資料夾中找不到任何支援的影片檔案。")
        print("請將影片檔案新增至該資料夾再執行。")
        return

    print(f"在 '{INPUT_DIR}' 中找到 {len(video_files)} 個影片，將開始處理...")
    if MAX_WORKERS > 1: print(f"使用 {MAX_WORKERS} 個併行處理程序。")

    # Sort files for consistent processing order
    video_files.sort()

    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_video, os.path.join(INPUT_DIR, filename), OUTPUT_DIR, i) for i, filename in enumerate(video_files)]
        
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"處理影片時發生未預期的錯誤: {e}")

    print("\n所有影片均已處理完畢！")

if __name__ == "__main__":
    main()
