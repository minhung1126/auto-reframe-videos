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
        {'name': 'NVIDIA NVENC', 'vcodec': 'h264_nvenc', 'params': ['-preset', 'slow', '-profile:v', 'high', '-cq:v', '18']},
        {'name': 'Intel QSV',    'vcodec': 'h264_qsv',   'params': ['-preset', 'slow', '-profile:v', 'high', '-cq:v', '18']},
        {'name': 'AMD AMF',      'vcodec': 'h264_amf',   'params': ['-quality', 'quality', '-rc', 'cqp', '-qp_i', '18', '-qp_p', '18', '-qp_b', '18']},
        {'name': 'Software (libx264)', 'vcodec': 'libx264', 'params': ['-preset', 'slower', '-profile:v', 'high', '-crf', '18']}
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




def process_video(input_path, output_dir, worker_id=0):
    base_name = os.path.basename(input_path)
    file_name, _ = os.path.splitext(base_name)
    
    raw_dir = os.path.join(output_dir, "raw")
    compressed_dir = os.path.join(output_dir, "compressed_fhd_20mbps")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(compressed_dir, exist_ok=True)
    logs_dir = os.path.join(output_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, f"{file_name}.log")

    high_quality_output_path = os.path.join(raw_dir, f"{file_name}_portrait_raw.mp4")
    compressed_output_path = os.path.join(compressed_dir, f"{file_name}_portrait_1080p_20mbps.mp4")
    high_quality_output_path_tmp = high_quality_output_path + ".tmp"
    compressed_output_path_tmp = compressed_output_path + ".tmp"

    if os.path.exists(compressed_output_path):
        print(f"[{base_name}] OK 已完成，跳過。")
        return

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"錯誤：無法開啟影片： {input_path}")
        return
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if not os.path.exists(high_quality_output_path):
        print(f"[{base_name}] >> 第 1 階段：高品質重構...")
        hq_target_h = orig_h
        hq_target_w = int(hq_target_h * 9 / 16)
        SMOOTHING_FACTOR = 0.01

        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(static_image_mode=False,
                            model_complexity=1, min_detection_confidence=0.5)
        cap = cv2.VideoCapture(input_path)
        
        crop_h = orig_h
        crop_w = int(crop_h * hq_target_w / hq_target_h)
        if crop_w > orig_w:
            print(f"錯誤：計算出的裁切寬度 ({crop_w}) 大於原始寬度 ({orig_w})。")
            cap.release()
            return

        chosen_vcodec, chosen_params = find_best_encoder(verbose=False)
        
        ffmpeg_cmd_hq = [
            'ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
            '-s', f'{hq_target_w}x{hq_target_h}', '-r', str(fps), '-i', '-',
            '-i', input_path, '-c:v', chosen_vcodec, *chosen_params,
            '-c:a', 'aac', '-b:a', '192k', '-map', '0:v:0', '-map', '1:a:0?',
            '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart', '-shortest', '-f', 'mp4', high_quality_output_path_tmp
        ]

        log_file_hq = open(log_path, 'w', encoding='utf-8')
        try:
            ffmpeg_process_hq = subprocess.Popen(ffmpeg_cmd_hq, stdin=subprocess.PIPE, stdout=log_file_hq, stderr=log_file_hq)
        except FileNotFoundError:
            print(f"[{base_name}] 錯誤：找不到 FFmpeg。")
            log_file_hq.close()
            return

        smoothed_x1 = float((orig_w - crop_w) // 2)
        progress_bar = tqdm(total=total_frames, desc=f"重構 {file_name}", position=worker_id)

        try:
            while True:
                ret, frame = cap.read()
                if not ret: break

                if progress_bar.n % LANDMARK_DETECTION_INTERVAL == 0:
                    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
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
                ffmpeg_process_hq.stdin.write(resized_frame.tobytes())
                progress_bar.update(1)
        except (IOError, BrokenPipeError):
            print(f"\n[{base_name}] 錯誤：與 FFmpeg 的連線中斷。")
        finally:
            progress_bar.close()
            if ffmpeg_process_hq.stdin: ffmpeg_process_hq.stdin.close()
            ffmpeg_process_hq.communicate()
            log_file_hq.close()
            cap.release()
            pose.close()

        if ffmpeg_process_hq.returncode == 0:
            os.rename(high_quality_output_path_tmp, high_quality_output_path)
            print(f"[{base_name}] OK 第 1 階段完成。")
        else:
            print(f"\n[{base_name}] FAIL 錯誤：第 1 階段失敗。")
            if os.path.exists(high_quality_output_path_tmp):
                os.remove(high_quality_output_path_tmp)
            return

    if not os.path.exists(compressed_output_path):
        print(f"[{base_name}] >> 第 2 階段：壓縮...")
        chosen_vcodec, chosen_params = find_best_encoder(verbose=False)
        compress_params = []
        if 'nvenc' in chosen_vcodec or 'qsv' in chosen_vcodec:
            compress_params.extend(['-preset', 'medium'])
        elif 'libx264' in chosen_vcodec:
            compress_params.extend(['-preset', 'medium', '-maxrate', '20M', '-bufsize', '40M'])

        ffmpeg_cmd_compress = [
            'ffmpeg', '-y', '-i', high_quality_output_path, '-c:v', chosen_vcodec,
            '-vf', 'scale=1080:1920', '-b:v', '20M', *compress_params,
            '-pix_fmt', 'yuv420p',
            '-c:a', 'copy', '-movflags', '+faststart', '-f', 'mp4', compressed_output_path_tmp
        ]

        try:
            result = subprocess.run(ffmpeg_cmd_compress, check=False, capture_output=True)
            with open(log_path, 'a', encoding='utf-8') as log_file:
                log_file.write("\n" + "="*20 + " STAGE 2: COMPRESSION " + "="*20 + "\n")
                log_file.write(result.stdout.decode(errors='ignore'))
                log_file.write(result.stderr.decode(errors='ignore'))
            
            if result.returncode != 0:
                raise subprocess.CalledProcessError(result.returncode, ffmpeg_cmd_compress, output=result.stdout, stderr=result.stderr)

            os.rename(compressed_output_path_tmp, compressed_output_path)
            print(f"[{base_name}] OK 第 2 階段完成。")
        except subprocess.CalledProcessError as e:
            print(f"\n[{base_name}] FAIL 錯誤：第 2 階段失敗。")
            print(f"FFmpeg 錯誤訊息：\n{e.stderr.decode(errors='ignore')}")
            if os.path.exists(compressed_output_path_tmp):
                os.remove(compressed_output_path_tmp)
            return
        except FileNotFoundError:
            print(f"\n[{base_name}] 錯誤：找不到 FFmpeg。")
            return

    print(f"[{base_name}] DONE 處理完成。")

def main():
    INPUT_DIR = "input_videos"
    OUTPUT_DIR = "output_videos"
    SUPPORTED_EXTENSIONS = ['.mp4', '.mov', '.avi', '-mkv', '.wmv', '.flv']

    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"確保 '{INPUT_DIR}' 和 '{OUTPUT_DIR}' 資料夾存在。")

    video_files = [f for f in os.listdir(INPUT_DIR) if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS]

    if not video_files:
        print(f"在 '{INPUT_DIR}' 中找不到影片。")
        return

    print(f"找到 {len(video_files)} 個影片，開始處理...")
    if MAX_WORKERS > 1: print(f"使用 {MAX_WORKERS} 個併行處理程序。")

    video_files.sort()

    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_video, os.path.join(INPUT_DIR, filename), OUTPUT_DIR, i) for i, filename in enumerate(video_files)]
        
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"處理影片時發生未預期的錯誤: {e}")

    print("\n所有影片均已處理完畢！")
    input("\n按下 ENTER 鍵退出...")

if __name__ == "__main__":
    main()