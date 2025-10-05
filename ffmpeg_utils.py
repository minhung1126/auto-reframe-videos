"""
FFmpeg utilities for video processing.
"""
import subprocess
import re

def get_best_encoder():
    """
    Analyzes ffmpeg's output to find the best available hardware encoder.
    Prioritizes NVIDIA (NVENC), then AMD (AMF), then Intel (QSV), and finally falls back to CPU (libx264).
    """
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

def check_has_audio(video_path):
    """
    Uses ffprobe to check if a video file has an audio stream.
    """
    command = [
        'ffprobe', '-v', 'error', '-select_streams', 'a:0', 
        '-show_entries', 'stream=codec_name', '-of', 'default=noprint_wrappers=1:nokey=1', 
        video_path
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', check=True)
        return result.stdout.strip() != ""
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def merge_audio(video_only_path, original_video_path, output_path):
    """
    Merges the audio from the original video into the video-only file.
    """
    command = [
        'ffmpeg', '-y', '-i', video_only_path, '-i', original_video_path,
        '-c:v', 'copy', '-c:a', 'copy', '-map', '0:v:0', '-map', '1:a:0', '-shortest',
        output_path
    ]
    try:
        subprocess.run(command, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[錯誤] 合併音訊失敗: {output_path}")
        print(e.stderr.decode('utf-8', errors='ignore'))
        return False

def compress_video(input_path, output_path, encoder, bitrate, width, height):
    """
    Compresses and resizes the video to the final output specifications.
    """
    command = [
        'ffmpeg', '-y', '-i', input_path,
        '-c:v', encoder,
        '-b:v', bitrate,
        '-s', f'{width}x{height}',
        '-c:a', 'copy',
        '-loglevel', 'error',
        output_path
    ]
    try:
        subprocess.run(command, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[錯誤] 影片壓縮失敗: {output_path}")
        print(e.stderr.decode('utf-8', errors='ignore'))
        return False

def open_ffmpeg_pipe_for_cropping(fps, encoder, output_width, output_height, temp_output_path):
    """
    Opens an FFmpeg subprocess with a pipe to receive frames for cropping.
    """
    command = [
        'ffmpeg',
        '-y',
        '-f', 'image2pipe',
        '-framerate', str(fps),
        '-vcodec', 'bmp',
        '-i', '-',
        '-an',
        '-c:v', encoder,
        '-pix_fmt', 'yuv420p',
        '-s', f'{output_width}x{output_height}',
        '-preset', 'fast',
    ]
    if encoder == 'libx264': 
        command.extend(['-crf', '21'])
    else: 
        command.extend(['-cq', '21'])
    command.append(temp_output_path)

    return subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
