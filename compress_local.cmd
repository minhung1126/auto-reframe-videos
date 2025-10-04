@echo off
setlocal

:: 設定輸入和輸出資料夾的名稱
set INPUT_DIR=input_for_compression
set OUTPUT_DIR=final_output

:: 檢查 FFmpeg 是否存在
where ffmpeg >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] FFmpeg not found in your system's PATH.
    echo Please install FFmpeg and add it to your PATH environment variable.
    pause
    exit /b
)

:: 建立資料夾 (如果不存在)
if not exist "%INPUT_DIR%" mkdir "%INPUT_DIR%"
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

echo Starting batch compression...

:: 迴圈處理輸入資料夾中的所有 .mp4 檔案
for %%f in ("%INPUT_DIR%\*.mp4") do (
    echo Compressing "%%~nxf"...
    
    :: 使用 ffmpeg 進行壓縮
    :: -i: 輸入檔案
    :: -c:v libx264: 使用 CPU 進行 H.264 編碼 (最通用)
    :: -b:v 10M: 設定目標視訊位元率為 10Mbps
    :: -preset medium: 編碼速度與壓縮率的平衡點 (可選: slow, fast, etc.)
    :: -c:a copy: 直接複製音訊，不重新編碼以節省時間
    :: -y: 如果輸出檔案已存在，則自動覆蓋
    ffmpeg -y -i "%%f" -c:v libx264 -b:v 10M -preset medium -c:a copy "%OUTPUT_DIR%\%%~nxf"
)

echo.
echo All videos have been processed.

echo.
pause
