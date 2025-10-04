@echo off
setlocal

:: =================== SETTINGS ===================
:: Set the target video bitrate (e.g., 10M for 10Mbps, 8000k for 8Mbps)
set BITRATE=10M

:: Set the names for input and output folders
set INPUT_DIR=input_for_compression
set OUTPUT_DIR=final_output
:: ================================================

:: Check if FFmpeg exists
where ffmpeg >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] FFmpeg not found in your system's PATH.
    echo Please install FFmpeg and add it to your PATH environment variable.
    pause
    exit /b
)

:: Create folders if they don't exist
if not exist "%INPUT_DIR%" mkdir "%INPUT_DIR%"
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

echo Starting batch compression...

:: Loop through all .mp4 files in the input directory
for %%f in ("%INPUT_DIR%\*.mp4") do (
    echo Compressing "%%~nxf" to %BITRATE%...
    
    :: Using ffmpeg for compression
    :: -i: Input file
    :: -c:v libx264: Use CPU for H.264 encoding (most compatible)
    :: -b:v %BITRATE%: Set the target video bitrate from the variable above
    :: -preset medium: A balance between encoding speed and compression ratio (options: slow, fast, etc.)
    :: -c:a copy: Copy the audio stream without re-encoding to save time
    :: -y: Overwrite output file if it exists
    ffmpeg -y -i "%%f" -c:v libx264 -b:v %BITRATE% -preset medium -c:a copy "%OUTPUT_DIR%\%%~nxf"
)

echo.
echo All videos have been processed.

echo.
pause
