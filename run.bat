@echo off
cd /d "%~dp0"
pythonw -m auto_reframe_core gui
if errorlevel 1 (
    python -m auto_reframe_core gui
    pause
)
