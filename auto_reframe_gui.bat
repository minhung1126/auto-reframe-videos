@echo off
cd /d "%~dp0"
pythonw auto_reframe_gui.py
if errorlevel 1 (
    python auto_reframe_gui.py
    pause
)
