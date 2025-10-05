@chcp 65001 >nul
@echo off
title Update Script
echo Pulling latest version from GitHub...
echo.
rem Execute git pull
git pull
echo.
echo ==================================================================
echo Update complete!
echo.
echo - If you see 'Already up to date', you already have the latest version.
echo - If you see a 'conflict' error, please resolve the conflicts or contact the developer.
echo ==================================================================
echo.
echo Press any key to exit...
pause >nul