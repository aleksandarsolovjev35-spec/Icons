@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title Icons - GPU check

if exist ".venv\Scripts\python.exe" (
    set "VPY=.venv\Scripts\python.exe"
) else (
    set "VPY=python"
)

echo ==================================================================
echo   Checking whether Real-ESRGAN can run on the video card
echo ==================================================================
echo.

"%VPY%" tools\gpu_check.py --bench
echo.
echo Report saved to gpu_check.txt
echo.
pause
