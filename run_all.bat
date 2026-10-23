@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title Icons pipeline - cut, clean, upscale 256

echo ==================================================================
echo   Icons: cut -^> clean -^> upscale 256
echo ==================================================================
echo.

if not exist "tools\run_all.py" (
    echo [ERROR] tools\run_all.py not found.
    echo Run this file from the repository root folder.
    echo.
    pause
    exit /b 1
)

rem ---------- find Python -------------------------------------------
set "PY="
py -3 --version >nul 2>nul && set "PY=py -3"
if not defined PY (
    python --version >nul 2>nul && set "PY=python"
)
if not defined PY (
    echo [ERROR] Python not found.
    echo Install Python 3.10+ from https://www.python.org/downloads/
    echo During setup enable "Add python.exe to PATH".
    echo.
    pause
    exit /b 1
)
echo Python: %PY%
%PY% --version
echo.

rem ---------- virtual environment -----------------------------------
if not exist ".venv\Scripts\python.exe" (
    echo Creating .venv ...
    %PY% -m venv .venv
    if errorlevel 1 (
        echo [ERROR] could not create .venv
        echo.
        pause
        exit /b 1
    )
)
set "VPY=.venv\Scripts\python.exe"

rem ---------- dependencies ------------------------------------------
"%VPY%" -c "import cv2, numpy" >nul 2>nul
if errorlevel 1 (
    echo Installing numpy and opencv-contrib ...
    "%VPY%" -m pip install --upgrade pip --quiet
    "%VPY%" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] could not install dependencies
        echo.
        pause
        exit /b 1
    )
)

"%VPY%" -c "import cv2; assert hasattr(cv2,'dnn_superres')" >nul 2>nul
if errorlevel 1 (
    echo Reinstalling opencv contrib build ...
    "%VPY%" -m pip install --force-reinstall opencv-contrib-python-headless
)

rem ---------- run ---------------------------------------------------
echo.
"%VPY%" tools\run_all.py --open %*
set "RC=%ERRORLEVEL%"

echo.
if "%RC%"=="0" (
    echo Done. Look in the output folder printed above.
) else (
    echo Failed with exit code %RC%.
    echo See the messages above.
)
echo.
echo Press any key to close this window.
pause >nul
exit /b %RC%
