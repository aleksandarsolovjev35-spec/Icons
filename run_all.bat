@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion
cd /d "%~dp0"

echo ==================================================================
echo   Иконки: нарезка - чистка - апскейл 256
echo ==================================================================
echo.

rem --- ищем Python -------------------------------------------------
set "PY="
where py >nul 2>nul && set "PY=py -3"
if not defined PY (
    where python >nul 2>nul && set "PY=python"
)
if not defined PY (
    echo [ОШИБКА] Python не найден.
    echo Поставь Python 3.10+ с https://www.python.org/downloads/
    echo При установке обязательно отметь галочку "Add Python to PATH".
    echo.
    pause
    exit /b 1
)
echo Python: %PY%
%PY% --version

rem --- виртуальное окружение --------------------------------------
if not exist ".venv\Scripts\python.exe" (
    echo.
    echo Создаю окружение .venv ...
    %PY% -m venv .venv
    if errorlevel 1 (
        echo [ОШИБКА] не удалось создать .venv
        pause
        exit /b 1
    )
)
set "VPY=.venv\Scripts\python.exe"

rem --- зависимости ------------------------------------------------
"%VPY%" -c "import cv2, numpy" >nul 2>nul
if errorlevel 1 (
    echo.
    echo Ставлю зависимости numpy и opencv-contrib ...
    "%VPY%" -m pip install --upgrade pip --quiet
    "%VPY%" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [ОШИБКА] не удалось поставить зависимости
        pause
        exit /b 1
    )
)

rem --- запуск -----------------------------------------------------
"%VPY%" -c "import cv2; assert hasattr(cv2,'dnn_superres')" >nul 2>nul
if errorlevel 1 (
    echo Ставлю contrib-сборку opencv с нуля ...
    "%VPY%" -m pip install --force-reinstall opencv-contrib-python-headless
)

echo.
"%VPY%" tools\run_all.py --open %*
set "RC=%ERRORLEVEL%"

echo.
if "%RC%"=="0" (
    echo Готово. Результат в папке icons_256
) else (
    echo Скрипт завершился с ошибкой, код %RC%
)
echo.
pause
exit /b %RC%
