@echo off
setlocal EnableExtensions
cd /d "%~dp0"

rem Double-click: cut -> clean -> Real-ESRGAN -> 256x256.
rem Same as run_all.bat, only with upscale switched on and a separate
rem output folder, so 151px and 256px versions can live side by side.

call "%~dp0run_all.bat" --up 256 --out icons_256 %*
