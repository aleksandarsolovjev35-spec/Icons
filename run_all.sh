#!/usr/bin/env bash
# Иконки: нарезка → чистка → апскейл 256. Запуск: ./run_all.sh
set -euo pipefail
cd "$(dirname "$0")"

echo "=================================================================="
echo "  Иконки: нарезка - чистка - апскейл 256"
echo "=================================================================="

PY="${PYTHON:-python3}"
if ! command -v "$PY" >/dev/null 2>&1; then
  echo "[ОШИБКА] не найден python3. Поставь Python 3.10+."
  exit 1
fi
"$PY" --version

if [ ! -x ".venv/bin/python" ]; then
  echo "Создаю окружение .venv ..."
  "$PY" -m venv .venv
fi
VPY=".venv/bin/python"

if ! "$VPY" -c "import cv2, numpy, cv2.dnn_superres" >/dev/null 2>&1; then
  echo "Ставлю зависимости numpy и opencv-contrib ..."
  "$VPY" -m pip install --upgrade pip --quiet
  "$VPY" -m pip install -r requirements.txt
fi

if ! "$VPY" -c "import cv2; assert hasattr(cv2,'dnn_superres')" >/dev/null 2>&1; then
  echo "Ставлю contrib-сборку opencv с нуля ..."
  "$VPY" -m pip install --force-reinstall opencv-contrib-python-headless
fi

"$VPY" tools/run_all.py "$@"

echo
echo "Готово. Результат в папке icons_256"
