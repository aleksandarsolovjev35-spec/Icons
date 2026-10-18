#!/usr/bin/env bash
# Rebuild the whole library from src/raw and validate the result.
#
#   ./tools/build_all.sh
#
# Requires: python3 with pillow/numpy/scipy in .venv, node for the rule tests.
set -euo pipefail

cd "$(dirname "$0")/.."
PY=.venv/bin/python
[ -x "$PY" ] || PY=python3

echo "==> cutting and normalising the artwork"
"$PY" tools/process_library.py

echo
echo "==> rebuilding the icon index"
"$PY" tools/build_library.py

echo
echo "==> testing the matching rules"
node tools/test_rules.mjs

echo
echo "==> rendering the contact sheet"
"$PY" tools/contact_sheet.py

echo
echo "done"
