@echo off
title OmniGamer Live Dashboard
set "PROJECT_ROOT=%~dp0"
cd /d "%PROJECT_ROOT%"

set "PYTHONPATH=%PROJECT_ROOT%;%PYTHONPATH%"

echo Activating Python 3.11 environment...
if not exist ".\.venv\Scripts\activate.bat" (
  echo Create the project environment first: py -3.11 -m venv .venv
  echo Then install requirements.txt into .venv.
  pause
  exit /b 1
)
call .\.venv\Scripts\activate.bat

echo Starting web dashboard at http://127.0.0.1:8765 ...
start http://127.0.0.1:8765
python run_dashboard.py

pause
