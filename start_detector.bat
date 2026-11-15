@echo off
setlocal
set "PROJECT_ROOT=%~dp0"
set "TOOLS_ROOT=%OMNIGAMER_TOOLS_ROOT%"
if not defined TOOLS_ROOT set "TOOLS_ROOT=%PROJECT_ROOT%..\OmniGamer_tools"
set "PYTHON=%TOOLS_ROOT%\detector-venv\Scripts\python.exe"
if not exist "%PYTHON%" (
  echo Detector environment not found: "%PYTHON%"
  exit /b 1
)
if not defined HF_HOME set "HF_HOME=%TOOLS_ROOT%\hf-cache"
if not defined HF_HUB_CACHE set "HF_HUB_CACHE=%HF_HOME%\hub"
if not defined DETECTOR_PRECISION set "DETECTOR_PRECISION=--fp16"
pushd "%PROJECT_ROOT%"
"%PYTHON%" "%PROJECT_ROOT%tools\detector_server.py" %DETECTOR_PRECISION% %*
set "RESULT=%ERRORLEVEL%"
popd
exit /b %RESULT%
