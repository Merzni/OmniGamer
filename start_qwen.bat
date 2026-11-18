@echo off
title OmniGamer Local VLM Server (Qwen3.5-4B Vision)
echo Starting local Vision-Language Model on GPU...

if not defined OMNIGAMER_LLAMA_SERVER (
  echo Set OMNIGAMER_LLAMA_SERVER to the llama-server.exe path.
  exit /b 1
)
if not defined OMNIGAMER_VLM_MODEL (
  echo Set OMNIGAMER_VLM_MODEL to the model GGUF path.
  exit /b 1
)
if not defined OMNIGAMER_VLM_PROJECTOR (
  echo Set OMNIGAMER_VLM_PROJECTOR to the mmproj GGUF path.
  exit /b 1
)
if defined OMNIGAMER_CUDA_DLL_DIR set "PATH=%OMNIGAMER_CUDA_DLL_DIR%;%PATH%"
set "LLAMA_DEVICE_ARGS="
if defined OMNIGAMER_LLAMA_DEVICE set "LLAMA_DEVICE_ARGS=--device %OMNIGAMER_LLAMA_DEVICE%"

"%OMNIGAMER_LLAMA_SERVER%" -m "%OMNIGAMER_VLM_MODEL%" --mmproj "%OMNIGAMER_VLM_PROJECTOR%" --host 127.0.0.1 --port 8081 -ngl 99 -c 4096 --alias qwen3.5-4b-local --reasoning off %LLAMA_DEVICE_ARGS%
