@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0stop_all.ps1" %*
exit /b %ERRORLEVEL%
