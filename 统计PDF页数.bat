@chcp 65001 >nul
@echo off
setlocal
"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -ExecutionPolicy Bypass -File "%~dp0windows统计页数.ps1"
pause
