@chcp 65001 >nul
@echo off
setlocal

set "TARGET=%~dp0"
set "TARGET=%TARGET:~0,-1%"
set "SEARCH=%~dp0"

:SEARCH_UP

if exist "%SEARCH%!工具\count-subdir.ps1" (
    "C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -ExecutionPolicy Bypass -File "%SEARCH%!工具\count-subdir.ps1" "%TARGET%"
    goto END
)

for %%I in ("%SEARCH%..") do set "PARENT=%%~fI\"
if /I "%PARENT%"=="%SEARCH%" goto NOT_FOUND
set "SEARCH=%PARENT%"
goto SEARCH_UP

:NOT_FOUND
echo.
echo ERROR: Cannot find:
echo !工具\count-subdir.ps1
echo.
echo Started from:
echo %TARGET%

:END
echo.
pause
endlocal
