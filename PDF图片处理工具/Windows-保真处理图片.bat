@echo off
setlocal EnableExtensions
title PDF Image Tool - Faithful Processing
cd /d "%~dp0"

echo ========================================
echo PDF Image Tool - Faithful Processing
echo ========================================
echo.

set "PYTHON_CMD="
where py >nul 2>&1
if errorlevel 1 goto CHECK_PYTHON
set "PYTHON_CMD=py -3"
goto PYTHON_FOUND

:CHECK_PYTHON
where python >nul 2>&1
if errorlevel 1 goto NO_PYTHON
set "PYTHON_CMD=python"

:PYTHON_FOUND
call %PYTHON_CMD% --version
if errorlevel 1 goto NO_PYTHON

echo.
echo Checking Python packages...
call %PYTHON_CMD% -c "import pymupdf, PIL, numpy, scipy" >nul 2>&1
if errorlevel 1 goto INSTALL_DEPS
goto RUN_PROCESSOR

:INSTALL_DEPS
echo Required packages are missing.
echo Installing PyMuPDF, Pillow, NumPy and SciPy...
echo This may take several minutes. Do not close this window.
echo.
call %PYTHON_CMD% -m pip install -r "%~dp0requirements.txt"
if errorlevel 1 goto INSTALL_FAILED

:RUN_PROCESSOR
echo.
echo Drag one or more PDF files into this window, then press Enter.
echo Press Enter on an empty line to open the file picker.
echo After each batch, the program keeps waiting. Type Q to exit.
echo.
call %PYTHON_CMD% "%~dp0pdf_image_processor.py" %*
if errorlevel 1 goto PROCESS_FAILED
goto END

:NO_PYTHON
echo.
echo ERROR: Python 3 was not found.
goto END

:INSTALL_FAILED
echo.
echo ERROR: Package installation failed.
goto END

:PROCESS_FAILED
echo.
echo ERROR: PDF extraction failed. Check the messages above.

:END
echo.
pause
endlocal
