@echo off
setlocal EnableExtensions
title PDF Page Image Processor
cd /d "%~dp0"

echo ========================================
echo PDF Page Image Processor - Windows
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
call %PYTHON_CMD% -c "import fitz, PIL, numpy, scipy" >nul 2>&1
if errorlevel 1 goto INSTALL_DEPS
goto GET_PDF

:INSTALL_DEPS
echo Required packages are missing.
echo Installing PyMuPDF, Pillow, NumPy and SciPy...
echo This may take several minutes. Do not close this window.
echo.
call %PYTHON_CMD% -m pip install -r "%~dp0requirements.txt"
if errorlevel 1 goto INSTALL_FAILED

:GET_PDF
if not "%~1"=="" goto PDF_FROM_ARGUMENT
echo.
echo Drag the PDF file into this window, then press Enter.
set /p "PDF_FILE=PDF path: "
set "PDF_FILE=%PDF_FILE:"=%"
goto CHECK_FILE

:PDF_FROM_ARGUMENT
set "PDF_FILE=%~1"

:CHECK_FILE
if not defined PDF_FILE goto NO_FILE
if not exist "%PDF_FILE%" goto FILE_NOT_FOUND

echo.
echo Processing:
echo "%PDF_FILE%"
echo.
call %PYTHON_CMD% "%~dp0pdf_image_processor.py" "%PDF_FILE%"
if errorlevel 1 goto PROCESS_FAILED

echo.
echo Completed. Output folders are beside the source PDF.
goto END

:NO_PYTHON
echo.
echo ERROR: Python 3 was not found.
echo Download Python from:
echo https://www.python.org/downloads/windows/
echo Select "Add python.exe to PATH" during installation.
goto END

:INSTALL_FAILED
echo.
echo ERROR: Package installation failed.
echo Check the messages above and your network connection.
goto END

:NO_FILE
echo.
echo ERROR: No PDF path was entered.
goto END

:FILE_NOT_FOUND
echo.
echo ERROR: The selected file does not exist:
echo "%PDF_FILE%"
goto END

:PROCESS_FAILED
echo.
echo ERROR: PDF processing failed.
echo Copy all messages shown above when requesting help.
goto END

:END
echo.
pause
endlocal
