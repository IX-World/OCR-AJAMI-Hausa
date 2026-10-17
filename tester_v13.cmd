@echo off
setlocal
set "OCR_PYTHON=%~dp0..\.venv-hausa\Scripts\python.exe"
if defined HAUSA_PYTHON set "OCR_PYTHON=%HAUSA_PYTHON%"
if not exist "%OCR_PYTHON%" set "OCR_PYTHON=C:\Users\admin\Documents\ChatGPT\AJAMI\.venv-v10\Scripts\python.exe"
if not exist "%OCR_PYTHON%" (
  echo Python introuvable. Definir HAUSA_PYTHON.
  exit /b 1
)
"%OCR_PYTHON%" -X utf8 -u "%~dp0train_v13.py" --evaluate-only %*
exit /b %errorlevel%
