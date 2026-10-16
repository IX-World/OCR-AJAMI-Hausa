@echo off
call "%~dp0python_ocr.cmd" --version v12 --fresh-start %*
exit /b %errorlevel%
