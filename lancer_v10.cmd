@echo off
call "%~dp0python_ocr.cmd" --version v10 %*
exit /b %errorlevel%
