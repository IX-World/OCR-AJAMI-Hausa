@echo off
call "%~dp0python_ocr.cmd" --version v11 %*
exit /b %errorlevel%
