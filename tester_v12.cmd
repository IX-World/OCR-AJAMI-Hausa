@echo off
call "%~dp0python_ocr.cmd" --version v12 --test-only %*
exit /b %errorlevel%
