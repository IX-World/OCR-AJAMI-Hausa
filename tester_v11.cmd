@echo off
call "%~dp0python_ocr.cmd" --version v11 --test-only %*
exit /b %errorlevel%
