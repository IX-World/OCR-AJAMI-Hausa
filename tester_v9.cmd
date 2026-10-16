@echo off
call "%~dp0python_ocr.cmd" --version v9 --test-only %*
exit /b %errorlevel%
