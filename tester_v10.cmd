@echo off
call "%~dp0python_ocr.cmd" --version v10 --test-only %*
exit /b %errorlevel%
