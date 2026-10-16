@echo off
call "%~dp0python_ocr.cmd" --version v11 --fresh-start %*
exit /b %errorlevel%
