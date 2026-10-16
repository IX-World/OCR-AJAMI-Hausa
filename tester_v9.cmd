@echo off
call "%~dp0python_ocr.cmd" --version v9 --require-group-disjoint --test-only %*
exit /b %errorlevel%
