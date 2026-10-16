@echo off
call "%~dp0python_ocr.cmd" --version v10 --require-group-disjoint --require-all-images --test-only %*
exit /b %errorlevel%
