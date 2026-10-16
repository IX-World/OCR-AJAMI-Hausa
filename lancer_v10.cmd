@echo off
call "%~dp0python_ocr.cmd" --version v10 --require-group-disjoint --require-all-images --fresh-start %*
exit /b %errorlevel%
