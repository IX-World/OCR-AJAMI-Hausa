@echo off
call "%~dp0python_ocr.cmd" --version v12 --require-group-disjoint --fresh-start %*
exit /b %errorlevel%
