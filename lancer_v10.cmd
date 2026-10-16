@echo off
call "%~dp0python_ocr.cmd" --version v10 --require-group-disjoint --fresh-start %*
exit /b %errorlevel%
