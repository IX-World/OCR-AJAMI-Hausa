@echo off
call "%~dp0v14_python.cmd" --trial A --check-only %*
exit /b %errorlevel%
