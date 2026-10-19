@echo off
call "%~dp0v14_python.cmd" --trial A %*
exit /b %errorlevel%
