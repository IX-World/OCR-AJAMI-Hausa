@echo off
call "%~dp0v14_python.cmd" --trial C %*
exit /b %errorlevel%
