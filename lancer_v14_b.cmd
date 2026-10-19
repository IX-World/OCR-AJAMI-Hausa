@echo off
call "%~dp0v14_python.cmd" --trial B %*
exit /b %errorlevel%
