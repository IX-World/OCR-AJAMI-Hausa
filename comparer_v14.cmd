@echo off
call "%~dp0v14_python.cmd" --compare %*
exit /b %errorlevel%
