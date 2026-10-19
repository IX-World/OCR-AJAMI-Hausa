@echo off
call "%~dp0v14_python.cmd" --trial D %*
exit /b %errorlevel%
