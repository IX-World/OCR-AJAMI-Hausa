@echo off
setlocal
for %%V in (v9 v10 v11 v12) do (
  call "%~dp0python_ocr.cmd" --version %%V --test-only %*
  if errorlevel 1 exit /b 1
)
