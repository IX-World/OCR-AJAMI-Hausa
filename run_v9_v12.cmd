@echo off
setlocal
for %%V in (v9 v10 v11 v12) do (
  call "%~dp0python_ocr.cmd" --version %%V --epochs 100 %*
  if errorlevel 1 exit /b 1
)
echo Les quatre entrainements et leurs tests publics sont termines.
