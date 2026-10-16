@echo off
if /I "%~1"=="--resume" goto resume
call "%~dp0python_ocr.cmd" --version v9 --output "%~dp0..\runs_hausa_20261015\v9_seed42_affinage" --epochs 40 --min-epochs 10 --patience 10 --finetune-from "%~dp0..\runs_hausa_20261015\v9_seed42\best_cer.pt" --learning-rate 0.0001 %*
exit /b %errorlevel%
:resume
call "%~dp0python_ocr.cmd" --version v9 --output "%~dp0..\runs_hausa_20261015\v9_seed42_affinage" --epochs 40 --min-epochs 10 --patience 10 --resume %*
exit /b %errorlevel%
