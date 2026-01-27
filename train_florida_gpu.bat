@echo off
echo ============================================================
echo FLORIDA FIRE MODEL - GPU TRAINING
echo ============================================================
echo.

REM Activate conda environment if needed
REM call conda activate your_env_name

REM Enable GPU memory growth to prevent OOM errors
set TF_FORCE_GPU_ALLOW_GROWTH=true

REM Optional: Set visible GPU devices (0 = first GPU)
REM set CUDA_VISIBLE_DEVICES=0

echo Starting Florida Fire Model Training...
echo.

python Training_Florida.py

echo.
echo ============================================================
echo Training Complete!
echo ============================================================
pause
