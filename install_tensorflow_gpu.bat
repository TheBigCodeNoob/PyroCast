@echo off
echo ========================================
echo Installing TensorFlow with DirectML (AMD GPU Support)
echo ========================================

echo.
echo Step 1: Uninstalling CPU-only TensorFlow...
pip uninstall -y tensorflow tensorflow-intel

echo.
echo Step 2: Installing TensorFlow-DirectML...
pip install tensorflow-directml

echo.
echo Step 3: Verifying GPU detection...
python -c "import tensorflow as tf; print('GPU Devices:', tf.config.list_physical_devices('GPU')); print('DirectML Available:', len(tf.config.list_physical_devices('GPU')) > 0)"

echo.
echo ========================================
echo Installation Complete!
echo ========================================
pause
