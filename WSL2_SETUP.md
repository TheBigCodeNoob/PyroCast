# AMD GPU Setup for PyroCast on Windows via WSL2

## Option 1: WSL2 + TensorFlow-ROCm (Recommended - No code changes)

### Step 1: Install WSL2 with Ubuntu
```powershell
# In PowerShell (Admin)
wsl --install -d Ubuntu-22.04
# Restart computer
```

### Step 2: Inside WSL2 Ubuntu
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and dependencies
sudo apt install python3.10 python3-pip git -y

# Install ROCm (AMD GPU drivers for Linux)
# Check your GPU compatibility at: https://rocm.docs.amd.com/en/latest/release/gpu_os_support.html
wget https://repo.radeon.com/amdgpu-install/latest/ubuntu/jammy/amdgpu-install_6.0.60002-1_all.deb
sudo apt install ./amdgpu-install_6.0.60002-1_all.deb
sudo amdgpu-install --usecase=rocm

# Install TensorFlow-ROCm
pip3 install tensorflow-rocm

# Verify GPU
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### Step 3: Copy Project to WSL2
```bash
# Access Windows files from WSL2
cd ~
cp -r /mnt/c/Users/nonna/Downloads/PyroCast .
cd PyroCast/spread_model

# Install dependencies
pip3 install numpy pandas geopandas shapely scikit-learn matplotlib
```

### Step 4: Train with GPU
```bash
python3 05_train_model.py
# Should show: Found 1 GPU(s): [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

**Expected speedup: 10-30x faster than CPU**

---

## Option 2: PyTorch with DirectML (Requires Python 3.10)

torch-directml requires Python 3.10 or lower. Create a separate conda environment:

```powershell
# Create conda environment with Python 3.10
conda create -n pyrocast_dml python=3.10 -y
conda activate pyrocast_dml

# Install PyTorch and DirectML
pip install torch==2.1.0 torchvision==0.16.0 torch-directml

# Install other dependencies
pip install tensorflow numpy pandas geopandas shapely scikit-learn matplotlib

# Run PyTorch training script
cd C:\Users\nonna\Downloads\PyroCast
python spread_model/05_train_pytorch.py
```

**Note:** The PyTorch training script (`05_train_pytorch.py`) has been created and will automatically use DirectML when available.

---

## Option 3: Cloud GPU (Quick temporary solution)

### Google Colab (Free GPU)
1. Upload your tfrecord files to Google Drive
2. Use Colab notebook with T4 GPU (free tier)
3. Mount Drive and run training
4. Download trained model

**Note:** Free tier has usage limits but perfect for testing.

---

## Recommendation
Use **WSL2 + ROCm** - it's the cleanest solution that keeps your existing TensorFlow code and provides native GPU acceleration on Windows.
