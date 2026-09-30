#!/usr/bin/env bash
# One-time WSL2 toolchain setup for the RTX 3050 flow (needs sudo).
# Installs: build tools, CMake, CUDA toolkit 12.8 (nvcc + cuBLAS, NO driver -- the
# driver stays on the Windows side and is exposed to WSL automatically) and the
# bundled Nsight Compute (ncu).
set -e

sudo apt-get update
sudo apt-get install -y build-essential cmake wget

# NVIDIA CUDA repo for WSL-Ubuntu (this variant intentionally ships no driver).
KR=/usr/share/keyrings/cuda-archive-keyring.gpg
if [ ! -f "$KR" ]; then
  wget -qO /tmp/cuda-keyring.deb \
    https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
  sudo dpkg -i /tmp/cuda-keyring.deb
  sudo apt-get update
fi

# Toolkit 12.8 matches the installed driver's CUDA 12.8. ncu is bundled.
sudo apt-get install -y cuda-toolkit-12-8

grep -q '/usr/local/cuda/bin' ~/.bashrc || \
  echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc

echo
echo "Setup done. Run:  source ~/.bashrc  then check:  nvcc --version ; ncu --version"
