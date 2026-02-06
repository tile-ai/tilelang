# This script is for automatic installation of DeepEP
# Please run this script in the root directory of TileScale

# Detect current CUDA arch and export TORCH_CUDA_ARCH_LIST using torch

CUDA_ARCH_LIST=$(python -c 'import torch; print(".".join(str(x) for x in torch.cuda.get_device_capability()) if torch.cuda.is_available() else "")' 2>/dev/null)

if [ -z "$CUDA_ARCH_LIST" ]; then
    echo "torch.cuda not available or failed to detect CUDA arch"
    exit 1
else
    export TORCH_CUDA_ARCH_LIST="$CUDA_ARCH_LIST"
    echo "TORCH_CUDA_ARCH_LIST set to $TORCH_CUDA_ARCH_LIST"
fi

# Ensure DeepEP is cloned into 3rdparty folder
if [ ! -d "3rdparty/DeepEP" ]; then
    echo "DeepEP is not cloned into 3rdparty folder"
    exit 1
fi

# Ensure NVSHMEM installed
if pip list | grep nvshmem > /dev/null 2>&1; then
    echo "NVSHMEM is already installed."
else
    pip install nvidia-nvshmem-cu12
fi

# Fix a bug of NVSHMEM linking
export NVSHMEM_DIR=$(python -c "import site; print(site.getsitepackages()[0])")/nvidia/nvshmem
echo "NVSHMEM_DIR is set to $NVSHMEM_DIR"
ln -sf $NVSHMEM_DIR/lib/libnvshmem_host.so.3 $NVSHMEM_DIR/lib/libnvshmem_host.so

# Install DeepEP
cd 3rdparty/DeepEP
python setup.py install
cd -

# Validate
python -c "import deep_ep"
if [ $? -ne 0 ]; then
    echo "Failed to import deep_ep"
    exit 1
fi
echo "DeepEP is installed successfully. ✅"
