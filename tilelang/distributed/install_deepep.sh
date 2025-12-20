# This script is for automatic installation of DeepEP for CI workflow

# Ensure DeepEP is cloned into 3rdparty folder
if [ ! -d "3rdparty/DeepEP" ]; then
    echo "DeepEP is not cloned into 3rdparty folder"
    exit 1
fi

# Ensure NVSHMEM installed
if pip list | grep nvshmem > /dev/null 2>&1; then
    echo "nvshmem is already installed."
else
    pip install nvidia-nvshmem-cu12
fi

# Fix a bug of NVSHMEM path
export NVSHMEM_DIR=$(python -c "import site; print(site.getsitepackages()[0])")/nvidia/nvshmem
echo "NVSHMEM_DIR is set to $NVSHMEM_DIR"
ln -sf $NVSHMEM_DIR/lib/libnvshmem_host.so.3 $NVSHMEM_DIR/lib/libnvshmem_host.so

# Install DeepEP
cd 3rdparty/DeepEP
python setup.py install
cd -

# Validate
python -c "import deep_ep; print(deep_ep.__version__)"
echo "DeepEP is installed successfully. ✅"
