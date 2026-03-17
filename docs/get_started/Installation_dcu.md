# Installation for DCU
## Building from Source
```bash
mkdir -p build
cd build
cmake .. -DUSE_CUDA=OFF -DUSE_ROCM=ON
make -j
```

```bash
export PYTHONPATH=/path/to/tilelang:$PYTHONPATH
python -c "import tilelang; print(tilelang.__version__)"
```

## Other Tips
### Missing tvm_ffi Module
If you encounter the error ModuleNotFoundError: No module named 'tvm_ffi', it means the TVM foreign function interface package was not installed. This often happens if the submodules were built manually. Fix it by running:
```
# Navigate to the tvm_ffi directory
cd 3rdparty/tvm/3rdparty/tvm_ffi

# Install the package in editable mode
pip install .

# Return to the project root
cd ../../../..
```
### DTK Path Configuration
If you encounter errors related to DTK path detection (e.g., hipcc not found or failure to retrieve GPU architecture), you may need to manually specify the DTK installation path in the source code.
Locate the file tilelang/contrib/rocm.py and modify the default value of the rocm_path parameter in the get_rocm_arch function (around line 231):

```
# File: tilelang/contrib/rocm.py

# Change from:
def get_rocm_arch(rocm_path="/opt/rocm"):
    ...

# To (for Hygon DCU environments):
def get_rocm_arch(rocm_path="/opt/dtk"): 
    ...
```