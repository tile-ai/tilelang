#!/bin/bash

nvcc -shared -rdc=true -Xcompiler -fPIC test_nvshmem_example.cu -o libnvshmem_example.so \
    -gencode=arch=compute_80,code=sm_80 \
    -I/home/aiscuser/miniconda3/envs/tilelang/include \
    -I/home/aiscuser/cy/tilelang/3rdparty/nvshmem//build/src/include \
    -L/home/aiscuser/cy/tilelang/3rdparty/nvshmem//build/src/lib \
    -lnvshmem_host -lnvshmem_device

/home/aiscuser/cy/tilelang/3rdparty/nvshmem/scripts/build/bin/nvshmrun -n 8 python test_nvshmem_example.py
