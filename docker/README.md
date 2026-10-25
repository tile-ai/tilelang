Build these images from the TileLang repository root, not from `docker/`.
They copy the current checkout and install the independently distributed,
patched TIRx dependency before building TileLang. They do not clone a different
TileLang revision or restore the former TVM submodule.

Provide `TVM_TIRX_WHEEL` as a URL or a path inside the build context. The wheel
must match `TVM_TIRX_VERSION`, the container platform, and the selected CUDA or
ROCm runtime. For source provisioning, use `TVM_TIRX_REPOSITORY` containing the
paired revision recorded in `TVM_TIRX_COMMIT`; the unpatched upstream baseline
is insufficient. See `MIGRATION_TIRX.md` for the dependency contract and C++20
toolchain requirements. Older image variants also need a C++20-capable compiler
compatible with their SDK.

```bash
git clone --recursive https://github.com/tile-ai/tilelang TileLang
cd TileLang
# build the image, this may take a while (around 10+ minutes on our test machine)
# replace the version number cu124 with the one you want to use
# replace .cu** with .rocm for AMD GPU
docker build -t tilelang_workspace -f docker/Dockerfile.cu124 \
  --build-arg TVM_TIRX_WHEEL=wheels/apache_tvm_tirx-PAIRED_VERSION-py3-none-PLATFORM.whl .
# run the container
# if it's nvidia
docker run -it --cap-add=SYS_ADMIN --network=host --gpus all --cap-add=SYS_PTRACE --shm-size=4G --security-opt seccomp=unconfined --security-opt apparmor=unconfined --name tilelang_test tilelang_workspace bash
# if it's amd
docker run -it --cap-add=SYS_ADMIN --network=host --device=/dev/kfd --device=/dev/dri  --cap-add=SYS_PTRACE --shm-size=4G --security-opt seccomp=unconfined --security-opt apparmor=unconfined --name tilelang_test tilelang_workspace bash
```
