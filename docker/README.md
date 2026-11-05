The CUDA and ROCm Dockerfiles provide device-toolchain containers for building
the matched TileLang frontend, TIRx compiler provider and FFI distributions.
Each image builds the local checkout supplied as its build context; initialize
the submodules for the selected revision before building. The Dockerfile names
identify the original SDK image families, not a claim that every image has been
validated with this compiler revision. Check that its host compiler supports
C++20 and is compatible with the selected CUDA/ROCm toolkit.

Run from the TileLang repository root:

```bash
git submodule update --init --recursive
docker build -t tilelang_workspace -f docker/Dockerfile.cu124 .
# Or use the ROCm image with its matching PyTorch installation:
docker build -t tilelang_rocm -f docker/Dockerfile.rocm .
```

The build uses `maint/scripts/build_native_artifacts.py` in dependency order
and leaves wheels and logs under `/root/native-dist`. An editable frontend is
installed over the matched provider. Rebuild the provider and frontend after
native source edits; see the [installation guide](../docs/get_started/Installation.md).
An initialized source checkout, including the fixed FFI, must be present in the
context. The Dockerfiles do not fetch a moving TileLang branch during the build.

Run with the host's driver and device access:

```bash
# NVIDIA
docker run --rm -it --gpus all --shm-size=4G tilelang_workspace bash
# AMD
docker run --rm -it --device=/dev/kfd --device=/dev/dri --shm-size=4G tilelang_rocm
```

Building the container or generating device code does not establish numerical
correctness on unavailable hardware. Run the relevant backend tests on the
matching device after checking package/library identity with
`maint/scripts/verify_native_install.py`.
