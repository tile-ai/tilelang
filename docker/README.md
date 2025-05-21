To ease the process of installing all the dependencies, we provide a Dockerfile and a simple guideline to build a Docker image with all of above installed. The Docker image is built on top of Ubuntu 20.04, and it contains all the dependencies required to run the experiments. We only provide the Dockerfile for NVIDIA GPU, and the Dockerfile for AMD GPU will be provided upon request.

```bash
git clone --recursive https://github.com/tile-ai/tilelang TileLang
cd TileLang/docker
# build the image, this may take a while (around 10+ minutes on our test machine)
# replace the version number cu124 with the one you want to use
# replace .cu** with .rocm for AMD GPU
docker build -t tilelang_workspace -f Dockerfile.cu124 .
# run the container
# if it's nvidia
docker run -it --cap-add=SYS_ADMIN --network=host --gpus all --cap-add=SYS_PTRACE --shm-size=4G --security-opt seccomp=unconfined --security-opt apparmor=unconfined --name tilelang_test tilelang_workspace bash
# if it's amd
docker run -it --cap-add=SYS_ADMIN --network=host --device=/dev/kfd --device=/dev/dri  --cap-add=SYS_PTRACE --shm-size=4G --security-opt seccomp=unconfined --security-opt apparmor=unconfined --name tilelang_test tilelang_workspace bash
```
