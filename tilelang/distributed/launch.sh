#!/bin/bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib:~/.local/lib/

export TILELANG_USE_NVSHMEM=1  # enable TileLang distributed mode
export TILELANG_USE_DISTRIBUTED=1  # enable TileLang distributed mode
export NVSHMEM_BOOTSTRAP_MPI_PLUGIN=nvshmem_bootstrap_torch.so
export NVSHMEM_DISABLE_CUDA_VMM=1  # moving from cpp to shell
export CUDA_DEVICE_MAX_CONNECTIONS=1

# set default communication env vars
export BYTED_TORCH_BYTECCL=O0
export NCCL_IB_TIMEOUT=${NCCL_IB_TIMEOUT:=23}

# set nccl log level
export NCCL_DEBUG=${NCCL_DEBUG:="WARN"}  # set env var. `NCCL_DEBUG` to expected NCCL log level
# Choices: [VERSION, WARN(default), INFO, TRACE],

# set launch configurations
nproc_per_node=${GPUS:=$(nvidia-smi --list-gpus | wc -l)}  # set env var. `GPUS` to # of GPUs per node
nnodes=${NODES:=1}  # set env var. `NODES` to # of nodes
node_rank=${NODE_RANK:=0}  # set env var. `NODE_RANK` to the rank of current node

master_addr=${ARNOLD_WORKER_0_HOST:="127.0.0.1"}
if [ -z ${ARNOLD_WORKER_0_PORT} ]; then
  master_port="8361"
else
  master_port=$(echo "$ARNOLD_WORKER_0_PORT" | cut -d "," -f 1)
fi
additional_args="--rdzv_endpoint=${master_addr}:${master_port}"
IB_HCA=mlx5


export NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:=3}
export NVSHMEM_IB_GID_INDEX=3

# set whether to use memory check
memcheck=${MEMCHECK:=0}  # set env var. `MEMCHECK` to 1 to enable memory check via compute-sanitizer
# This is especially useful for debugging memory issues, e.g. CUDA misalignment errors and TMA stuff.

PYTHON_EXEC="$(which python)"
CMD="${PYTHON_EXEC} -m torch.distributed.run \
  --node_rank=${node_rank} \
  --nproc_per_node=${nproc_per_node} \
  --nnodes=${nnodes} \
  ${TILELANG_EXTRA_TORCHRUN_ARGS} ${additional_args} $@"

if [ ${memcheck} -eq 1 ]; then
    CMD="compute-sanitizer --tool memcheck ${CMD}"
fi

echo ${CMD}
${CMD}

ret=$?
exit $ret
