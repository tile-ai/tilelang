### Set environment variables for intranode usage ###
export TILELANG_USE_DISTRIBUTED=1
export NCCL_IB_DISABLE=1  # disable annoying IB-related logging from NCCL

### Optional ###
# export TILESCALE_USE_VMM=1  # Force using VMM instead of IPC
# export TILESCALE_MASTER_PORT=8080  # Set the master port