# Run Examples

## Examples without NVSHMEM

Before running, enable TileLang’s distributed mode:

```bash
export TILELANG_USE_DISTRIBUTED=1
```
Then start an example directly with Python:
```bash
 python examples/distributed/primitives/example_put_warp.py
```

## Examples using NVSHMEM APIs

Use the provided launcher `tilelang/distributed/launch.sh` to start programs that use the NVSHMEM API. For example, to run with 2 GPUs:
```bash
GPUS=2 ./tilelang/distributed/launch.sh examples/distributed/example_allgather.py
```
You can change GPUS to the number of local GPUs you want to use. The launcher will set the required environment variables and invoke `torch.distributed.run`.
