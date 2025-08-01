/bin/python /workspace/tilelang/examples/amd/example_amd_flash_attn_fwd_k_block.py     --batch 2     --seq_len 4096     --dim 128     --heads 16     --groups 8     --is_causal
# /workspace/aiter/3rdparty/composable_kernel/build/bin/tile_example_fmha_fwd \
#     -b=2 \
#     -s=4096 \
#     -d=128 \
#     -h=16 \
#     -h_k=2 \
#     -prec=fp16 \
#     -mask=t \
#     -v=0

# hipcc -std=c++17 -fPIC --offload-arch=gfx942 -S \
#   ~/.tilelang/cache/0dad5010ab3d8e01b2a86e56208af334ad62865b92cb03038f9df84fda8d8a99/kernel.cu \
#   -o ./kernel.s \
#   -I/workspace/tilelang/3rdparty/composable_kernel/include \
#   -I/workspace/tilelang/src