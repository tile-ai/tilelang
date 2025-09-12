/root/miniconda3/envs/py312/bin/python3 examples/amd/example_amd_flash_attn_fwd.py \
    --batch 2 \
    --heads 16 \
    --seq_len 4096 \
    --dim 128 \
    --is_causal \
    --groups 2

/root/composable_kernel/build/bin/tile_example_fmha_fwd  \
-b=2 -h=16 -s=4096 -d=128 -mask=t -v=1 -warmup=5 -repeat=20
