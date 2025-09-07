python3 examples/amd/example_amd_flash_attn_fwd.py \
    --batch 2 \
    --heads 16 \
    --seq_len 4096 \
    --dim 128 \
    --is_causal

/root/composable_kernel/build/bin/tile_example_fmha_fwd  \
-b=2 -h=16 -s=4096 -d=128 -mask=t -v=1 -warmup=5 -repeat=200

python3 examples/amd/example_amd_flash_attn_bwd.py \
    --batch 2 \
    --h 16 \
    --n_ctx 4096 \
    --d_head_qk 128 \
    --d_head_v 128 \
    --groups 16 \
    --causal True

/root/composable_kernel/build/bin/tile_example_fmha_bwd -b=2 -h=16 -s=4096 -d=128 -mask=t -warmup=5 -repeat=20