

# MHA forward
python examples/flash_attention_sm100/mha_fwd_bshd.py --variant ss --seq_len 8192
python examples/flash_attention_sm100/mha_fwd_bshd.py --variant ts --seq_len 8192
python examples/flash_attention_sm100/mha_fwd_bshd.py --variant wasp --seq_len 8192

# GQA forward
python examples/flash_attention_sm100/gqa_fwd_bshd.py --variant ss --groups 4 --seq_len 8192
python examples/flash_attention_sm100/gqa_fwd_bshd.py --variant ts --groups 4 --seq_len 8192
python examples/flash_attention_sm100/gqa_fwd_bshd.py --variant wasp --groups 4 --seq_len 8192


python examples/flash_attention/example_mha_fwd_bshd.py  --batch 2 --heads 4 --seq_len 8192 
python examples/flash_attention/example_gqa_fwd_bshd.py  --batch 2 --heads 4 --seq_len 8192 --groups 4



nvidia@tegra-ubuntu:~/wahao/tilelang$ bash examples/flash_attention_sm100/perf.sh 
Loading tilelang libs from dev root: /home/nvidia/wahao/tilelang/build
Namespace(batch=2, heads=4, seq_len=8192, dim=128, is_causal=False, variant='ss')
=== Blackwell Flash Attention (SS) ===
batch=2, heads=4, seq_len=8192, dim=128, causal=False
Correctness check passed.
Blackwell (ss): 8.05 ms
Blackwell (ss): 34.13 TFlops
Loading tilelang libs from dev root: /home/nvidia/wahao/tilelang/build
Namespace(batch=2, heads=4, seq_len=8192, dim=128, is_causal=False, variant='ts')
=== Blackwell Flash Attention (TS) ===
batch=2, heads=4, seq_len=8192, dim=128, causal=False
Correctness check passed.
Blackwell (ts): 5.72 ms
Blackwell (ts): 48.03 TFlops
Loading tilelang libs from dev root: /home/nvidia/wahao/tilelang/build
Namespace(batch=2, heads=4, seq_len=8192, dim=128, is_causal=False, variant='wasp')
=== Blackwell Flash Attention (WASP) ===
batch=2, heads=4, seq_len=8192, dim=128, causal=False
2026-03-11 08:28:39  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `main` with `out_idx=[3]`
2026-03-11 08:28:51  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `main`
Correctness check passed.
Blackwell (wasp): 9.64 ms
Blackwell (wasp): 28.51 TFlops
Loading tilelang libs from dev root: /home/nvidia/wahao/tilelang/build
=== Blackwell GQA Forward (SS) ===
batch=2, heads=4, head_kv=4, groups=1, seq_len=8192, dim=128, causal=False
Correctness check passed.
Blackwell GQA fwd (ss): 8.05 ms
Blackwell GQA fwd (ss): 34.15 TFlops
Loading tilelang libs from dev root: /home/nvidia/wahao/tilelang/build
=== Blackwell GQA Forward (TS) ===
batch=2, heads=4, head_kv=4, groups=1, seq_len=8192, dim=128, causal=False
Correctness check passed.
Blackwell GQA fwd (ts): 5.72 ms
Blackwell GQA fwd (ts): 48.04 TFlops
Loading tilelang libs from dev root: /home/nvidia/wahao/tilelang/build
=== Blackwell GQA Forward (WASP) ===
batch=2, heads=4, head_kv=1, groups=4, seq_len=8192, dim=128, causal=False
2026-03-11 08:29:59  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `main` with `out_idx=[3]`
2026-03-11 08:30:11  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `main`
Correctness check passed.
Blackwell GQA fwd (wasp): 10.08 ms
Blackwell GQA fwd (wasp): 27.27 TFlops
Loading tilelang libs from dev root: /home/nvidia/wahao/tilelang/build
2026-03-11 08:30:32,289 WARNING:Tunable parameters ['block_M', 'block_N', 'num_stages', 'threads'] already provided during auto-tuning. Skipping compilation and using direct JIT
All checks pass.
Ref: 44.71 ms
Ref: 6.15 TFlops
Tile-lang: 11.31 ms
Tile-lang: 24.30 TFlops
Loading tilelang libs from dev root: /home/nvidia/wahao/tilelang/build
2026-03-11 08:30:54,638 WARNING:Tunable parameters ['block_M', 'block_N', 'num_stages', 'threads'] already provided during auto-tuning. Skipping compilation and using direct JIT
2026-03-11 08:30:57  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `main` with `out_idx=[3]`
2026-03-11 08:31:03  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `main`
All checks pass.
Ref: 45.17 ms
Ref: 6.09 TFlops
Tile-lang: 10.63 ms
Tile-lang: 25.86 TFlops


