# Gated Delta Net (GDN) kernel implementation with TileLang

## Requirement

- TileLang: `0.1.5+17fafc1b3026d910a83eb8052fdf811ba56be0b1`
- Triton: `3.3.0` (used for comparison)
- FLA: commit `f03cb3ae` (used for comparison)

## Get started

 The [chunk_delta_h](common/chunk_delta_h.py) implements the most critical forward kernel of GDN. It's a good start to understand the GDN logic and the TileLang optimization.

## Acknowledgments

This kernel was developed by Yu Cheng and Zhengju Tang following in-depth discussions with Xiaomi's LLM-Core Team (MiMo).
