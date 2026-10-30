# Ascend Quantization Implementation References

Read only the implementation that matches the target dataflow:

- [per_token_cast_asc.py](code/per_token_cast_asc.py): per-token scaling, multi-path host dispatch, stochastic rounding, packed scale layouts, two-dimensional tiling, and tail handling.
- [per_channel_cast_with_psum_asc.py](code/per_channel_cast_with_psum_asc.py): expert-segmented dispatch from prefix sums, per-channel input scales, per-token output scales, and packed output layouts.
- [per_block_cast_lossless_asc.py](code/per_block_cast_lossless_asc.py): per-block scale merging, FP4/FP8 bit-level conversion, vector-register rearrangement, and packed scale storage.

These `_asc.py` files contain complete host and kernel implementation structures for source reading. They are not launch examples or maturity-rated templates. During generation, reuse the matching dtype/layout semantics, dispatch boundaries, buffer organization, and tail rules. During tuning, inspect which host branch is active before comparing tile sizes, task mapping, scale reuse, buffer versions, conversion order, or packed-write layouts.
