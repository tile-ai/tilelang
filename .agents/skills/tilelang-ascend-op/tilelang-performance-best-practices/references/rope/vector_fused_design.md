# RoPE SIMD Fusion

## Implementation Flow

Place load, rotation, and store in the same tile without writing intermediate results to GM. For half-split layout, read left/right and compute left*cos-right*sin and right*cos+left*sin. For interleaved layout, read adjacent even/odd elements and apply the same formulas. Derive the code directly from `references/rope/code/rope_vf_common.py`.

If the caller stores sin/cos along the half dimension, use T.copy directly. If they are stored along the full dimension, the factory must define the index mapping explicitly. The current baseline uses T.SimtVF; a T.SimdVF version may replace it only after eliminating the backend's `unsupported scalar instruction`, passing lowering, and passing performance A/B testing.

## Gates

Promote computation to fp32, and cover extreme angles, repeated positions, non-integral SIMD tails, and aliasing. Compare GM bytes, lane utilization, register pressure, and end-to-end latency before and after fusion.
