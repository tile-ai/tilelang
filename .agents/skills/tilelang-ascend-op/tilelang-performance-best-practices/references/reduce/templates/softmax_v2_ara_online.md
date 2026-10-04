# ARA Online Softmax

## Applicability

The logical input layout is `[A1, R, A0]`, with softmax applied along the middle axis `R`. Use this design when `R × tile_A0` cannot fit entirely in UB and eliminating one full input traversal offsets the overhead of online state updates.

## Data Flow

Each task exclusively owns one `(a1, a0_tile)`. `running_max[A0]` and `running_sum[A0]` use fp32 and remain resident in UB. The first pass streams over R chunks:

```text
chunk_max = max(x_chunk, axis=R)
m_new     = max(m_old, chunk_max)
l_new     = l_old * exp(m_old - m_new)
          + sum(exp(x_chunk - m_new), axis=R)
```

The equivalent chunk-combination form is:

```text
chunk_sum = sum(exp(x_chunk - chunk_max), axis=R)
l_new = l_old * exp(m_old - m_new) + chunk_sum * exp(chunk_max - m_new)
```

The second pass rereads the input and outputs `exp(x - running_max) / running_sum`. Fill max lanes in the tail R chunk with negative infinity and sum lanes with zero. Mask the A0 tail so that padding cannot change the state.

See `dav310/softmax_v2_ara_online.py` for Python strategy data and the scalar combination formula. Derive the PTO SIMD implementation from a validated `T.SimdVF` reduce/exp instruction combination in the `current repository`. If strided R chunks cannot form efficient bursts, prefer fusing layout conversion over elementwise GM access.

## Accuracy and Performance Gates

- Compare against fp32 `torch.softmax`, covering `R=tile±1`, extremely long R, repeated maxima, extremely large positive and negative values, and the NaN/Inf semantics required by the interface.
- Compare the total GM bytes and latency of the two-pass online design against the three-pass recompute design.
- Report the two fp32 state arrays in UB, input/output tiles, buffer versions, and safety margin.
- Select the online design only when all targeted PTO accuracy tests pass and end-to-end performance is faster.
