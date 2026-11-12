# Communication and PTO GEMM Pipeline Balancing

Partition the M or token dimension into chunks. Each chunk has a communication time `C_i` and a GEMM time `G_i`; the host wrapper forms a pipeline using an asynchronous collective and two streams. Select the chunk size through a systematic search rather than hard-coding an empirical ratio.

Every candidate must satisfy the following constraints: each element belongs to exactly one chunk; the communication buffer lifetime covers the asynchronous operation; GEMM reads a chunk only after the corresponding event completes; output write regions do not overlap; and the pipeline waits for all operations before returning. Determine the ordering of long and short chunks from measured `C_i/G_i` values.

Correctness tests must cover every rank and all chunk tails. The performance report must include the serial baseline, total communication time, total compute time, end-to-end time, overlap ratio, additional launches, and extra memory usage. Reject any approach that optimizes only a local stage while regressing end-to-end performance.
