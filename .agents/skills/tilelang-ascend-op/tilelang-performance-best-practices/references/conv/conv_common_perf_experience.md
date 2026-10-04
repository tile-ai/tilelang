# PTO Convolution Implementation Process

1. Use the PyTorch reference to establish output-shape, padding, and bias semantics.
2. Select either a direct primitive or tiled im2col+GEMM in the Python factory.
3. Assign each task one output spatial/Cout tile. Generate the A tile in UB/L1 from the valid window and write 0 for out-of-bounds padding.
4. Load weights into L1 with layout [Cout,K], call T.gemm(..., transpose_B=True), and accumulate in fp32 L0C.
5. Fuse bias and activation before converting the output tile, preserving the reference operation order.
6. For depthwise convolution, combine tasks across multiple groups, but never mix K accumulation from different groups.

The tile, K expansion, resident weights, and pipeline stages must fit the L1/L0/UB budgets. Cover every output edge and group tail. Search performance using the actual shapes instead of reusing fixed hardware heuristics.
