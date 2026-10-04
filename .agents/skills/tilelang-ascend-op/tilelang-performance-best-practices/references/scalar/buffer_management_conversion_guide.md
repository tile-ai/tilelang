# PTO Buffer Management

Use `T.alloc_shared` for UB, `T.alloc_fragment` for register- or thread-local storage, and `T.alloc_l1` plus `T.alloc_l0a`/`T.alloc_l0b`/`T.alloc_l0c` for the Cube path. Use `T.copy` and `T.dual_copy` for memory-hierarchy transfers validated in the target repository.

The physical byte count of a pipelined tile is the element count×bytes per dtype×number of versions; then add padding, resident data, and a safety margin. Resident weights and reduction state must not be multi-versioned. Rely on backend merge/reuse only for buffers with nonoverlapping lifetimes, and confirm the result in the generated code.

For tail tiles, access only valid elements in GM, while still allocating UB for the complete register footprint. Reinitialize mutable state for every task.
