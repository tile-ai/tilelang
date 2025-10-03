# Project 1 Prompt — TT Backend MVP

- based on the high level technical plan in README.md and docs/tenstorrent, make a detailed plan to implement MVP for matrix multiplication, but default sharding, default schedule, default bfloat16, DRAM tensors. place the project plan in docs/tenstorrent/project_1.md (make a detailed markdown file). the plan should include all the steps to modify transforms, passes, etc to allow dry test of basic matmul, ie we can genereate metalium host code and kernels (reader, compute, writer).
- feedback: the default sharding should be "interleaved tensors", the new TensorAccessor in TT-Metalium supports this.
- feedback: the default sharding should be "interleaved tensors", the new TensorAccessor in TT-Metalium supports this, search TT-Metalium repo if you're not familiar.
- feedback: add the TileLang MVP Gemm test, there needs to be a python test that needs to pass after all workstreams are done.
- feedback: for each workstream , there needs be more details on which transforms/files need to be modified, if we're adding new transform/files, justify why can't modify/augment existing ones.
- feedback: for each workstream, there need to be dedicated unit tests.

