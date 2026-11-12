# Local GEMM and Communication Overlap

Data local to the current rank requires no communication. The Python wrapper first obtains the local slice from the input and invokes PTO GEMM while launching a collective for remote data on another stream. The local result writes to its unique slice of the final output and must not overlap any remote chunk.

For communication-before-compute scenarios, prioritize overlapping local GEMM with the first collective. For compute-before-communication scenarios, local GEMM may be processed after pipelining remote chunks. Use the target runtime's official interfaces for every stream/event, and make the wrapper wait for required events before exposing the result.

Validate a coverage bitmap for local/nonlocal slices, rank boundaries, `world size=1`, and a multi-rank reference. Report local GEMM, collective, wait gap, and end-to-end latency in performance results.
